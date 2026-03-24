"""Price-trigger strategy: auto-buy YES when a bucket hits the target price.

Based on historical analysis of 677 settled Austin temperature markets:
- Buckets reaching $0.85 win 95.1% of the time (n=41, EV=+$0.10/contract)
- Buckets reaching $0.80 win 85.2% of the time (n=54, EV=+$0.05/contract)

Strategy: Poll Kalshi API every N seconds. When any bucket's YES ask price
crosses below our buy threshold, buy it.

This is data-driven and deterministic — no LLM, no weather models, no forecasts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

from weather_bets import config
from weather_bets.executor import WeatherExecutor
from weather_bets.kalshi_weather import fetch_weather_markets
from weather_bets.models import CityConfig

logger = logging.getLogger(__name__)

# ── Strategy Configuration ──────────────────────────────────────────────

@dataclass
class TriggerConfig:
    """Configuration for a price trigger level."""
    buy_at: float              # Buy when YES ask <= this price (e.g., 0.85)
    historical_win_rate: float # From analysis (e.g., 0.951)
    max_contracts: int = 50    # Max contracts per trigger per ticker
    enabled: bool = True

# Trigger levels from our historical analysis
TRIGGERS = [
    TriggerConfig(buy_at=0.85, historical_win_rate=0.951, max_contracts=50),
    TriggerConfig(buy_at=0.80, historical_win_rate=0.852, max_contracts=30),
    # Add more tiers if desired:
    # TriggerConfig(buy_at=0.75, historical_win_rate=0.774, max_contracts=20),
]

POLL_INTERVAL_SECONDS = 60   # Check prices every 60 seconds
LOG_FILE = Path(__file__).parent / "data" / "trigger_log.json"


@dataclass
class TriggerState:
    """Tracks which tickers have been bought at which levels."""
    bought: dict = field(default_factory=dict)  # {ticker: {price_level: order_id}}
    session_trades: int = 0
    session_pnl: float = 0.0
    started_at: str = ""


class PriceTriggerStrategy:
    """Polls Kalshi prices and buys when targets are hit."""

    def __init__(
        self,
        executor: WeatherExecutor,
        city: CityConfig,
        triggers: list[TriggerConfig] | None = None,
        execution_mode: str = "dry",
    ):
        self.executor = executor
        self.city = city
        self.triggers = triggers or TRIGGERS
        self.mode = execution_mode
        self.state = TriggerState(started_at=datetime.now(timezone.utc).isoformat())
        self._running = False

    def already_bought(self, ticker: str, price_level: float) -> bool:
        """Check if we've already bought this ticker at this price level today."""
        return ticker in self.state.bought and price_level in self.state.bought[ticker]

    async def check_and_buy(self) -> list[dict]:
        """One poll cycle: fetch prices, check triggers, execute if hit."""
        actions = []

        try:
            # Get today's date
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Fetch current bucket prices
            buckets = await fetch_weather_markets(self.city, target_date=today)
            if not buckets:
                logger.debug("[Trigger] No buckets found for today")
                return actions

            for bucket in buckets:
                for trigger in self.triggers:
                    if not trigger.enabled:
                        continue

                    # The winning bucket's price RISES during the day
                    # (from ~$0.20 toward $1.00 as certainty increases).
                    # We buy when it first reaches our threshold on the way UP.
                    if bucket.yes_price <= 0 or bucket.yes_price < trigger.buy_at:
                        continue

                    # Already bought this one?
                    if self.already_bought(bucket.ticker, trigger.buy_at):
                        continue

                    # ── Risk Management: drawdown floor + 50% rule ──
                    remaining_budget = await self.executor.get_betting_budget()

                    if remaining_budget == -1.0:
                        logger.warning("[Trigger] 🛑 DRAWDOWN FLOOR HIT — all betting suspended")
                        return actions  # Stop entire cycle

                    if remaining_budget <= 0.50:
                        logger.warning("[Trigger] ⛔ 50% budget exhausted — skipping")
                        return actions

                    # Dynamic sizing: use configured % of balance
                    balance = await self.executor.get_balance()
                    bet_size = min(
                        balance * config.BET_SIZE_PCT,
                        remaining_budget,  # Never exceed remaining budget
                    )
                    contracts = min(
                        trigger.max_contracts,
                        max(1, int(bet_size / bucket.yes_price)),
                    )
                    cost = contracts * bucket.yes_price

                    # Final check: does this trade fit within remaining budget?
                    if cost > remaining_budget:
                        contracts = max(1, int(remaining_budget / bucket.yes_price))
                        cost = contracts * bucket.yes_price
                        logger.info(
                            f"[Trigger] 📉 Scaled to {contracts} contracts "
                            f"(${cost:.2f}) to fit 50% budget"
                        )

                    # Kalshi fee: roundup(0.07 * C * P * (1-P))
                    import math
                    fee_per_contract = math.ceil(7 * bucket.yes_price * (1 - bucket.yes_price)) / 100
                    total_fee = fee_per_contract * contracts

                    ev_per_contract = trigger.historical_win_rate - bucket.yes_price
                    ev_total = ev_per_contract * contracts
                    roi = ev_per_contract / bucket.yes_price

                    action = {
                        "ticker": bucket.ticker,
                        "bucket": bucket.label,
                        "yes_price": bucket.yes_price,
                        "trigger": trigger.buy_at,
                        "historical_wr": trigger.historical_win_rate,
                        "contracts": contracts,
                        "cost": round(cost, 2),
                        "fee": round(total_fee, 4),
                        "ev_per_contract": round(ev_per_contract, 4),
                        "ev_total": round(ev_total, 4),
                        "roi": f"{roi:+.1%}",
                        "time": datetime.now(timezone.utc).isoformat(),
                        "mode": self.mode,
                        "status": "pending",
                    }

                    logger.info(
                        f"[Trigger] 🎯 HIT! {bucket.ticker} ({bucket.label}) "
                        f"YES=${bucket.yes_price:.2f} ≤ ${trigger.buy_at:.2f} trigger | "
                        f"WR={trigger.historical_win_rate:.1%} | "
                        f"{contracts}x @ ${bucket.yes_price:.2f} = ${cost:.2f} | "
                        f"EV=${ev_total:+.4f} ({roi:+.1%})"
                    )

                    if self.mode == "dry":
                        logger.info(f"[Trigger] 🔸 DRY RUN — would buy {contracts}x {bucket.ticker}")
                        action["status"] = "dry_run"
                    elif self.mode == "live":
                        # Place the actual order via executor
                        order_id = await self._place_order(
                            bucket.ticker, contracts, bucket.yes_price
                        )
                        if order_id:
                            action["status"] = "filled"
                            action["order_id"] = order_id
                            logger.info(f"[Trigger] ✅ ORDER PLACED: {order_id}")
                        else:
                            action["status"] = "failed"
                            logger.error(f"[Trigger] ❌ Order FAILED for {bucket.ticker}")
                    elif self.mode == "semi":
                        logger.info(f"[Trigger] ⏸️ SEMI — queued for approval: {bucket.ticker}")
                        action["status"] = "queued"

                    # Track that we've processed this ticker at this level
                    if bucket.ticker not in self.state.bought:
                        self.state.bought[bucket.ticker] = {}
                    self.state.bought[bucket.ticker][trigger.buy_at] = action.get("order_id", "pending")
                    self.state.session_trades += 1

                    actions.append(action)

        except Exception as e:
            logger.error(f"[Trigger] Poll error: {e}", exc_info=True)

        return actions

    async def _place_order(self, ticker: str, contracts: int, price: float) -> str | None:
        """Place a limit buy order on Kalshi. Returns order_id or None."""
        if not self.executor._http or not self.executor._private_key:
            logger.error("[Trigger] Executor not authenticated")
            return None

        try:
            price_cents = int(price * 100)
            path = "/portfolio/orders"
            full_path = f"/trade-api/v2{path}"
            order_payload = {
                "ticker": ticker,
                "client_order_id": str(uuid.uuid4()),
                "type": "limit",
                "action": "buy",
                "side": "yes",
                "count": contracts,
                "yes_price": price_cents,
            }

            headers = self.executor._auth_headers("POST", full_path)
            resp = await self.executor._http.post(path, json=order_payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            order_id = data.get("order", {}).get("order_id", "unknown")

            # Save to trade log
            from weather_bets.trade_log import save_trade
            save_trade({
                "id": order_id,
                "ticker": ticker,
                "city": self.city.code,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "bucket": "",
                "side": "yes",
                "price": price,
                "qty": contracts,
                "cost": round(price * contracts, 2),
                "edge": round((0.951 - price) * 100, 1),  # approximate
                "our_prob": 95.1,
                "market_prob": price * 100,
                "reasoning": f"Price trigger: YES=${price:.2f} <= threshold, "
                             f"historical WR=95.1%, EV=${0.951 - price:+.4f}/contract",
                "mode": "live",
                "time": datetime.now(timezone.utc).isoformat(),
                "settled": False,
                "won": None,
                "pnl": None,
            })

            return order_id

        except Exception as e:
            logger.error(f"[Trigger] Order placement failed: {e}", exc_info=True)
            return None

    async def run(self, poll_interval: int = POLL_INTERVAL_SECONDS):
        """Main loop: poll prices and execute triggers."""
        self._running = True
        logger.info(
            f"[Trigger] 🚀 Starting price trigger strategy\n"
            f"  City: {self.city.name}\n"
            f"  Mode: {self.mode}\n"
            f"  Triggers: {[f'${t.buy_at:.2f} (WR={t.historical_win_rate:.1%})' for t in self.triggers]}\n"
            f"  Poll interval: {poll_interval}s"
        )

        all_actions = []

        while self._running:
            actions = await self.check_and_buy()
            if actions:
                all_actions.extend(actions)
                self._save_log(all_actions)

            await asyncio.sleep(poll_interval)

    def stop(self):
        """Stop the polling loop."""
        self._running = False
        logger.info(
            f"[Trigger] 🛑 Stopped. Session: {self.state.session_trades} trades"
        )

    def _save_log(self, actions: list[dict]):
        """Save action log to disk."""
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text(json.dumps(actions, indent=2))


async def main():
    """Run the price trigger strategy as a standalone script."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse mode from args
    mode = "dry"  # default to dry run
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if mode not in ("dry", "semi", "live"):
        print(f"Usage: python -m weather_bets.price_trigger [dry|semi|live]")
        print(f"  dry  = log what would happen (default)")
        print(f"  semi = queue for approval")
        print(f"  live = execute trades automatically")
        sys.exit(1)

    # Initialize executor (needed for auth + balance)
    executor = WeatherExecutor()
    await executor.initialize()

    city = config.CITIES["AUS"]
    strategy = PriceTriggerStrategy(
        executor=executor,
        city=city,
        execution_mode=mode,
    )

    try:
        await strategy.run()
    except KeyboardInterrupt:
        strategy.stop()
    finally:
        await executor.close()


if __name__ == "__main__":
    asyncio.run(main())
