"""Kalshi order executor for weather bets — places orders via Trade API v2."""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from datetime import datetime

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, utils

from weather_bets.config import (
    BET_SIZE_PCT,
    BET_SIZE_USD,
    DRAWDOWN_FLOOR_USD,
    EXECUTION_MODE,
    KALSHI_API_KEY_ID,
    KALSHI_BASE_URL,
    KALSHI_PRIVATE_KEY_PATH,
    MARKET_CONVICTION_THRESHOLD,
)
from weather_bets.models import BetRecommendation, PlacedBet

logger = logging.getLogger(__name__)

# Maximum percentage of available balance we'll put at risk across all open bets
MAX_BALANCE_USAGE_PCT = 0.50  # 50% rule — never bet more than half of available funds


class WeatherExecutor:
    """Places weather bet orders on Kalshi."""

    def __init__(self):
        self._http: httpx.AsyncClient | None = None
        self._private_key = None
        self._key_type: str = "unknown"  # "rsa" or "ec"
        self.placed_bets: list[PlacedBet] = []
        self._cached_balance: float | None = None
        self._balance_fetched_at: float = 0

    async def initialize(self) -> None:
        """Set up HTTP client and load private key."""
        self._http = httpx.AsyncClient(
            base_url=KALSHI_BASE_URL,
            timeout=15.0,
        )

        # Load private key for request signing
        try:
            from pathlib import Path
            key_path = Path(KALSHI_PRIVATE_KEY_PATH)
            if key_path.exists():
                key_data = key_path.read_bytes()
                self._private_key = serialization.load_pem_private_key(key_data, password=None)
                # Detect key type
                from cryptography.hazmat.primitives.asymmetric import rsa as rsa_mod, ec as ec_mod
                if isinstance(self._private_key, rsa_mod.RSAPrivateKey):
                    self._key_type = "rsa"
                elif isinstance(self._private_key, ec_mod.EllipticCurvePrivateKey):
                    self._key_type = "ec"
                logger.info(f"[Executor] Loaded Kalshi private key (type={self._key_type})")
            else:
                logger.warning(f"[Executor] Private key not found at {KALSHI_PRIVATE_KEY_PATH}")
        except Exception as e:
            logger.error(f"[Executor] Failed to load private key: {e}")

        # ── Sync trade log against Kalshi on startup ──
        await self._sync_trade_log_from_kalshi()

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """Sign a Kalshi API request using the private key (RSA-PSS or EC)."""
        if not self._private_key:
            return ""

        message = f"{timestamp_ms}{method}{path}"
        message_bytes = message.encode("utf-8")

        if self._key_type == "rsa":
            # Kalshi uses RSA-PSS with MGF1(SHA256)
            signature = self._private_key.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
        else:
            signature = self._private_key.sign(
                message_bytes,
                ec.ECDSA(hashes.SHA256()),
            )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict:
        """Generate auth headers for a Kalshi API request.
        
        path should be the full path (e.g., /trade-api/v2/portfolio/balance).
        Query params are stripped before signing per Kalshi docs.
        """
        timestamp_ms = int(time.time() * 1000)
        # Strip query params for signing
        sign_path = path.split("?")[0]
        signature = self._sign_request(method, sign_path, timestamp_ms)
        return {
            "KALSHI-ACCESS-KEY": KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
            "Content-Type": "application/json",
        }

    async def _sync_trade_log_from_kalshi(self) -> None:
        """
        On startup, fetch all orders from Kalshi and reconcile with local trade log.

        - Any Kalshi order not in local log gets added (catches missed writes)
        - Any local log entry with no matching Kalshi order_id gets flagged as unconfirmed
        - This ensures the local log always reflects reality, across restarts

        This is the single source of truth enforcement.
        """
        if not self._http or not self._private_key:
            logger.warning("[Executor] Cannot sync — not authenticated yet")
            return

        from weather_bets.trade_log import load_trades, TRADES_FILE
        import json as _json

        try:
            path = "/portfolio/orders"
            full_path = f"/trade-api/v2{path}"
            headers = self._auth_headers("GET", full_path)
            resp = await self._http.get(path, headers=headers)
            resp.raise_for_status()
            kalshi_orders = resp.json().get("orders", [])

            local_trades = load_trades()
            local_ids = {t.get("id") for t in local_trades if t.get("id")}
            kalshi_ids = {o.get("order_id") for o in kalshi_orders if o.get("order_id")}

            added = 0
            removed = 0

            # Add any Kalshi orders missing from local log
            for order in kalshi_orders:
                oid = order.get("order_id")
                if oid and oid not in local_ids:
                    # Reconstruct a trade record from the Kalshi order
                    yes_price = order.get("yes_price", 0) / 100.0
                    qty = order.get("count", 0)
                    trade_record = {
                        "id": oid,
                        "ticker": order.get("ticker", ""),
                        "city": order.get("ticker", "")[:3] if order.get("ticker") else "",
                        "date": "",
                        "bucket": "",
                        "side": order.get("side", "yes"),
                        "price": yes_price,
                        "qty": qty,
                        "cost": round(yes_price * qty, 2),
                        "edge": 0,
                        "our_prob": 0,
                        "market_prob": yes_price * 100,
                        "spread": "recovered",
                        "reasoning": "Recovered from Kalshi order history on startup",
                        "mode": "live",
                        "time": order.get("created_time", datetime.utcnow().isoformat()),
                        "settled": order.get("status") in ("settled", "canceled"),
                        "won": None,
                        "pnl": None,
                    }
                    local_trades.append(trade_record)
                    added += 1
                    logger.info(f"[Sync] Added missing order to trade log: {oid} ({order.get('ticker')})")

            # Flag local entries that have no matching Kalshi order (unconfirmed/ghost)
            cleaned_trades = []
            for trade in local_trades:
                tid = trade.get("id", "")
                # Short IDs (8 char hex) are dry-run or pre-fix entries — keep but mark
                if len(tid) <= 10 and trade.get("mode") == "live":
                    logger.warning(f"[Sync] Removing unconfirmed trade from log: {tid} ({trade.get('ticker')})")
                    removed += 1
                    continue  # Drop it
                cleaned_trades.append(trade)

            if added > 0 or removed > 0:
                TRADES_FILE.write_text(_json.dumps(cleaned_trades, indent=2))
                logger.info(f"[Sync] Trade log reconciled: +{added} added, -{removed} removed. Total: {len(cleaned_trades)}")
            else:
                logger.info(f"[Sync] Trade log in sync with Kalshi ({len(cleaned_trades)} trades)")

        except Exception as e:
            logger.error(f"[Executor] Startup sync failed: {e}", exc_info=True)

    async def get_balance(self) -> float:
        """Fetch available balance from Kalshi. Caches for 60 seconds."""
        now = time.time()
        if self._cached_balance is not None and (now - self._balance_fetched_at) < 60:
            return self._cached_balance

        if not self._http or not self._private_key:
            logger.warning("[Executor] Cannot fetch balance — not authenticated")
            return 0.0

        try:
            path = "/portfolio/balance"
            full_path = f"/trade-api/v2{path}"
            headers = self._auth_headers("GET", full_path)
            resp = await self._http.get(path, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Kalshi returns balance in cents
            balance_cents = data.get("balance", 0)
            balance_usd = balance_cents / 100.0
            self._cached_balance = balance_usd
            self._balance_fetched_at = now
            logger.info(f"[Executor] 💰 Account balance: ${balance_usd:.2f}")
            return balance_usd
        except Exception as e:
            logger.error(f"[Executor] Failed to fetch balance: {e}")
            return self._cached_balance or 0.0

    async def get_betting_budget(self) -> float:
        """
        Calculate how much we're allowed to bet this scan cycle.

        Rules:
        1. Hard floor: stop all betting if balance < DRAWDOWN_FLOOR_USD
        2. 50% rule: never put more than half total balance at risk across open bets
        3. Dynamic sizing: individual bet size = BET_SIZE_PCT of current balance
        """
        balance = await self.get_balance()

        # ── Rule 1: Hard floor ──
        if balance < DRAWDOWN_FLOOR_USD:
            logger.warning(
                f"[Executor] 🛑 DRAWDOWN FLOOR HIT — balance ${balance:.2f} < "
                f"floor ${DRAWDOWN_FLOOR_USD:.2f}. ALL BETTING SUSPENDED."
            )
            return -1.0  # Sentinel: caller should treat as hard stop

        max_at_risk = balance * MAX_BALANCE_USAGE_PCT

        # Subtract cost of existing open (unsettled) bets
        from weather_bets.trade_log import load_trades
        open_trades = [t for t in load_trades() if not t.get("settled")]
        open_cost = sum(t.get("cost", 0) for t in open_trades)

        remaining_budget = max(0, max_at_risk - open_cost)
        logger.info(
            f"[Executor] Budget: ${balance:.2f} balance × 50% = ${max_at_risk:.2f} max | "
            f"${open_cost:.2f} open | ${remaining_budget:.2f} available to bet"
        )
        return remaining_budget

    def dynamic_bet_size(self, balance: float) -> float:
        """
        Calculate bet size dynamically as % of current balance.
        Shrinks as balance drops, grows as balance grows.
        Capped at BET_SIZE_USD maximum.
        """
        size = balance * BET_SIZE_PCT
        size = min(size, BET_SIZE_USD)
        size = max(size, 0.50)  # Minimum $0.50 — don't place micro-bets
        return round(size, 2)

    async def place_bet(self, rec: BetRecommendation) -> PlacedBet:
        """
        Place a bet on Kalshi based on the LLM recommendation.
        
        In dry-run mode, logs the order but doesn't execute.
        In live mode, places the actual order.
        """
        opp = rec.opportunity
        ticker = rec.ticker or opp.bucket.ticker
        price_cents = int(opp.bucket.yes_price * 100)
        
        # Calculate contracts: each contract costs price_cents cents
        # and pays $1 (100 cents) on win
        contracts = max(1, int(rec.bet_size_usd / opp.bucket.yes_price))

        bet = PlacedBet(
            id=str(uuid.uuid4())[:8],
            ticker=ticker,
            city=opp.city,
            date=opp.date,
            bucket_label=opp.bucket.label,
            side=rec.side,
            price=opp.bucket.yes_price,
            quantity=contracts,
            cost_usd=opp.bucket.yes_price * contracts,
            our_probability=opp.our_probability,
            market_probability=opp.market_probability,
            edge_percent=opp.edge_percent,
            llm_reasoning=rec.reasoning,
        )

        if EXECUTION_MODE == "dry":
            logger.info(
                f"[Executor] 🔸 DRY RUN — Would place order:\n"
                f"  Ticker: {ticker}\n"
                f"  Side: {rec.side.upper()}\n"
                f"  Price: ${opp.bucket.yes_price:.2f} ({price_cents}¢)\n"
                f"  Contracts: {contracts}\n"
                f"  Total cost: ${bet.cost_usd:.2f}\n"
                f"  Edge: {opp.edge_percent:+.1f}%\n"
                f"  LLM confidence: {rec.confidence:.0%}\n"
                f"  Reasoning: {rec.reasoning}"
            )
        elif EXECUTION_MODE == "live":
            # ── DRAWDOWN + BUDGET CHECK ──
            remaining_budget = await self.get_betting_budget()

            if remaining_budget == -1.0:
                # Hard floor hit — full stop
                logger.warning(f"[Executor] 🛑 Bet blocked — drawdown floor active")
                return bet

            # ── DYNAMIC SIZING: use % of balance, not fixed amount ──
            balance = self._cached_balance or 0
            dynamic_size = self.dynamic_bet_size(balance)
            if bet.cost_usd > dynamic_size:
                old_contracts = contracts
                contracts = max(1, int(dynamic_size / opp.bucket.yes_price))
                bet.quantity = contracts
                bet.cost_usd = opp.bucket.yes_price * contracts
                logger.info(
                    f"[Executor] 📐 Dynamic sizing: ${dynamic_size:.2f} "
                    f"({balance:.2f} × {BET_SIZE_PCT:.0%}) → "
                    f"{old_contracts} → {contracts} contracts"
                )

            # ── 50% RULE: check remaining budget after sizing ──
            if bet.cost_usd > remaining_budget:
                if remaining_budget <= 0.50:
                    logger.warning(
                        f"[Executor] ⛔ SKIPPING — 50% budget exhausted "
                        f"(would cost ${bet.cost_usd:.2f}, ${remaining_budget:.2f} remaining)"
                    )
                    return bet
                # Scale down further to fit remaining budget
                old_contracts = contracts
                contracts = max(1, int(remaining_budget / opp.bucket.yes_price))
                bet.quantity = contracts
                bet.cost_usd = opp.bucket.yes_price * contracts
                logger.info(
                    f"[Executor] 📉 Scaled down: {old_contracts} → {contracts} contracts "
                    f"(${bet.cost_usd:.2f}) to fit 50% budget"
                )
            if not self._http or not self._private_key:
                logger.error("[Executor] Cannot place live order — not authenticated")
                return bet

            try:
                path = "/portfolio/orders"
                full_path = f"/trade-api/v2{path}"
                order_payload = {
                    "ticker": ticker,
                    "client_order_id": str(uuid.uuid4()),
                    "type": "limit",
                    "action": "buy",
                    "side": rec.side,
                    "count": contracts,
                    "yes_price": price_cents,
                }

                headers = self._auth_headers("POST", full_path)
                resp = await self._http.post(path, json=order_payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                order_id = data.get("order", {}).get("order_id", "unknown")
                logger.info(
                    f"[Executor] ✅ LIVE ORDER PLACED — order_id={order_id}\n"
                    f"  Ticker: {ticker}  Contracts: {contracts}  Cost: ${bet.cost_usd:.2f}"
                )
                bet.id = order_id
                # ✅ Only append to placed_bets after confirmed API success
                self.placed_bets.append(bet)
                return bet

            except Exception as e:
                logger.error(f"[Executor] ❌ Order failed — NOT logging trade: {e}", exc_info=True)
                return bet  # Return without logging

        # Dry run — always log
        self.placed_bets.append(bet)
        return bet

    async def check_settlements(self) -> list[dict]:
        """
        Check Kalshi for settled positions and auto-mark them in the trade log.

        For each open (unsettled) trade, fetches the order status from Kalshi.
        If the market has settled, determines win/loss and calls mark_trade_settled().

        Returns a list of newly settled trades with their outcome.
        """
        from weather_bets.trade_log import load_trades, mark_trade_settled

        trades = load_trades()
        open_trades = [t for t in trades if not t.get("settled") and t.get("id")]
        if not open_trades:
            logger.debug("[Settlements] No open trades to check")
            return []

        if not self._http or not self._private_key:
            logger.warning("[Settlements] Cannot check settlements — not authenticated")
            return []

        newly_settled = []

        for trade in open_trades:
            order_id = trade.get("id")
            ticker = trade.get("ticker", "?")
            try:
                path = f"/portfolio/orders/{order_id}"
                full_path = f"/trade-api/v2{path}"
                headers = self._auth_headers("GET", full_path)
                resp = await self._http.get(path, headers=headers)

                if resp.status_code == 404:
                    logger.debug(f"[Settlements] Order {order_id} not found on Kalshi")
                    continue

                resp.raise_for_status()
                data = resp.json()
                order = data.get("order", {})

                status = order.get("status", "")
                # Kalshi order statuses: "resting", "filled", "canceled", "pending_settlement", "settled"
                if status not in ("settled", "canceled"):
                    logger.debug(f"[Settlements] {ticker} order={order_id} status={status} — not settled yet")
                    continue

                if status == "canceled":
                    # Cancelled orders didn't execute — mark as settled/loss (cost already 0 in practice)
                    logger.info(f"[Settlements] {ticker} was CANCELED — marking as loss")
                    mark_trade_settled(ticker, won=False)
                    newly_settled.append({"ticker": ticker, "outcome": "canceled"})
                    continue

                # For settled orders, check if we won by looking at fill data
                # If our side (yes/no) won, payout_amount > 0
                payout = order.get("payout", 0)
                side = trade.get("side", "yes")
                qty = trade.get("qty", 0)
                cost = trade.get("cost", 0)

                # Kalshi returns payout in cents
                payout_usd = payout / 100.0 if payout else 0
                won = payout_usd > 0

                logger.info(
                    f"[Settlements] ✅ Settled {ticker}: {'WIN' if won else 'LOSS'} "
                    f"(payout=${payout_usd:.2f}, cost=${cost:.2f})"
                )
                mark_trade_settled(ticker, won=won)
                newly_settled.append({
                    "ticker": ticker,
                    "outcome": "win" if won else "loss",
                    "payout_usd": payout_usd,
                    "cost_usd": cost,
                    "pnl": payout_usd - cost,
                })

            except Exception as e:
                logger.warning(f"[Settlements] Failed to check {ticker} ({order_id}): {e}")

        if newly_settled:
            logger.info(f"[Settlements] Settled {len(newly_settled)} trades this cycle")

        return newly_settled

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
