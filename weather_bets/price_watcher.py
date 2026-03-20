"""Real-time price watcher for Kalshi weather markets.

Polls every PRICE_WATCH_INTERVAL seconds, tracks rolling price history,
and emits momentum signals for buy/sell decisions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from weather_bets.config import (
    ACTIVE_CITIES,
    CITIES,
    KALSHI_BASE_URL,
    PRICE_WATCH_INTERVAL,
)
from weather_bets.kalshi_weather import fetch_weather_markets
from weather_bets.trade_log import get_open_tickers

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Thresholds
BUY_THRESHOLD = 0.15   # price dropped >15% over 3 consecutive readings
SELL_THRESHOLD = 0.20  # price rose >20% over 3 consecutive readings
MOMENTUM_WINDOW = 3    # consecutive readings required
HISTORY_MAXLEN = 10    # rolling window per ticker


class PriceWatcher:
    """Continuously polls Kalshi market prices and detects momentum signals."""

    def __init__(self, scan_state: dict):
        """
        Args:
            scan_state: Reference to main.py's scan_state dict.
                        Used to check consensus forecasts for buy signals.
        """
        self.scan_state = scan_state
        # ticker -> deque of {"ts": float, "price": float}
        self._history: dict[str, deque] = {}
        # ticker -> "buy" | "sell" | None
        self._signals: dict[str, str | None] = {}
        # Full state exposed to /api/prices
        self.state: dict = {
            "tickers": {},      # ticker -> {price, history, signal, direction}
            "signals": {},      # active signals only
            "last_update": None,
            "poll_count": 0,
        }
        self._running = False

    # ── Public API ──────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a snapshot of current watcher state (safe to read from main.py)."""
        return dict(self.state)

    def get_signal(self, ticker: str) -> str | None:
        """Return current signal for a ticker: 'buy', 'sell', or None."""
        return self._signals.get(ticker)

    def clear_signal(self, ticker: str) -> None:
        """Clear a signal after it's been acted on."""
        self._signals.pop(ticker, None)
        self.state["signals"].pop(ticker, None)
        if ticker in self.state["tickers"]:
            self.state["tickers"][ticker]["signal"] = None

    # ── Background loop ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Run the price-watch loop forever (call as asyncio task)."""
        self._running = True
        logger.info(f"[PriceWatcher] Started — polling every {PRICE_WATCH_INTERVAL}s")
        # Small initial delay so the scan loop can warm up first
        await asyncio.sleep(30)
        while self._running:
            try:
                await self._poll_prices()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[PriceWatcher] Poll error: {e}", exc_info=True)
            await asyncio.sleep(PRICE_WATCH_INTERVAL)

    async def stop(self) -> None:
        self._running = False

    # ── Internal ────────────────────────────────────────────────────────

    async def _poll_prices(self) -> None:
        """Fetch prices for all watched tickers and update state."""
        now = time.time()
        tickers_fetched: set[str] = set()

        # 1. Gather tickers to watch: open positions + next 2 days markets
        open_tickers = get_open_tickers()
        target_dates = _next_n_dates(2)

        for city_code in ACTIVE_CITIES:
            city = CITIES.get(city_code)
            if not city:
                continue
            for target_date in target_dates:
                try:
                    buckets = await fetch_weather_markets(city, target_date)
                    for b in buckets:
                        if b.yes_price <= 0:
                            continue
                        ticker = b.ticker
                        tickers_fetched.add(ticker)
                        self._record_price(ticker, now, b.yes_price)
                except Exception as e:
                    logger.warning(f"[PriceWatcher] Failed fetching {city_code}/{target_date}: {e}")

        # 2. Detect signals for all tickers with enough history
        new_signals: dict[str, str] = {}
        for ticker in list(self._history.keys()):
            signal = self._detect_signal(ticker, open_tickers)
            self._signals[ticker] = signal
            if signal:
                new_signals[ticker] = signal
                logger.info(f"[PriceWatcher] 🚨 Signal: {signal.upper()} on {ticker}")

        # 3. Update exposed state
        tickers_state: dict = {}
        for ticker, hist in self._history.items():
            hist_list = list(hist)
            prices = [h["price"] for h in hist_list]
            direction = _price_direction(prices)
            tickers_state[ticker] = {
                "ticker": ticker,
                "current_price": prices[-1] if prices else None,
                "history": [
                    {"ts": h["ts"], "price": h["price"], "time": _fmt_ts(h["ts"])}
                    for h in hist_list
                ],
                "signal": self._signals.get(ticker),
                "direction": direction,
                "in_open_positions": ticker in open_tickers,
            }

        self.state["tickers"] = tickers_state
        self.state["signals"] = new_signals
        self.state["last_update"] = datetime.now(timezone.utc).isoformat()
        self.state["poll_count"] = self.state.get("poll_count", 0) + 1

        if new_signals:
            logger.info(f"[PriceWatcher] Active signals: {new_signals}")
        else:
            logger.info(
                f"[PriceWatcher] Poll #{self.state['poll_count']} — "
                f"{len(tickers_state)} tickers, no signals"
            )

    def _record_price(self, ticker: str, ts: float, price: float) -> None:
        """Append a price reading to ticker's rolling history."""
        if ticker not in self._history:
            self._history[ticker] = deque(maxlen=HISTORY_MAXLEN)
        self._history[ticker].append({"ts": ts, "price": price})

    def _detect_signal(self, ticker: str, open_tickers: set[str]) -> str | None:
        """
        Detect momentum signal for a ticker.

        Buy:  price dropped >15% over last 3 readings AND consensus says YES likely.
        Sell: price rose >20% over last 3 readings AND we hold this position.
        """
        hist = self._history.get(ticker)
        if not hist or len(hist) < MOMENTUM_WINDOW:
            return None

        readings = [h["price"] for h in hist]
        # Last MOMENTUM_WINDOW readings
        window = readings[-MOMENTUM_WINDOW:]
        baseline = window[0]
        latest = window[-1]

        if baseline <= 0:
            return None

        # Check all consecutive moves are in same direction
        is_monotone_drop = all(window[i] <= window[i - 1] for i in range(1, len(window)))
        is_monotone_rise = all(window[i] >= window[i - 1] for i in range(1, len(window)))

        pct_change = (latest - baseline) / baseline

        # SELL signal: price rose >20% consecutively AND we hold it
        if (
            is_monotone_rise
            and pct_change >= SELL_THRESHOLD
            and ticker in open_tickers
        ):
            logger.info(
                f"[PriceWatcher] SELL signal {ticker}: "
                f"price rose {pct_change:+.1%} over {MOMENTUM_WINDOW} readings "
                f"({baseline:.2f} → {latest:.2f})"
            )
            return "sell"

        # BUY signal: price dropped >15% consecutively AND consensus says YES likely
        if (
            is_monotone_drop
            and abs(pct_change) >= BUY_THRESHOLD
            and ticker not in open_tickers
            and self._consensus_says_yes(ticker)
        ):
            logger.info(
                f"[PriceWatcher] BUY signal {ticker}: "
                f"price dropped {pct_change:+.1%} over {MOMENTUM_WINDOW} readings "
                f"({baseline:.2f} → {latest:.2f})"
            )
            return "buy"

        return None

    def _consensus_says_yes(self, ticker: str) -> bool:
        """
        Check if our model's consensus forecast still says YES is likely
        (our_probability > 50%) for this ticker.
        """
        opportunities = self.scan_state.get("opportunities", [])
        for opp in opportunities:
            if opp.get("ticker") == ticker:
                return opp.get("our_prob", 0) > 50.0
        # No data — be conservative and don't buy
        return False


# ── Helpers ─────────────────────────────────────────────────────────────

def _next_n_dates(n: int) -> list[str]:
    """Return the next n calendar dates (today + tomorrow, etc.) as YYYY-MM-DD."""
    today = datetime.now(timezone.utc).date()
    return [(today + timedelta(days=i)).isoformat() for i in range(n)]


def _price_direction(prices: list[float]) -> str:
    """Summarize recent price direction as 'up', 'down', or 'flat'."""
    if len(prices) < 2:
        return "flat"
    recent = prices[-3:] if len(prices) >= 3 else prices
    if recent[-1] > recent[0] * 1.02:
        return "up"
    if recent[-1] < recent[0] * 0.98:
        return "down"
    return "flat"


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")
