"""Persistent trade log — saves all bets and scan results to disk."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Store in the weather_bets directory
LOG_DIR = Path(__file__).parent / "data"
TRADES_FILE = LOG_DIR / "trades.json"
DRY_TRADES_FILE = LOG_DIR / "dry_trades.json"
SCANS_FILE = LOG_DIR / "scans.json"


def _ensure_dir():
    LOG_DIR.mkdir(exist_ok=True)


def load_trades() -> list[dict]:
    """Load all historical trades from disk."""
    _ensure_dir()
    if TRADES_FILE.exists():
        try:
            return json.loads(TRADES_FILE.read_text())
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[TradeLog] Could not load trades: {e}")
    return []


def save_trade(trade: dict) -> None:
    """Append a trade to the persistent log.
    
    Dry-run trades go to dry_trades.json; real trades go to trades.json.
    """
    _ensure_dir()
    is_dry = trade.get("mode") == "dry"
    target_file = DRY_TRADES_FILE if is_dry else TRADES_FILE
    
    existing = []
    if target_file.exists():
        try:
            existing = json.loads(target_file.read_text())
        except (json.JSONDecodeError, Exception):
            existing = []
    
    trade["logged_at"] = datetime.now(timezone.utc).isoformat()
    existing.append(trade)
    target_file.write_text(json.dumps(existing, indent=2))
    
    label = "DRY" if is_dry else "LIVE"
    logger.info(f"[TradeLog] Saved {label} trade #{len(existing)}: {trade.get('ticker', '?')} "
                f"${trade.get('cost', 0):.2f}")


def already_bet_on(ticker: str) -> bool:
    """Check if we already have an open (unsettled) bet on this ticker."""
    trades = load_trades()
    for t in trades:
        if t.get("ticker") == ticker and not t.get("settled"):
            logger.info(f"[TradeLog] Already have open bet on {ticker} — skipping")
            return True
    return False


def get_open_tickers() -> set[str]:
    """Get all tickers with open (unsettled) bets."""
    trades = load_trades()
    return {t["ticker"] for t in trades if not t.get("settled") and "ticker" in t}


def save_scan_result(scan: dict) -> None:
    """Append a scan summary to the persistent log."""
    _ensure_dir()
    scans = []
    if SCANS_FILE.exists():
        try:
            scans = json.loads(SCANS_FILE.read_text())
        except (json.JSONDecodeError, Exception):
            pass

    scan["scanned_at"] = datetime.now(timezone.utc).isoformat()
    scans.append(scan)

    # Keep last 500 scans only
    if len(scans) > 500:
        scans = scans[-500:]

    SCANS_FILE.write_text(json.dumps(scans, indent=2))


def get_trade_summary() -> dict:
    """Get P&L summary from trade history."""
    trades = load_trades()
    if not trades:
        return {
            "total_trades": 0,
            "total_cost": 0,
            "total_pnl": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "win_rate": 0,
        }

    total_cost = sum(t.get("cost", 0) for t in trades)
    settled = [t for t in trades if t.get("settled")]
    wins = [t for t in settled if t.get("won")]
    losses = [t for t in settled if not t.get("won")]
    pending = [t for t in trades if not t.get("settled")]

    # P&L: for wins, payout is $1 per contract minus cost
    # for losses, loss is the cost
    total_pnl = 0
    for t in wins:
        total_pnl += t.get("qty", 0) * 1.00 - t.get("cost", 0)
    for t in losses:
        total_pnl -= t.get("cost", 0)

    win_rate = len(wins) / len(settled) * 100 if settled else 0

    return {
        "total_trades": len(trades),
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_pnl, 2),
        "wins": len(wins),
        "losses": len(losses),
        "pending": len(pending),
        "win_rate": round(win_rate, 1),
    }


def mark_trade_settled(ticker: str, won: bool) -> None:
    """Mark a trade as settled with win/loss result."""
    trades = load_trades()
    for t in trades:
        if t.get("ticker") == ticker and not t.get("settled"):
            t["settled"] = True
            t["won"] = won
            t["settled_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"[TradeLog] Settled {ticker}: {'WIN' if won else 'LOSS'}")
            break
    TRADES_FILE.write_text(json.dumps(trades, indent=2))
