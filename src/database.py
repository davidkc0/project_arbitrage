"""SQLite persistence layer for the arbitrage tool."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import aiosqlite

from src.config import settings
from src.models import MatchedPair, Platform, VolumeStat

logger = logging.getLogger(__name__)

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS matched_pairs (
    id TEXT PRIMARY KEY,
    polymarket_id TEXT NOT NULL,
    kalshi_ticker TEXT NOT NULL,
    polymarket_question TEXT DEFAULT '',
    kalshi_question TEXT DEFAULT '',
    match_confidence REAL DEFAULT 0.0,
    confirmed_by_user INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    opportunity_id TEXT NOT NULL,
    leg_a_json TEXT NOT NULL,
    leg_b_json TEXT NOT NULL,
    total_cost REAL DEFAULT 0.0,
    expected_profit REAL DEFAULT 0.0,
    actual_profit REAL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL,
    platform_a TEXT NOT NULL,
    platform_a_market_id TEXT NOT NULL,
    platform_a_side TEXT NOT NULL,
    platform_a_quantity REAL NOT NULL,
    platform_b TEXT NOT NULL,
    platform_b_market_id TEXT NOT NULL,
    platform_b_side TEXT NOT NULL,
    platform_b_quantity REAL NOT NULL,
    total_cost REAL NOT NULL,
    expected_payout REAL DEFAULT 1.0,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    realized_pnl REAL
);

CREATE TABLE IF NOT EXISTS volume_stats (
    platform TEXT PRIMARY KEY,
    total_volume_usd REAL DEFAULT 0.0,
    total_trades INTEGER DEFAULT 0,
    total_markets_traded INTEGER DEFAULT 0,
    first_trade_date TEXT,
    last_trade_date TEXT
);

CREATE TABLE IF NOT EXISTS opportunities_log (
    id TEXT PRIMARY KEY,
    event_a_question TEXT,
    event_b_question TEXT,
    buy_yes_platform TEXT,
    buy_yes_price REAL,
    buy_no_platform TEXT,
    buy_no_price REAL,
    total_cost REAL,
    net_edge REAL,
    net_edge_percent REAL,
    match_confidence REAL,
    discovered_at TEXT NOT NULL
);
"""


class Database:
    """Async SQLite database for persisting arbitrage data."""

    def __init__(self, db_path: str | None = None):
        self._path = db_path or settings.db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._path)
        await self._db.executescript(DB_SCHEMA)
        await self._db.commit()
        logger.info(f"[DB] Initialized at {self._path}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ── Matched Pairs ───────────────────────────────────────────────────

    async def save_matched_pair(self, pair: MatchedPair) -> None:
        if not self._db:
            return
        await self._db.execute(
            """INSERT OR REPLACE INTO matched_pairs
               (id, polymarket_id, kalshi_ticker, polymarket_question,
                kalshi_question, match_confidence, confirmed_by_user, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pair.id,
                pair.polymarket_id,
                pair.kalshi_ticker,
                pair.polymarket_question,
                pair.kalshi_question,
                pair.match_confidence,
                1 if pair.confirmed_by_user else 0,
                pair.created_at.isoformat(),
            ),
        )
        await self._db.commit()

    async def load_matched_pairs(self) -> list[MatchedPair]:
        if not self._db:
            return []
        cursor = await self._db.execute("SELECT * FROM matched_pairs")
        rows = await cursor.fetchall()
        pairs = []
        for row in rows:
            pairs.append(
                MatchedPair(
                    id=row[0],
                    polymarket_id=row[1],
                    kalshi_ticker=row[2],
                    polymarket_question=row[3],
                    kalshi_question=row[4],
                    match_confidence=row[5],
                    confirmed_by_user=bool(row[6]),
                    created_at=datetime.fromisoformat(row[7]),
                )
            )
        return pairs

    # ── Volume Stats ────────────────────────────────────────────────────

    async def update_volume(
        self, platform: Platform, volume_usd: float, market_id: str
    ) -> None:
        if not self._db:
            return
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """INSERT INTO volume_stats (platform, total_volume_usd, total_trades,
                total_markets_traded, first_trade_date, last_trade_date)
               VALUES (?, ?, 1, 1, ?, ?)
               ON CONFLICT(platform) DO UPDATE SET
                total_volume_usd = total_volume_usd + ?,
                total_trades = total_trades + 1,
                last_trade_date = ?""",
            (platform.value, volume_usd, now, now, volume_usd, now),
        )
        await self._db.commit()

    async def get_volume_stats(self) -> dict[str, VolumeStat]:
        if not self._db:
            return {}
        cursor = await self._db.execute("SELECT * FROM volume_stats")
        rows = await cursor.fetchall()
        stats = {}
        for row in rows:
            platform_name = row[0]
            stats[platform_name] = VolumeStat(
                platform=Platform(platform_name),
                total_volume_usd=row[1],
                total_trades=row[2],
                total_markets_traded=row[3],
                first_trade_date=(
                    datetime.fromisoformat(row[4]) if row[4] else None
                ),
                last_trade_date=(
                    datetime.fromisoformat(row[5]) if row[5] else None
                ),
            )
        return stats

    # ── Opportunities Log ───────────────────────────────────────────────

    async def log_opportunity(self, opp_data: dict) -> None:
        if not self._db:
            return
        await self._db.execute(
            """INSERT OR REPLACE INTO opportunities_log
               (id, event_a_question, event_b_question, buy_yes_platform,
                buy_yes_price, buy_no_platform, buy_no_price, total_cost,
                net_edge, net_edge_percent, match_confidence, discovered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                opp_data.get("id", ""),
                opp_data.get("event_a_question", ""),
                opp_data.get("event_b_question", ""),
                opp_data.get("buy_yes_platform", ""),
                opp_data.get("buy_yes_price", 0),
                opp_data.get("buy_no_platform", ""),
                opp_data.get("buy_no_price", 0),
                opp_data.get("total_cost", 0),
                opp_data.get("net_edge", 0),
                opp_data.get("net_edge_percent", 0),
                opp_data.get("match_confidence", 0),
                opp_data.get("discovered_at", datetime.utcnow().isoformat()),
            ),
        )
        await self._db.commit()
