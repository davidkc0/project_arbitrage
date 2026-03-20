"""Pydantic data models for the arbitrage tool."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Platform(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class OutcomeSide(str, Enum):
    YES = "YES"
    NO = "NO"


class TradeStatus(str, Enum):
    PENDING = "pending"          # Queued, awaiting user confirmation
    CONFIRMED = "confirmed"      # User confirmed, executing
    PARTIALLY_FILLED = "partial" # One leg filled
    FILLED = "filled"            # Both legs filled
    CANCELLED = "cancelled"      # User cancelled or system killed
    FAILED = "failed"            # Execution error
    RESOLVED = "resolved"        # Market resolved, profit/loss realized


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------

class OutcomePrice(BaseModel):
    """A single outcome's pricing data."""
    side: OutcomeSide
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_price: float = 0.0
    volume: float = 0.0
    # Order book depth at best bid/ask (in USD)
    bid_depth: float = 0.0
    ask_depth: float = 0.0


class MarketEvent(BaseModel):
    """Normalized representation of a prediction market event from any platform."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    platform: Platform
    platform_id: str               # Original ID on the source platform
    question: str                   # Human-readable event question
    category: str = ""              # e.g., "Politics", "Crypto", "Sports"
    outcomes: list[OutcomePrice] = Field(default_factory=list)
    end_date: datetime | None = None
    is_active: bool = True
    url: str = ""                   # Link to the event on the platform
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Polymarket-specific
    condition_id: str = ""          # Polymarket condition ID
    token_ids: list[str] = Field(default_factory=list)  # YES/NO token IDs

    # Kalshi-specific
    ticker: str = ""                # Kalshi event ticker


# ---------------------------------------------------------------------------
# Arbitrage
# ---------------------------------------------------------------------------

class ArbitrageOpportunity(BaseModel):
    """A matched pair of events across two platforms with a calculated edge."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_a: MarketEvent            # The event on platform A
    event_b: MarketEvent            # The same event on platform B
    match_confidence: float = 0.0   # Fuzzy match score (0–100)

    # Side-by-side prices for display
    yes_a: float = 0.0              # YES price on platform A
    yes_b: float = 0.0              # YES price on platform B
    no_a: float = 0.0               # NO price on platform A
    no_b: float = 0.0               # NO price on platform B
    yes_spread: float = 0.0         # |yes_a - yes_b| (price gap for YES)
    no_spread: float = 0.0          # |no_a - no_b| (price gap for NO)

    # Best strategy found
    buy_yes_platform: Platform = Platform.POLYMARKET
    buy_yes_price: float = 0.0      # Price to buy YES on one platform
    buy_no_platform: Platform = Platform.KALSHI
    buy_no_price: float = 0.0       # Price to buy NO on the other platform
    total_cost: float = 0.0         # buy_yes + buy_no (should be < 1.0)
    gross_edge: float = 0.0         # 1.0 - total_cost
    fee_estimate: float = 0.0       # Estimated fees on both platforms
    net_edge: float = 0.0           # gross_edge - fee_estimate
    net_edge_percent: float = 0.0   # net_edge / total_cost * 100

    # Liquidity constraints
    max_bet_size: float = 0.0       # Limited by order book depth
    available_depth_a: float = 0.0
    available_depth_b: float = 0.0

    # Time-value ranking
    days_to_expiry: float = 365.0   # Days until market resolves
    annualized_edge: float = 0.0    # net_edge_percent * (365 / days_to_expiry)

    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    is_stale: bool = False


# ---------------------------------------------------------------------------
# Execution & Positions
# ---------------------------------------------------------------------------

class TradeLeg(BaseModel):
    """One leg of an arbitrage trade (one side on one platform)."""
    platform: Platform
    platform_order_id: str = ""
    market_id: str
    side: OutcomeSide
    price: float
    quantity: float               # Number of contracts / shares
    cost: float = 0.0            # price * quantity
    filled_quantity: float = 0.0
    filled_avg_price: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    placed_at: datetime | None = None
    filled_at: datetime | None = None


class Trade(BaseModel):
    """A complete arbitrage trade (two legs)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    opportunity_id: str
    leg_a: TradeLeg
    leg_b: TradeLeg
    total_cost: float = 0.0
    expected_profit: float = 0.0
    actual_profit: float | None = None  # Set after resolution
    status: TradeStatus = TradeStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None


class Position(BaseModel):
    """An open position from an executed arbitrage trade."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trade_id: str
    platform_a: Platform
    platform_a_market_id: str
    platform_a_side: OutcomeSide
    platform_a_quantity: float
    platform_b: Platform
    platform_b_market_id: str
    platform_b_side: OutcomeSide
    platform_b_quantity: float
    total_cost: float
    expected_payout: float = 1.0  # Always $1.00 per share for binary
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: datetime | None = None
    realized_pnl: float | None = None


# ---------------------------------------------------------------------------
# Volume / Airdrop Tracking
# ---------------------------------------------------------------------------

class VolumeStat(BaseModel):
    """Cumulative volume stats for airdrop farming."""
    platform: Platform
    total_volume_usd: float = 0.0
    total_trades: int = 0
    total_markets_traded: int = 0
    first_trade_date: datetime | None = None
    last_trade_date: datetime | None = None


# ---------------------------------------------------------------------------
# Matched Pair Cache
# ---------------------------------------------------------------------------

class MatchedPair(BaseModel):
    """Cached confirmed match between events on different platforms."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    polymarket_id: str
    kalshi_ticker: str
    polymarket_question: str = ""
    kalshi_question: str = ""
    match_confidence: float = 0.0
    confirmed_by_user: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
