"""Data models for the weather betting system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class BetDecision(Enum):
    BET = "bet"
    SKIP = "skip"


@dataclass
class CityConfig:
    """Configuration for a city's weather betting."""
    code: str              # Kalshi city code (e.g., "AUS")
    name: str              # Human name (e.g., "Austin")
    kalshi_series: str     # Kalshi series ticker (e.g., "KXHIGHAUS")
    nws_office: str        # NWS forecast office (e.g., "EWX")
    nws_grid_x: int        # NWS grid X coordinate
    nws_grid_y: int        # NWS grid Y coordinate
    station_id: str        # ICAO station ID (e.g., "KAUS")
    lat: float
    lon: float


@dataclass
class ForecastData:
    """NWS forecast for a specific city and date."""
    city: str
    date: str                          # "2026-03-20"
    high_temp_f: int                   # Point forecast high (e.g., 91)
    low_temp_f: int | None = None
    short_forecast: str = ""           # "Sunny", "Partly Cloudy"
    detailed_forecast: str = ""
    wind_speed: str = ""
    precip_probability: int = 0
    fetched_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TemperatureBucket:
    """A single temperature range bucket from Kalshi."""
    ticker: str            # e.g., "KXHIGHAUS-26MAR20-B90.5"
    label: str             # e.g., "90-91°"
    low_bound: float | None    # 90 (None for "X or below")
    high_bound: float | None   # 91 (None for "X or above")
    yes_price: float       # Market price for YES (0.00 - 1.00)
    no_price: float        # Market price for NO
    volume: float = 0
    close_time: str = ""


@dataclass
class EdgeOpportunity:
    """A betting opportunity where forecast disagrees with market."""
    city: str
    date: str
    bucket: TemperatureBucket
    forecast_high: int
    our_probability: float          # Our estimated probability (0-1)
    market_probability: float       # Implied from YES price (0-1)
    edge: float                     # our_prob - market_prob
    edge_percent: float             # edge * 100
    expected_value: float           # edge * payout - (1 - our_prob) * cost
    forecast_detail: str = ""


@dataclass
class BetRecommendation:
    """LLM's recommendation on whether/how to bet."""
    decision: BetDecision
    opportunity: EdgeOpportunity
    confidence: float               # 0-1, LLM's confidence in the bet
    bet_size_usd: float             # Recommended bet size
    reasoning: str                  # LLM's explanation
    side: str = "yes"               # "yes" or "no"
    ticker: str = ""


@dataclass
class PlacedBet:
    """Record of a bet that was placed."""
    id: str = ""
    ticker: str = ""
    city: str = ""
    date: str = ""
    bucket_label: str = ""
    side: str = "yes"
    price: float = 0.0
    quantity: int = 0
    cost_usd: float = 0.0
    our_probability: float = 0.0
    market_probability: float = 0.0
    edge_percent: float = 0.0
    llm_reasoning: str = ""
    placed_at: datetime = field(default_factory=datetime.utcnow)
    settled: bool = False
    won: bool = False
    pnl: float = 0.0


@dataclass
class SpreadBet:
    """A spread bet across multiple adjacent temperature buckets."""
    city: str
    date: str
    forecast_high: int
    buckets: list[TemperatureBucket]       # The buckets in the spread
    bucket_probabilities: list[float]       # Our probability for each bucket
    total_probability: float                # Combined probability of spread hitting
    total_cost: float                       # Total cost to buy YES on all buckets
    profit_if_hit: float                    # $1.00 - total_cost
    expected_profit: float                  # total_probability * profit_if_hit - (1 - total_prob) * total_cost
    roi_percent: float                      # expected_profit / total_cost * 100
    forecast_detail: str = ""


@dataclass
class SpreadRecommendation:
    """LLM's recommendation on a spread bet."""
    decision: BetDecision
    spread: SpreadBet
    confidence: float
    bet_size_usd: float                     # Total USD across all legs
    allocations: list[float]                # USD allocation per bucket
    reasoning: str

