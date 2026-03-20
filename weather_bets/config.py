"""Configuration for the weather betting system."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from weather_bets.models import CityConfig

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── API Keys ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
KALSHI_API_KEY_ID: str = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")
KALSHI_BASE_URL: str = "https://api.elections.kalshi.com/trade-api/v2"

# ── Betting Parameters ──────────────────────────────────────────────────
BET_SIZE_USD: float = float(os.getenv("WEATHER_BET_SIZE", "10"))
MIN_EDGE: float = float(os.getenv("WEATHER_MIN_EDGE", "0.10"))  # 10%
SCAN_INTERVAL: int = int(os.getenv("WEATHER_SCAN_INTERVAL", "1800"))  # 30 min
EXECUTION_MODE: str = os.getenv("WEATHER_EXECUTION_MODE", "dry")  # "dry" or "live"

# ── NWS Forecast Settings ───────────────────────────────────────────────
# Standard deviation for temperature forecast uncertainty (°F)
# NWS 24h forecasts are typically within ±2-3°F
FORECAST_STDEV_24H: float = 3.0   # Next day (NWS can disagree with other sources by 2-3°F)
FORECAST_STDEV_48H: float = 4.0   # Day after

# ── City Definitions ────────────────────────────────────────────────────
# Each city maps to its Kalshi series ticker, NWS grid, and airport station.
# Kalshi settles on NWS Climatological Report (Daily) for the airport station.

CITIES: dict[str, CityConfig] = {
    "AUS": CityConfig(
        code="AUS",
        name="Austin",
        kalshi_series="KXHIGHAUS",
        nws_office="EWX",
        nws_grid_x=159,       # Grid for Bergstrom airport, NOT downtown
        nws_grid_y=88,
        station_id="KAUS",
        lat=30.1975,    # Austin-Bergstrom International Airport
        lon=-97.6664,
    ),
    # Future: add more cities
    # "SFO": CityConfig(
    #     code="SFO",
    #     name="San Francisco",
    #     kalshi_series="KXHIGHSFO",
    #     nws_office="MTR",
    #     nws_grid_x=85,
    #     nws_grid_y=105,
    #     station_id="KSFO",
    #     lat=37.6213,
    #     lon=-122.3790,
    # ),
}

# Active cities for scanning (start with Austin)
ACTIVE_CITIES: list[str] = ["AUS"]

# ── Server ───────────────────────────────────────────────────────────────
SERVER_HOST: str = "127.0.0.1"
SERVER_PORT: int = 8001
