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
SCAN_INTERVAL: int = int(os.getenv("WEATHER_SCAN_INTERVAL", "1800"))  # 30 min
PRICE_WATCH_INTERVAL: int = int(os.getenv("WEATHER_PRICE_WATCH_INTERVAL", "120"))  # 2 min
EXECUTION_MODE: str = os.getenv("WEATHER_EXECUTION_MODE", "dry")  # "dry" or "live"

# ── Risk Management ─────────────────────────────────────────────────────
# Bet size as % of current balance (dynamic sizing)
BET_SIZE_PCT: float = float(os.getenv("WEATHER_BET_SIZE_PCT", "0.06"))  # 6% of balance per bet
# Hard floor — stop all betting if balance drops below this
DRAWDOWN_FLOOR_USD: float = float(os.getenv("WEATHER_DRAWDOWN_FLOOR", "15.0"))
# Market sanity check — if any adjacent bucket is priced above this,
# we need strong conviction (our prob > bucket threshold + margin) to bet against it
MARKET_CONVICTION_THRESHOLD: float = float(os.getenv("WEATHER_CONVICTION_THRESHOLD", "0.65"))


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
    "LAX": CityConfig(
        code="LAX",
        name="Los Angeles",
        kalshi_series="KXHIGHLAX",
        nws_office="LOX",
        nws_grid_x=148,
        nws_grid_y=41,
        station_id="KLAX",
        lat=33.9425,    # Los Angeles International Airport
        lon=-118.4081,
    ),
    "CHI": CityConfig(
        code="CHI",
        name="Chicago",
        kalshi_series="KXHIGHCHI",
        nws_office="LOT",
        nws_grid_x=66,
        nws_grid_y=77,
        station_id="KORD",
        lat=41.9742,    # O'Hare International Airport
        lon=-87.9073,
    ),
    "MIA": CityConfig(
        code="MIA",
        name="Miami",
        kalshi_series="KXHIGHMIA",
        nws_office="MFL",
        nws_grid_x=106,
        nws_grid_y=51,
        station_id="KMIA",
        lat=25.7959,    # Miami International Airport
        lon=-80.2870,
    ),
    "DEN": CityConfig(
        code="DEN",
        name="Denver",
        kalshi_series="KXHIGHDEN",
        nws_office="BOU",
        nws_grid_x=74,
        nws_grid_y=66,
        station_id="KDEN",
        lat=39.8561,    # Denver International Airport
        lon=-104.6737,
    ),
    "HOU": CityConfig(
        code="HOU",
        name="Houston",
        kalshi_series="KXHIGHTHOU",
        nws_office="HGX",
        nws_grid_x=65,
        nws_grid_y=97,
        station_id="KIAH",
        lat=29.9844,    # George Bush Intercontinental
        lon=-95.3414,
    ),
    "DAL": CityConfig(
        code="DAL",
        name="Dallas",
        kalshi_series="KXHIGHTDAL",
        nws_office="FWD",
        nws_grid_x=80,
        nws_grid_y=108,
        station_id="KDFW",
        lat=32.8998,    # DFW International Airport
        lon=-97.0403,
    ),
    "LAS": CityConfig(
        code="LAS",
        name="Las Vegas",
        kalshi_series="KXHIGHTLV",
        nws_office="VEF",
        nws_grid_x=126,
        nws_grid_y=59,
        station_id="KLAS",
        lat=36.0840,    # Harry Reid International Airport
        lon=-115.1537,
    ),
    "SAT": CityConfig(
        code="SAT",
        name="San Antonio",
        kalshi_series="KXHIGHTSATX",
        nws_office="EWX",
        nws_grid_x=140,
        nws_grid_y=78,
        station_id="KSAT",
        lat=29.5337,    # San Antonio International Airport
        lon=-98.4698,
    ),
    "PHX": CityConfig(
        code="PHX",
        name="Phoenix",
        kalshi_series="KXHIGHTPHX",
        nws_office="PSR",
        nws_grid_x=159,
        nws_grid_y=56,
        station_id="KPHX",
        lat=33.4373,    # Phoenix Sky Harbor International Airport
        lon=-112.0078,
    ),
}

# Active cities for scanning
# Only include cities with active Kalshi temperature high markets
# SFO/NYC/DAL/PHX/BOS/SEA currently have no active markets
ACTIVE_CITIES: list[str] = ["AUS"]  # Start with Austin only — expand after validation

# ── Server ───────────────────────────────────────────────────────────────
SERVER_HOST: str = "127.0.0.1"
SERVER_PORT: int = 8001
