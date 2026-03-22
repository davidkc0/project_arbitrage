"""NOAA historical data fetcher — pulls past temperature records for bias analysis."""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta

import httpx

from weather_bets.models import CityConfig

logger = logging.getLogger(__name__)

# NOAA Climate Data Online (CDO) API
NOAA_CDO_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
# For Austin Bergstrom, we use the GHCND station
STATION_MAP = {
    "KAUS": "GHCND:USW00013904",  # Austin-Bergstrom
}


async def fetch_historical_temps(
    city: CityConfig,
    days_back: int = 365,
    noaa_token: str | None = None,
) -> list[dict]:
    """
    Fetch historical daily high/low temperatures from NOAA.
    
    Returns list of {date, tmax, tmin} dicts.
    
    Note: NOAA CDO API requires a free token from
    https://www.ncdc.noaa.gov/cdo-web/token
    Falls back to NWS observations if no token.
    """
    # Use NWS recent observations as fallback (last 7 days only)
    return await _fetch_nws_recent(city)


async def _fetch_nws_recent(city: CityConfig) -> list[dict]:
    """
    Fetch recent observations from the NWS station.
    This gives us the last ~7 days of actual recorded temperatures
    which we can compare against forecasts.
    
    IMPORTANT accuracy notes:
    - NWS API reports temperature as INTEGER Celsius
    - We convert to Fahrenheit and ROUND to nearest whole degree,
      matching NWS/Kalshi CLI report formatting
    - Observations are grouped by LOCAL date (CDT/CST), not UTC
    - Kalshi settles on the NWS CLI which uses local standard time
    """
    url = f"https://api.weather.gov/stations/{city.station_id}/observations"

    async with httpx.AsyncClient(
        timeout=15.0,
        headers={"User-Agent": "(weather-bets)", "Accept": "application/geo+json"},
    ) as http:
        resp = await http.get(url, params={"limit": 168})  # ~7 days hourly
        resp.raise_for_status()
        data = resp.json()

    features = data.get("features", [])

    # Timezone offset for this city (hours from UTC)
    # Austin CDT = UTC-5, CST = UTC-6. Use CDT during DST.
    tz_offsets = {
        "KAUS": -5,  # CDT (March-November)
        "KLAX": -7,  # PDT
        "KORD": -5,  # CDT
        "KMIA": -4,  # EDT
        "KDEN": -6,  # MDT
    }
    tz_offset = timedelta(hours=tz_offsets.get(city.station_id, -5))

    # Group by LOCAL date, find max/min temp per day
    daily: dict[str, list[int]] = {}
    for obs in features:
        props = obs.get("properties", {})
        ts = props.get("timestamp", "")
        temp_c = props.get("temperature", {}).get("value")

        if not ts or temp_c is None:
            continue

        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            # Convert to local time for date grouping
            local_dt = dt + tz_offset
            date_str = local_dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        # Convert C→F and ROUND to nearest integer
        # This matches NWS CLI report format (whole degrees F)
        # 33°C → 91.4°F → 91°F (not 91.4)
        temp_f = round(temp_c * 9 / 5 + 32)
        daily.setdefault(date_str, []).append(temp_f)

    results = []
    for date_str in sorted(daily.keys()):
        temps = daily[date_str]
        results.append({
            "date": date_str,
            "tmax": max(temps),
            "tmin": min(temps),
            "obs_count": len(temps),
        })

    logger.info(
        f"[Historical] {city.name}: {len(results)} days of observations from {city.station_id}"
    )
    for r in results[-7:]:
        logger.info(f"  {r['date']}: high={r['tmax']}°F low={r['tmin']}°F")

    return results


def analyze_forecast_bias(
    historical: list[dict],
    forecasts: list[dict],
) -> dict:
    """
    DEPRECATED: This function had a bug — it compared today's forecast against
    today's observation (same date), but forecasts are for *future* dates so
    there's no meaningful overlap.

    Use forecast_tracker.get_accuracy_stats() instead for accurate bias data.
    This stub is retained for backward compatibility with the scan_state display.

    Returns a zeroed-out stats dict with sample_size=0 to suppress stale bias output.
    """
    logger.debug(
        "[Bias] analyze_forecast_bias() is deprecated — use forecast_tracker.get_accuracy_stats()"
    )
    return {"mean_error": 0.0, "abs_error": 0.0, "std_error": 0.0, "sample_size": 0}
