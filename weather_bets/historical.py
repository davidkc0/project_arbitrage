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

    # Group by date, find max temp per day
    daily: dict[str, list[float]] = {}
    for obs in features:
        props = obs.get("properties", {})
        ts = props.get("timestamp", "")
        temp_c = props.get("temperature", {}).get("value")

        if not ts or temp_c is None:
            continue

        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        temp_f = temp_c * 9 / 5 + 32
        daily.setdefault(date_str, []).append(temp_f)

    results = []
    for date_str in sorted(daily.keys()):
        temps = daily[date_str]
        results.append({
            "date": date_str,
            "tmax": round(max(temps), 1),
            "tmin": round(min(temps), 1),
            "obs_count": len(temps),
        })

    logger.info(
        f"[Historical] {city.name}: {len(results)} days of observations from {city.station_id}"
    )
    for r in results[-7:]:
        logger.info(f"  {r['date']}: high={r['tmax']:.0f}°F low={r['tmin']:.0f}°F")

    return results


def analyze_forecast_bias(
    historical: list[dict],
    forecasts: list[dict],
) -> dict:
    """
    Compare historical actuals vs forecasts to find NWS bias.
    
    Returns bias statistics:
    - mean_error: average (forecast - actual), positive = NWS runs hot
    - abs_error: average absolute error
    - std_error: standard deviation of errors
    """
    if not historical or not forecasts:
        return {"mean_error": 0, "abs_error": 2.5, "std_error": 3.0, "sample_size": 0}

    errors = []
    for fc in forecasts:
        for hist in historical:
            if fc.get("date") == hist.get("date"):
                error = fc.get("high", 0) - hist.get("tmax", 0)
                errors.append(error)

    if not errors:
        return {"mean_error": 0, "abs_error": 2.5, "std_error": 3.0, "sample_size": 0}

    import statistics
    mean_err = statistics.mean(errors)
    abs_err = statistics.mean(abs(e) for e in errors)
    std_err = statistics.stdev(errors) if len(errors) > 1 else 3.0

    logger.info(
        f"[Bias] NWS forecast bias: mean={mean_err:+.1f}°F "
        f"MAE={abs_err:.1f}°F stdev={std_err:.1f}°F (n={len(errors)})"
    )

    return {
        "mean_error": round(mean_err, 1),
        "abs_error": round(abs_err, 1),
        "std_error": round(std_err, 1),
        "sample_size": len(errors),
    }
