"""Open-Meteo forecast fetcher — free second-opinion weather data.

Uses the same underlying models as Apple Weather (HRRR, GFS, etc.)
No API key required. Provides multi-model ensemble forecasts.
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx

from weather_bets.models import CityConfig

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


async def fetch_open_meteo_forecast(city: CityConfig) -> list[dict]:
    """
    Fetch daily high temperature forecast from Open-Meteo.
    
    Returns list of {date, high_f, low_f, model} dicts for next 7 days.
    """
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone": "America/Chicago",
        "forecast_days": 7,
    }

    async with httpx.AsyncClient(timeout=15.0) as http:
        resp = await http.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    highs = daily.get("temperature_2m_max", [])
    lows = daily.get("temperature_2m_min", [])

    results = []
    for d, h, lo in zip(dates, highs, lows):
        results.append({
            "date": d,
            "high_f": round(h, 1),
            "low_f": round(lo, 1),
            "source": "Open-Meteo",
        })

    if results:
        logger.info(f"[Open-Meteo] {city.name}: {len(results)} day forecast")
        for r in results[:3]:
            logger.info(f"  {r['date']}: high={r['high_f']}°F low={r['low_f']}°F")

    return results


async def fetch_multi_model_forecast(city: CityConfig) -> list[dict]:
    """
    Fetch forecasts from multiple weather models for comparison.
    
    Models:
    - GFS (NOAA) — same base as NWS
    - HRRR (NOAA High-Res) — better for short-range
    - ICON (German) — independent model
    - GEM (Canadian) — independent model
    
    Returns list of {date, model, high_f} for model comparison.
    """
    models = {
        "gfs_seamless": "GFS (NOAA)",
        "icon_seamless": "ICON (German)",
        "gem_seamless": "GEM (Canadian)",
    }

    all_forecasts = []

    for model_id, model_name in models.items():
        try:
            params = {
                "latitude": city.lat,
                "longitude": city.lon,
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": "America/Chicago",
                "forecast_days": 7,
                "models": model_id,
            }

            async with httpx.AsyncClient(timeout=15.0) as http:
                resp = await http.get(OPEN_METEO_URL, params=params)

                if resp.status_code != 200:
                    logger.warning(f"[Open-Meteo] {model_name} failed: {resp.status_code}")
                    continue

                data = resp.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            highs = daily.get("temperature_2m_max", [])

            for d, h in zip(dates, highs):
                if h is not None:
                    all_forecasts.append({
                        "date": d,
                        "model": model_name,
                        "high_f": round(h, 1),
                    })

        except Exception as e:
            logger.warning(f"[Open-Meteo] {model_name} error: {e}")

    if all_forecasts:
        # Group by date and show consensus
        from collections import defaultdict
        by_date = defaultdict(list)
        for f in all_forecasts:
            by_date[f["date"]].append(f)

        for date in sorted(by_date.keys())[:3]:
            entries = by_date[date]
            temps = [e["high_f"] for e in entries]
            avg = sum(temps) / len(temps) if temps else 0
            spread = max(temps) - min(temps) if len(temps) > 1 else 0
            models_str = ", ".join(f"{e['model']}={e['high_f']}°F" for e in entries)
            logger.info(
                f"[Models] {date}: avg={avg:.0f}°F spread={spread:.0f}°F — {models_str}"
            )

    return all_forecasts


def build_consensus(
    nws_high: float,
    open_meteo_forecasts: list[dict],
    target_date: str,
) -> dict:
    """
    Build a consensus forecast from multiple sources.
    
    Returns {consensus_high, nws_high, alt_high, spread, sources}
    """
    alt_temps = [
        f["high_f"] for f in open_meteo_forecasts
        if f.get("date") == target_date
    ]

    if not alt_temps:
        return {
            "consensus_high": nws_high,
            "nws_high": nws_high,
            "alt_high": None,
            "alt_avg": None,
            "spread": 0,
            "sources": ["NWS"],
            "model_details": [],
        }

    alt_avg = sum(alt_temps) / len(alt_temps)
    all_temps = [nws_high] + alt_temps
    consensus = sum(all_temps) / len(all_temps)
    spread = max(all_temps) - min(all_temps)

    model_details = [
        f for f in open_meteo_forecasts if f.get("date") == target_date
    ]

    logger.info(
        f"[Consensus] NWS={nws_high}°F, Alt avg={alt_avg:.0f}°F → "
        f"Consensus={consensus:.0f}°F (spread={spread:.0f}°F)"
    )

    return {
        "consensus_high": round(consensus, 1),
        "nws_high": nws_high,
        "alt_high": round(alt_avg, 1),
        "alt_avg": round(alt_avg, 1),
        "spread": round(spread, 1),
        "sources": ["NWS"] + [f["model"] for f in model_details],
        "model_details": model_details,
    }
