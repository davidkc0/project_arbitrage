"""NWS API client — fetches weather forecasts for temperature betting."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx

from weather_bets.models import CityConfig, ForecastData

logger = logging.getLogger(__name__)

NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {
    "User-Agent": "(weather-bets, contact@example.com)",
    "Accept": "application/geo+json",
}


async def fetch_forecast(city: CityConfig) -> list[ForecastData]:
    """
    Fetch the NWS 7-day forecast for a city.
    Returns a ForecastData for each daytime period (today, tomorrow, etc.).
    """
    url = f"{NWS_BASE}/gridpoints/{city.nws_office}/{city.nws_grid_x},{city.nws_grid_y}/forecast"

    async with httpx.AsyncClient(timeout=15.0, headers=NWS_HEADERS) as http:
        resp = await http.get(url)
        resp.raise_for_status()
        data = resp.json()

    periods = data.get("properties", {}).get("periods", [])
    forecasts: list[ForecastData] = []

    for period in periods:
        if not period.get("isDaytime", False):
            continue

        start = period.get("startTime", "")
        # Parse date from ISO string
        try:
            dt = datetime.fromisoformat(start)
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        forecasts.append(ForecastData(
            city=city.code,
            date=date_str,
            high_temp_f=period.get("temperature", 0),
            short_forecast=period.get("shortForecast", ""),
            detailed_forecast=period.get("detailedForecast", ""),
            wind_speed=period.get("windSpeed", ""),
            precip_probability=period.get("probabilityOfPrecipitation", {}).get("value", 0) or 0,
        ))

    logger.info(f"[NWS] Fetched {len(forecasts)} daytime forecasts for {city.name}")
    return forecasts


async def fetch_hourly_forecast(city: CityConfig, target_date: str) -> list[dict]:
    """
    Fetch hourly forecast for a specific date to get temperature distribution.
    Returns list of hourly temperature records for the target date.
    """
    url = f"{NWS_BASE}/gridpoints/{city.nws_office}/{city.nws_grid_x},{city.nws_grid_y}/forecast/hourly"

    async with httpx.AsyncClient(timeout=15.0, headers=NWS_HEADERS) as http:
        resp = await http.get(url)
        resp.raise_for_status()
        data = resp.json()

    periods = data.get("properties", {}).get("periods", [])
    hourly_temps: list[dict] = []

    for period in periods:
        start = period.get("startTime", "")
        try:
            dt = datetime.fromisoformat(start)
            if dt.strftime("%Y-%m-%d") == target_date:
                hourly_temps.append({
                    "hour": dt.strftime("%H:%M"),
                    "temp_f": period.get("temperature", 0),
                    "wind": period.get("windSpeed", ""),
                    "forecast": period.get("shortForecast", ""),
                })
        except (ValueError, TypeError):
            continue

    if hourly_temps:
        temps = [h["temp_f"] for h in hourly_temps]
        logger.info(
            f"[NWS] Hourly temps for {city.name} on {target_date}: "
            f"range {min(temps)}-{max(temps)}°F ({len(hourly_temps)} hours)"
        )

    return hourly_temps
