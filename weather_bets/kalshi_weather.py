"""Kalshi weather market scanner — fetches KXHIGH / KXLOW temperature markets."""

from __future__ import annotations

import logging
import re

import httpx

from weather_bets.models import CityConfig, TemperatureBucket

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


async def fetch_weather_markets(
    city: CityConfig,
    target_date: str | None = None,
) -> list[TemperatureBucket]:
    """
    Fetch all temperature range buckets for a city from Kalshi.
    
    Args:
        city: City configuration
        target_date: Optional filter like "2026-03-20" to only get that day's markets
    
    Returns:
        List of TemperatureBucket objects with current pricing
    """
    async with httpx.AsyncClient(base_url=KALSHI_BASE, timeout=30.0) as http:
        resp = await http.get("/events", params={
            "limit": 100,
            "status": "open",
            "with_nested_markets": "true",
            "series_ticker": city.kalshi_series,
        })
        resp.raise_for_status()
        data = resp.json()

    buckets: list[TemperatureBucket] = []
    events = data.get("events", [])

    for event in events:
        event_ticker = event.get("event_ticker", "")

        # Filter by date if specified (ticker format: KXHIGHAUS-26MAR20)
        if target_date:
            # Convert "2026-03-20" to "26MAR20"
            from datetime import datetime
            dt = datetime.strptime(target_date, "%Y-%m-%d")
            date_code = dt.strftime("%y%b%d").upper()  # "26MAR20"
            if date_code not in event_ticker.upper():
                continue

        for market in event.get("markets", []):
            ticker = market.get("ticker", "")
            title = market.get("title", "")

            # Parse temperature range from title
            low_bound, high_bound, label = _parse_temp_range(title)

            # Get pricing
            yes_bid = market.get("yes_bid_dollars")
            if yes_bid is None:
                yes_bid = (market.get("yes_bid") or 0) / 100.0
            else:
                yes_bid = float(yes_bid)

            yes_ask = market.get("yes_ask_dollars")
            if yes_ask is None:
                yes_ask = (market.get("yes_ask") or 0) / 100.0
            else:
                yes_ask = float(yes_ask)

            no_bid = market.get("no_bid_dollars")
            if no_bid is None:
                no_bid = (market.get("no_bid") or 0) / 100.0
            else:
                no_bid = float(no_bid)

            no_ask = market.get("no_ask_dollars")
            if no_ask is None:
                no_ask = (market.get("no_ask") or 0) / 100.0
            else:
                no_ask = float(no_ask)

            # Use ask prices for buying (what we'd actually pay)
            yes_price = yes_ask if yes_ask > 0 else yes_bid
            no_price = no_ask if no_ask > 0 else no_bid

            volume = float(market.get("volume", 0))
            close_time = market.get("close_time", "")

            bucket = TemperatureBucket(
                ticker=ticker,
                label=label,
                low_bound=low_bound,
                high_bound=high_bound,
                yes_price=yes_price,
                no_price=no_price,
                volume=volume,
                close_time=close_time,
            )
            buckets.append(bucket)

    # Sort by low_bound
    buckets.sort(key=lambda b: b.low_bound if b.low_bound is not None else -999)

    if buckets:
        logger.info(
            f"[Kalshi] {city.name}: {len(buckets)} temperature buckets for "
            f"{target_date or 'all dates'}"
        )
        for b in buckets:
            logger.info(f"  {b.label:15} YES=${b.yes_price:.2f}  NO=${b.no_price:.2f}")

    return buckets


def _parse_temp_range(title: str) -> tuple[float | None, float | None, str]:
    """
    Parse temperature range from Kalshi market title.
    
    Examples:
        "Will the high temp in Austin be <88°" → (None, 87, "≤87°")
        "Will the high temp in Austin be 90-91°" → (90, 91, "90-91°")
        "Will the high temp in Austin be ≥96°" → (96, None, "≥96°")
    """
    title_lower = title.lower()

    # Pattern: "be <88°" or "less than 88" or "be ≤87°"
    m = re.search(r'[<≤]\s*(\d+)', title)
    if m:
        val = float(m.group(1))
        # "< 88" means 87 or below
        if '<' in title and '≤' not in title:
            return (None, val - 1, f"≤{int(val-1)}°")
        return (None, val, f"≤{int(val)}°")

    # Pattern: "be 90-91°" or "between 90-91" 
    m = re.search(r'(\d+)\s*[-–]\s*(\d+)', title)
    if m:
        low = float(m.group(1))
        high = float(m.group(2))
        return (low, high, f"{int(low)}-{int(high)}°")

    # Pattern: "be ≥96°" or ">= 96" or "96° or above"
    m = re.search(r'[>≥]\s*(\d+)', title)
    if m:
        val = float(m.group(1))
        return (val, None, f"≥{int(val)}°")

    # Pattern: "above" with number
    m = re.search(r'(\d+).*(?:or above|or more|or higher)', title_lower)
    if m:
        val = float(m.group(1))
        return (val, None, f"≥{int(val)}°")

    # Fallback
    return (None, None, title[:20])
