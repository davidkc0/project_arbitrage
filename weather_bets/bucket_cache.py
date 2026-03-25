"""In-memory cache for Kalshi temperature buckets.

Bucket *structure* (labels, bounds, tickers) is static for the day.
Only *prices* change.  This module fetches once per city/date, and
throttles price refreshes to a configurable TTL (default 5 min).

NWS temperature polling is NOT affected — this only caches Kalshi API calls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from weather_bets.models import CityConfig, TemperatureBucket

logger = logging.getLogger(__name__)

# Minimum seconds between price refreshes for the same city/date
PRICE_REFRESH_TTL = 300  # 5 minutes


@dataclass
class _CacheEntry:
    buckets: list[TemperatureBucket]
    fetched_at: float  # time.time() of last full fetch
    prices_refreshed_at: float  # time.time() of last price refresh


# Key = (city_code, date_str)
_cache: dict[tuple[str, str], _CacheEntry] = {}


def get_buckets(city_code: str, date: str) -> list[TemperatureBucket] | None:
    """Return cached buckets or None if not cached."""
    entry = _cache.get((city_code, date))
    if entry:
        return entry.buckets
    return None


def set_buckets(
    city_code: str, date: str, buckets: list[TemperatureBucket]
) -> None:
    """Store buckets in the cache."""
    now = time.time()
    _cache[(city_code, date)] = _CacheEntry(
        buckets=buckets,
        fetched_at=now,
        prices_refreshed_at=now,
    )
    logger.info(
        f"[BucketCache] Cached {len(buckets)} buckets for {city_code}/{date}"
    )


async def get_or_fetch(
    city_code: str,
    city_config: CityConfig,
    date: str,
) -> list[TemperatureBucket]:
    """Return cached buckets, fetching from Kalshi if not yet cached today."""
    cached = get_buckets(city_code, date)
    if cached:
        return cached

    from weather_bets.kalshi_weather import fetch_weather_markets

    buckets = await fetch_weather_markets(city_config, target_date=date)
    if buckets:
        set_buckets(city_code, date, buckets)
    return buckets


async def refresh_prices(
    city_code: str,
    city_config: CityConfig,
    date: str,
    force: bool = False,
) -> list[TemperatureBucket]:
    """Re-fetch fresh prices from Kalshi, respecting TTL.

    Returns updated buckets.  If TTL hasn't elapsed and force=False,
    returns the existing cached buckets without an API call.
    """
    entry = _cache.get((city_code, date))
    now = time.time()

    if entry and not force:
        elapsed = now - entry.prices_refreshed_at
        if elapsed < PRICE_REFRESH_TTL:
            logger.debug(
                f"[BucketCache] Price refresh skipped for {city_code}/{date} "
                f"({elapsed:.0f}s < {PRICE_REFRESH_TTL}s TTL)"
            )
            return entry.buckets

    from weather_bets.kalshi_weather import fetch_weather_markets

    buckets = await fetch_weather_markets(city_config, target_date=date)
    if buckets:
        set_buckets(city_code, date, buckets)
        logger.info(
            f"[BucketCache] Refreshed prices for {city_code}/{date}"
        )
        return buckets

    # API failed — return stale data if available
    if entry:
        logger.warning(
            f"[BucketCache] Price refresh failed for {city_code}/{date}, "
            f"using stale cache"
        )
        return entry.buckets
    return []


def invalidate(city_code: str | None = None, date: str | None = None) -> None:
    """Clear cache entries.

    - Both None: clear everything
    - city_code only: clear all dates for that city
    - date only: clear all cities for that date
    - Both: clear specific entry
    """
    if city_code and date:
        _cache.pop((city_code, date), None)
    elif city_code:
        keys = [k for k in _cache if k[0] == city_code]
        for k in keys:
            del _cache[k]
    elif date:
        keys = [k for k in _cache if k[1] == date]
        for k in keys:
            del _cache[k]
    else:
        _cache.clear()
    logger.info(
        f"[BucketCache] Invalidated cache "
        f"(city={city_code or '*'}, date={date or '*'})"
    )
