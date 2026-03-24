"""Weather data summary — aggregates all data sources into a human-readable printout.

Collects:
- Current temperature (Synoptic + IEM)
- Day max so far
- CLEAR_DAY_PATTERNS predicted high from current hour
- NWS daily forecast
- GFS/ICON/GEM model forecasts
- NWS hourly forecast (remaining hours)
- Yesterday's actual high
- Kalshi bucket prices
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from weather_bets.bet_engine import CLEAR_DAY_PATTERNS

logger = logging.getLogger(__name__)


def build_weather_summary(
    city_name: str,
    current_temp_f: float | None,
    day_max_f: float | None,
    current_hour_ct: int,
    month: int,
    nws_forecast_high: int | None,
    model_forecasts: dict | None = None,  # {"GFS": 85.0, "ICON": 86.3, ...}
    consensus_high: float | None = None,
    hourly_forecast: list[dict] | None = None,  # [{"hour": 13, "temp_f": 83}, ...]
    yesterday_high: float | None = None,
    sky_conditions: str = "",
    buckets: list | None = None,  # list of TemperatureBucket
    iem_high: float | None = None,
    iem_source: str = "IEM_DSM",
) -> str:
    """Build a comprehensive weather data summary for logging and Claude."""

    lines = []
    now_ct = datetime.now(timezone(timedelta(hours=-5)))
    lines.append(f"{'='*60}")
    lines.append(f"AUSTIN WEATHER DATA — {now_ct.strftime('%Y-%m-%d %I:%M %p')} CT")
    lines.append(f"{'='*60}")

    # ── Current Conditions ──
    lines.append("")
    lines.append("CURRENT CONDITIONS")
    if current_temp_f is not None:
        lines.append(f"  Temperature: {current_temp_f:.1f}°F")
    if day_max_f is not None:
        lines.append(f"  Day max so far: {day_max_f:.0f}°F")
    if iem_high is not None:
        lines.append(f"  IEM DSM reported high: {iem_high:.0f}°F")
    if sky_conditions:
        lines.append(f"  Sky: {sky_conditions}")
    lines.append(f"  Current hour (CT): {current_hour_ct}:00")

    # ── Historical Pattern ──
    pattern = CLEAR_DAY_PATTERNS.get(month, {}).get(current_hour_ct)
    if pattern and day_max_f is not None:
        avg_rise, p10_rise, p90_rise, n_samples = pattern
        pred_high = day_max_f + avg_rise
        pred_low = day_max_f + p10_rise
        pred_high_bound = day_max_f + p90_rise

        lines.append("")
        lines.append(f"HISTORICAL PATTERN (Month {month}, {current_hour_ct}:00 CT)")
        lines.append(f"  Avg remaining rise: +{avg_rise:.1f}°F → predicted high: {pred_high:.0f}°F")
        lines.append(f"  P10 remaining rise: +{p10_rise}°F → low estimate:  {pred_low:.0f}°F")
        lines.append(f"  P90 remaining rise: +{p90_rise}°F → high estimate: {pred_high_bound:.0f}°F")
        lines.append(f"  Based on {n_samples} clear days in month {month} over 5 years")
    elif pattern and current_temp_f is not None:
        avg_rise, p10_rise, p90_rise, n_samples = pattern
        pred_high = current_temp_f + avg_rise
        lines.append("")
        lines.append(f"HISTORICAL PATTERN (Month {month}, {current_hour_ct}:00 CT)")
        lines.append(f"  Avg remaining rise: +{avg_rise:.1f}°F → predicted high: {pred_high:.0f}°F")
        lines.append(f"  Based on {n_samples} clear days")
    else:
        lines.append("")
        lines.append("HISTORICAL PATTERN")
        lines.append(f"  No pattern data for month={month} hour={current_hour_ct}")

    # ── Forecasts ──
    lines.append("")
    lines.append("FORECASTS")
    if nws_forecast_high is not None:
        lines.append(f"  NWS:       {nws_forecast_high}°F")
    if model_forecasts:
        for model_name, temp in model_forecasts.items():
            lines.append(f"  {model_name:9s} {temp:.1f}°F")
    if consensus_high is not None:
        lines.append(f"  Consensus: {consensus_high:.0f}°F")

    # ── Hourly Forecast ──
    if hourly_forecast:
        # Only show remaining hours — hour field may be int, "12", or "12:00"
        def parse_hour(h):
            raw = h.get("hour", 0)
            if isinstance(raw, int):
                return raw
            return int(str(raw).split(":")[0])

        remaining = [h for h in hourly_forecast if parse_hour(h) >= current_hour_ct]
        if remaining:
            lines.append("")
            lines.append("NWS HOURLY (remaining)")
            hourly_str = " → ".join(
                f"{parse_hour(h)}:00={h['temp_f']:.0f}°F"
                for h in remaining[:8]  # max 8 hours
            )
            lines.append(f"  {hourly_str}")

    # ── Yesterday ──
    if yesterday_high is not None:
        lines.append("")
        lines.append("YESTERDAY")
        lines.append(f"  Actual high: {yesterday_high:.0f}°F")

    # ── Kalshi Markets ──
    if buckets:
        lines.append("")
        lines.append("KALSHI MARKETS (today)")
        # Find market favorite
        fav = max(buckets, key=lambda b: b.yes_price)
        for b in buckets:
            marker = "  ★ market favorite" if b.ticker == fav.ticker else ""
            lines.append(f"  {b.label:12s} YES=${b.yes_price:.2f}  NO=${b.no_price:.2f}{marker}")

    lines.append(f"{'='*60}")

    summary = "\n".join(lines)

    # Log it
    for line in lines:
        logger.info(f"[Summary] {line}")

    return summary
