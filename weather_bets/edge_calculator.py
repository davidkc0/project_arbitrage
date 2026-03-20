"""Edge calculator — builds spread bets across adjacent temperature buckets.

Strategy: Instead of betting on a single bucket, we build "spreads" by
buying YES on 2-3 adjacent buckets that cover the likely temperature range.
This gives higher certainty (70-80%) with steady returns.

Think of it like an options iron condor — we're covering a range and
profiting as long as the actual temp lands anywhere within it.
"""

from __future__ import annotations

import logging
import math

from weather_bets.config import FORECAST_STDEV_24H, FORECAST_STDEV_48H, MIN_EDGE
from weather_bets.models import (
    EdgeOpportunity,
    ForecastData,
    SpreadBet,
    TemperatureBucket,
)

logger = logging.getLogger(__name__)


def calculate_spreads(
    forecast: ForecastData,
    buckets: list[TemperatureBucket],
    days_ahead: int = 1,
) -> list[SpreadBet]:
    """
    Build spread bets centered on the NWS forecast.
    
    Creates multiple spread options:
    - 2-bucket spread (forecast bucket + most likely adjacent)
    - 3-bucket spread (forecast bucket + both adjacents)
    
    Returns spreads sorted by expected profit.
    """
    stdev = FORECAST_STDEV_24H if days_ahead <= 1 else FORECAST_STDEV_48H
    mean = forecast.high_temp_f

    # Calculate probability for every bucket
    bucket_probs: list[tuple[TemperatureBucket, float]] = []
    for b in buckets:
        prob = _bucket_probability(mean, stdev, b.low_bound, b.high_bound)
        bucket_probs.append((b, prob))

    # Find the forecast bucket (which bucket contains the forecast temp)
    forecast_idx = _find_forecast_bucket(buckets, mean)

    logger.info(
        f"[Spread] Forecast: {forecast.city} {forecast.date} = {mean}°F ± {stdev}°F"
    )
    for i, (b, p) in enumerate(bucket_probs):
        marker = " ★ FORECAST" if i == forecast_idx else ""
        logger.info(
            f"  {b.label:12} | prob={p:.1%}  YES=${b.yes_price:.2f}{marker}"
        )

    if forecast_idx is None:
        logger.warning("[Spread] Could not identify forecast bucket")
        return []

    spreads: list[SpreadBet] = []

    # Generate spread combinations centered on the forecast bucket
    # 1-bucket: just the forecast bucket
    # 2-bucket: forecast + adjacent (left or right, whichever has higher prob)
    # 3-bucket: forecast + both adjacents
    for width_name, indices in _spread_windows(forecast_idx, len(buckets)):
        selected = [(bucket_probs[i][0], bucket_probs[i][1]) for i in indices]
        spread_buckets = [s[0] for s in selected]
        spread_probs = [s[1] for s in selected]

        total_prob = sum(spread_probs)
        total_cost = sum(b.yes_price for b in spread_buckets)

        # Only one bucket pays out $1.00, but total cost includes all legs
        # Profit = $1.00 - total_cost (when any one bucket hits)
        profit_if_hit = 1.00 - total_cost
        
        # Expected profit = P(hit) * profit - P(miss) * cost
        # But since only ONE bucket can win, we pay for all and get $1 back
        expected_profit = total_prob * 1.00 - total_cost

        roi = (expected_profit / total_cost * 100) if total_cost > 0 else 0

        labels = " + ".join(b.label for b in spread_buckets)
        logger.info(
            f"  [{width_name}] {labels}: "
            f"prob={total_prob:.1%} cost=${total_cost:.2f} "
            f"profit=${profit_if_hit:.2f} EV=${expected_profit:.3f} ROI={roi:+.1f}%"
        )

        spread = SpreadBet(
            city=forecast.city,
            date=forecast.date,
            forecast_high=mean,
            buckets=spread_buckets,
            bucket_probabilities=spread_probs,
            total_probability=total_prob,
            total_cost=total_cost,
            profit_if_hit=profit_if_hit,
            expected_profit=expected_profit,
            roi_percent=roi,
            forecast_detail=forecast.detailed_forecast,
        )
        spreads.append(spread)

    # Sort by expected profit descending
    spreads.sort(key=lambda s: s.expected_profit, reverse=True)

    return spreads


def _spread_windows(
    center: int, total: int
) -> list[tuple[str, list[int]]]:
    """Generate spread window combinations centered on the forecast bucket."""
    windows = []

    # 1-bucket: just the forecast
    windows.append(("1-bucket", [center]))

    # 2-bucket spreads
    if center > 0:
        windows.append(("2-bucket left", [center - 1, center]))
    if center < total - 1:
        windows.append(("2-bucket right", [center, center + 1]))

    # 3-bucket: forecast + both adjacents
    if center > 0 and center < total - 1:
        windows.append(("3-bucket", [center - 1, center, center + 1]))

    # Wide 4-bucket if possible
    if center > 0 and center < total - 2:
        windows.append(("4-bucket wide", [center - 1, center, center + 1, center + 2]))
    elif center > 1 and center < total - 1:
        windows.append(("4-bucket wide", [center - 2, center - 1, center, center + 1]))

    return windows


def _find_forecast_bucket(buckets: list[TemperatureBucket], temp: float) -> int | None:
    """Find which bucket index contains the forecast temperature."""
    for i, b in enumerate(buckets):
        if b.low_bound is not None and b.high_bound is not None:
            if b.low_bound <= temp <= b.high_bound:
                return i
        elif b.low_bound is None and b.high_bound is not None:
            if temp <= b.high_bound:
                return i
        elif b.low_bound is not None and b.high_bound is None:
            if temp >= b.low_bound:
                return i
    return None


# ── Keep the single-bucket edge calculation for backward compat ─────────

def calculate_edges(
    forecast: ForecastData,
    buckets: list[TemperatureBucket],
    days_ahead: int = 1,
) -> list[EdgeOpportunity]:
    """Calculate per-bucket edges (used for dashboard display)."""
    stdev = FORECAST_STDEV_24H if days_ahead <= 1 else FORECAST_STDEV_48H
    mean = forecast.high_temp_f
    opportunities = []

    for bucket in buckets:
        our_prob = _bucket_probability(mean, stdev, bucket.low_bound, bucket.high_bound)
        market_prob = bucket.yes_price
        edge = our_prob - market_prob
        ev = our_prob * (1.0 - bucket.yes_price) - (1.0 - our_prob) * bucket.yes_price

        if edge > 0 and our_prob >= 0.10:
            opportunities.append(EdgeOpportunity(
                city=forecast.city, date=forecast.date, bucket=bucket,
                forecast_high=mean, our_probability=our_prob,
                market_probability=market_prob, edge=edge,
                edge_percent=edge * 100, expected_value=ev,
                forecast_detail=forecast.detailed_forecast,
            ))

    opportunities.sort(key=lambda o: o.our_probability, reverse=True)
    return opportunities


# ── Math helpers ────────────────────────────────────────────────────────

def _bucket_probability(
    mean: float, stdev: float, low: float | None, high: float | None
) -> float:
    """Probability that temp falls in [low, high] given N(mean, stdev)."""
    if low is None and high is not None:
        return _norm_cdf(high + 0.5, mean, stdev)
    elif low is not None and high is not None:
        return _norm_cdf(high + 0.5, mean, stdev) - _norm_cdf(low - 0.5, mean, stdev)
    elif low is not None and high is None:
        return 1.0 - _norm_cdf(low - 0.5, mean, stdev)
    return 0.0


def _norm_cdf(x: float, mean: float, stdev: float) -> float:
    if stdev <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (stdev * math.sqrt(2))))
