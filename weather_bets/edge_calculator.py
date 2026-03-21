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

from weather_bets.config import FORECAST_STDEV_24H, FORECAST_STDEV_48H, MIN_EDGE, MARKET_CONVICTION_THRESHOLD
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
    current_observed_high: float | None = None,
    hourly_forecast: list[dict] | None = None,
) -> list[SpreadBet]:
    """
    Build spread bets centered on the NWS forecast.
    
    Creates multiple spread options:
    - 2-bucket spread (forecast bucket + most likely adjacent)
    - 3-bucket spread (forecast bucket + both adjacents)
    
    For same-day bets: uses current observed temp + hourly forecast to
    determine a realistic floor and tighter distribution. If it's 88°F
    at 1 PM and hourly says 90°F peak at 4 PM, the floor is 88°F but
    there's still significant upside — the distribution shifts accordingly.
    
    Returns spreads sorted by expected profit.
    """
    stdev = FORECAST_STDEV_24H if days_ahead <= 1 else FORECAST_STDEV_48H
    mean = forecast.high_temp_f
    
    temp_floor = None
    if current_observed_high is not None and days_ahead <= 1:
        temp_floor = current_observed_high
        
        # Use hourly forecast to estimate remaining heating potential
        hourly_peak = None
        if hourly_forecast:
            hourly_temps = [h["temp_f"] for h in hourly_forecast]
            if hourly_temps:
                hourly_peak = max(hourly_temps)
        
        if hourly_peak and hourly_peak > current_observed_high:
            remaining_rise = hourly_peak - current_observed_high
            logger.info(
                f"[Spread] Current: {current_observed_high}°F | "
                f"Hourly NWS peak: {hourly_peak}°F | "
                f"Consensus forecast: {mean}°F"
            )
            # Hourly peak is from NWS which runs hot. Use the LOWER of
            # hourly peak and consensus forecast as our mean — this avoids
            # inheriting NWS hot bias through the hourly data.
            # But never go below current observed (that's a hard floor).
            effective_mean = min(hourly_peak, mean)
            effective_mean = max(effective_mean, current_observed_high)
            logger.info(
                f"[Spread] Effective mean: min(hourly={hourly_peak}, "
                f"consensus={mean}) = {effective_mean}°F"
            )
            mean = effective_mean
            # Tighten stdev — intraday we have better info, but hourly
            # forecasts can still be off by 1-3°F
            stdev = min(stdev, 2.5)
        elif current_observed_high >= mean:
            # Already at or past the forecast — likely near or past peak
            logger.info(
                f"[Spread] ⚠️ Current observed ({current_observed_high}°F) "
                f"≥ consensus ({mean}°F) — may have peaked or still rising slightly"
            )
            mean = current_observed_high + 1
            stdev = min(stdev, 2.0)  # Very tight — we're near the actual high
        
        logger.info(
            f"[Spread] Same-day adjustment: floor={current_observed_high}°F, "
            f"effective mean={mean}°F, stdev={stdev}°F"
        )

    # Calculate probability for every bucket
    bucket_probs: list[tuple[TemperatureBucket, float]] = []
    for b in buckets:
        prob = _bucket_probability(mean, stdev, b.low_bound, b.high_bound, temp_floor)
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
        # Fallback: use the bucket with the highest probability as center
        if bucket_probs:
            forecast_idx = max(range(len(bucket_probs)), key=lambda i: bucket_probs[i][1])
            logger.info(
                f"[Spread] Forecast temp ({mean}°F) outside bucket range — "
                f"using highest-prob bucket as center: {bucket_probs[forecast_idx][0].label}"
            )
        else:
            logger.warning("[Spread] No buckets available")
            return []

    spreads: list[SpreadBet] = []

    # Generate spread combinations centered on the forecast bucket
    # 1-bucket: just the forecast bucket
    # 2-bucket: forecast + adjacent (left or right, whichever has higher prob)
    # 3-bucket: forecast + both adjacents
    for width_name, indices in _spread_windows(forecast_idx, len(buckets)):
        selected = [(bucket_probs[i][0], bucket_probs[i][1]) for i in indices]

        # Prune dead-weight legs: drop any bucket that adds cost but <2% probability
        # (keeps the spread from paying for near-impossible outcomes)
        pruned = [(b, p) for b, p in selected if p >= 0.02 or b.yes_price <= 0.01]
        if len(pruned) < len(selected):
            dropped = [b.label for b, p in selected if (b, p) not in pruned]
            logger.info(f"  [{width_name}] Pruned dead-weight legs: {dropped}")
            selected = pruned
        if not selected:
            continue

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

        # ── Market sanity check ──
        passes, reason = market_sanity_check(spread_buckets, buckets, mean)
        if not passes:
            logger.warning(f"  [SANITY FAIL] {labels}: {reason}")
            continue

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


def market_sanity_check(
    spread_buckets: list[TemperatureBucket],
    all_buckets: list[TemperatureBucket],
    our_mean: float,
) -> tuple[bool, str]:
    """
    Check whether the market is pricing an adjacent bucket so strongly
    that we'd be fighting the tape by betting our spread.

    Returns (passes, reason). If passes=False, skip the spread.

    Logic: If any bucket NOT in our spread has a YES price above
    MARKET_CONVICTION_THRESHOLD, the market is highly confident the
    outcome will be there. We should only override this if our mean
    forecast is solidly on the other side.
    """
    spread_tickers = {b.ticker for b in spread_buckets}

    for b in all_buckets:
        if b.ticker in spread_tickers:
            continue  # This is a bucket we're betting on

        if b.yes_price >= MARKET_CONVICTION_THRESHOLD:
            # Market is very confident about this bucket. Check if our
            # forecast mean is actually closer to our spread than to this bucket.
            # Find the midpoint of the dominant bucket
            if b.low_bound is not None and b.high_bound is not None:
                bucket_mid = (b.low_bound + b.high_bound) / 2
            elif b.low_bound is None and b.high_bound is not None:
                bucket_mid = b.high_bound - 1
            elif b.low_bound is not None and b.high_bound is None:
                bucket_mid = b.low_bound + 1
            else:
                continue

            # Find midpoint of our spread range
            spread_lows = [b2.low_bound for b2 in spread_buckets if b2.low_bound is not None]
            spread_highs = [b2.high_bound for b2 in spread_buckets if b2.high_bound is not None]
            if not spread_lows or not spread_highs:
                continue
            spread_mid = (min(spread_lows) + max(spread_highs)) / 2

            dist_to_dominant = abs(our_mean - bucket_mid)
            dist_to_spread = abs(our_mean - spread_mid)

            if dist_to_dominant < dist_to_spread:
                reason = (
                    f"Market prices {b.label} at ${b.yes_price:.2f} "
                    f"({b.yes_price:.0%} confident) — our forecast ({our_mean}°F) "
                    f"is actually closer to that bucket than our spread. "
                    f"Skipping to avoid fighting the tape."
                )
                return False, reason

            # Our forecast is clearly on our side, but market conviction is high —
            # allow but warn
            logger.warning(
                f"[Sanity] Market strongly prices {b.label} @ ${b.yes_price:.2f} "
                f"but our mean ({our_mean}°F) favors our spread — proceeding with caution"
            )

    return True, "ok"


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
    current_observed_high: float | None = None,
    hourly_forecast: list[dict] | None = None,
) -> list[EdgeOpportunity]:
    """Calculate per-bucket edges (used for dashboard display)."""
    stdev = FORECAST_STDEV_24H if days_ahead <= 1 else FORECAST_STDEV_48H
    mean = forecast.high_temp_f
    temp_floor = None
    if current_observed_high is not None and days_ahead <= 1:
        temp_floor = current_observed_high
        hourly_peak = None
        if hourly_forecast:
            hourly_temps = [h["temp_f"] for h in hourly_forecast]
            if hourly_temps:
                hourly_peak = max(hourly_temps)
        if hourly_peak and hourly_peak > current_observed_high:
            effective_mean = min(hourly_peak, mean)
            effective_mean = max(effective_mean, current_observed_high)
            mean = effective_mean
            stdev = min(stdev, 2.5)
        elif current_observed_high >= mean:
            mean = current_observed_high + 1
            stdev = min(stdev, 2.0)
    opportunities = []

    for bucket in buckets:
        our_prob = _bucket_probability(mean, stdev, bucket.low_bound, bucket.high_bound, temp_floor)
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
    mean: float, stdev: float, low: float | None, high: float | None,
    temp_floor: float | None = None,
) -> float:
    """Probability that temp falls in [low, high] given N(mean, stdev).
    
    If temp_floor is set, the distribution is truncated: the daily high
    cannot be below temp_floor (it's already been observed). Probabilities
    for buckets entirely below the floor are 0, and remaining probabilities
    are renormalized.
    """
    # If bucket is entirely below the floor, probability is 0
    if temp_floor is not None and high is not None and high < temp_floor:
        return 0.0
    
    # Raw probability from normal distribution
    if low is None and high is not None:
        raw = _norm_cdf(high + 0.5, mean, stdev)
    elif low is not None and high is not None:
        # If floor cuts into this bucket, adjust the lower bound
        effective_low = max(low, temp_floor) if temp_floor is not None else low
        raw = _norm_cdf(high + 0.5, mean, stdev) - _norm_cdf(effective_low - 0.5, mean, stdev)
        raw = max(0.0, raw)
    elif low is not None and high is None:
        raw = 1.0 - _norm_cdf(low - 0.5, mean, stdev)
    else:
        return 0.0
    
    # If we have a floor, renormalize: P(bucket | temp >= floor)
    if temp_floor is not None:
        p_above_floor = 1.0 - _norm_cdf(temp_floor - 0.5, mean, stdev)
        if p_above_floor > 0.01:  # Avoid division by near-zero
            raw = raw / p_above_floor
    
    return min(raw, 1.0)


def _norm_cdf(x: float, mean: float, stdev: float) -> float:
    if stdev <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (stdev * math.sqrt(2))))
