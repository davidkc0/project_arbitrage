"""Triple-strategy bet decision engine for Kalshi temperature markets.

Play 1 (YES): Predict which bucket the daily high lands in → buy YES
Play 2 (NO):  Identify dead/overpriced buckets → buy NO for base income
Play 3 (FAV): Buy YES on market favorite when priced 50-65¢ (historical sweet spot)

Uses:
- intraday_predictor for temperature trajectory prediction
- rounding_map for NWS conversion edge detection
- kalshi_weather for live bucket pricing
- executor for order placement
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from weather_bets.rounding_map import (
    get_kalshi_bucket,
    get_bucket_label,
    is_crossing_temp,
    should_skip_bucket,
)

logger = logging.getLogger(__name__)

# ── Historical remaining-rise patterns (from 10yr KAUS data, clear days) ──
# month -> hour -> (avg_rise, p10_rise, p90_rise, n_samples)
CLEAR_DAY_PATTERNS = {
    1:  {10: (9.3, 6, 14, 186), 11: (5.9, 3, 9, 185), 12: (3.5, 1, 6, 186),
         13: (1.8, 0, 3, 186),  14: (0.9, 0, 2, 186),  15: (1.0, 0, 2, 185)},
    2:  {10: (9.4, 5, 13, 149), 11: (6.2, 3, 9, 149), 12: (3.8, 1, 6, 149),
         13: (2.1, 0, 4, 149),  14: (1.1, 0, 2, 149),  15: (0.7, 0, 1, 149)},
    3:  {10: (10.9, 6, 16, 167), 11: (7.8, 4, 12, 167), 12: (5.1, 2, 8, 167),
         13: (2.9, 1, 5, 167),   14: (1.4, 0, 3, 167),  15: (0.6, 0, 2, 167)},
    4:  {10: (9.9, 6, 14, 146), 11: (7.2, 4, 10, 146), 12: (4.9, 2, 8, 146),
         13: (3.0, 1, 6, 146),   14: (1.6, 0, 3, 146),  15: (0.9, 0, 2, 146)},
    5:  {10: (8.6, 4, 12, 147), 11: (6.1, 3, 9, 146),  12: (4.2, 1, 7, 147),
         13: (2.7, 1, 5, 147),   14: (1.5, 0, 3, 147),  15: (0.9, 0, 2, 147)},
    6:  {10: (9.2, 6, 13, 148), 11: (6.8, 4, 10, 148), 12: (4.8, 3, 7, 147),
         13: (3.0, 1, 5, 148),   14: (1.7, 0, 3, 148),  15: (1.0, 0, 2, 147)},
    7:  {10: (9.7, 6, 13, 186), 11: (7.0, 4, 10, 186), 12: (4.8, 2, 7, 186),
         13: (3.0, 1, 5, 186),   14: (1.7, 0, 3, 187),  15: (1.1, 0, 2, 186)},
    8:  {10: (9.8, 6, 13, 180), 11: (7.0, 4, 10, 180), 12: (4.9, 3, 7, 180),
         13: (3.3, 1, 5, 178),   14: (2.2, 0, 4, 180),  15: (1.5, 0, 2, 180)},
    9:  {10: (8.9, 6, 12, 160), 11: (6.2, 4, 9, 160),  12: (4.1, 2, 6, 160),
         13: (2.5, 1, 4, 160),   14: (1.3, 0, 3, 160),  15: (0.8, 0, 2, 160)},
    10: {10: (10.5, 7, 14, 191), 11: (7.0, 4, 10, 191), 12: (4.5, 2, 7, 191),
         13: (2.7, 1, 5, 191),   14: (1.5, 0, 3, 191),  15: (0.7, 0, 2, 191)},
    11: {10: (8.2, 5, 12, 170), 11: (5.2, 3, 8, 170),  12: (3.0, 1, 5, 170),
         13: (1.5, 0, 3, 170),   14: (0.6, 0, 1, 170),  15: (0.9, 0, 2, 170)},
    12: {10: (9.2, 5, 14, 160), 11: (5.8, 3, 9, 160),  12: (3.4, 1, 6, 160),
         13: (1.5, 0, 3, 160),   14: (0.7, 0, 2, 160),  15: (0.9, 0, 2, 160)},
}

# Month → earliest hour where p90-p10 range ≤ 4°F (2 Kalshi buckets)
EARLIEST_BET_HOUR = {
    1: 13, 2: 12, 3: 14, 4: 13, 5: 13, 6: 12,
    7: 12, 8: 11, 9: 10, 10: 13, 11: 12, 12: 13,
}


def predict_high_combined(
    current_temp: float,
    day_max: float | None,
    current_hour: int,
    month: int,
    consensus_high: float | None = None,
    nws_hourly: list[dict] | None = None,
) -> dict:
    """Combined prediction using forecasts + patterns + observed data.

    Priority:
      1. NWS hourly peak (tells us WHEN the high occurs and how much is left)
      2. Consensus forecast (NWS + GFS + ICON + GEM weighted average)
      3. Current observed max (floor — high can't go below this)
      4. Historical pattern (confidence bands only)

    Returns dict with 'predicted_high', 'pred_low', 'pred_high',
    'peak_hour', 'source', 'confidence'.
    """
    # Floor: day max or current temp
    floor = max(day_max or current_temp, current_temp)

    # ── NWS Hourly: find peak hour and peak temp ──
    nws_peak_temp = None
    nws_peak_hour = None
    if nws_hourly:
        def _parse_hour(h):
            raw = h.get("hour", 0)
            if isinstance(raw, int):
                return raw
            return int(str(raw).split(":")[0])

        for h in nws_hourly:
            t = h.get("temp_f", 0)
            hr = _parse_hour(h)
            if nws_peak_temp is None or t > nws_peak_temp:
                nws_peak_temp = t
                nws_peak_hour = hr

    # ── Historical pattern for confidence bands ──
    pattern = CLEAR_DAY_PATTERNS.get(month, {}).get(current_hour)
    if pattern:
        avg_rise, p10_rise, p90_rise, n_samples = pattern
    else:
        avg_rise, p10_rise, p90_rise, n_samples = 0, 0, 0, 0

    # ── Determine primary prediction ──
    sources = []

    # Source 1: NWS hourly peak
    if nws_peak_temp is not None:
        sources.append(("nws_hourly", nws_peak_temp))

    # Source 2: Consensus forecast
    if consensus_high is not None:
        sources.append(("consensus", consensus_high))

    # Source 3: Pattern-based
    pattern_pred = floor + avg_rise
    sources.append(("pattern", pattern_pred))

    # Primary = highest of NWS hourly peak and consensus (they see the future)
    # But never below the observed floor
    if nws_peak_temp is not None and consensus_high is not None:
        predicted = max(nws_peak_temp, consensus_high, floor)
        source = "nws_hourly+consensus"
    elif nws_peak_temp is not None:
        predicted = max(nws_peak_temp, floor)
        source = "nws_hourly"
    elif consensus_high is not None:
        predicted = max(consensus_high, floor)
        source = "consensus"
    else:
        predicted = max(pattern_pred, floor)
        source = "pattern_only"

    # ── Confidence bands ──
    # Use pattern P10/P90 as offset from FLOOR (not from prediction)
    # But ensure prediction is within bounds
    pred_low = max(floor + p10_rise, floor)
    pred_high = floor + p90_rise

    # If forecast says higher than pattern P90, widen the high bound
    if predicted > pred_high:
        pred_high = predicted + 1

    # Ensure prediction is >= floor
    predicted = max(predicted, floor)

    # Confidence: tighter when NWS hourly and consensus agree
    confidence = "low"
    if nws_peak_temp is not None and consensus_high is not None:
        spread = abs(nws_peak_temp - consensus_high)
        if spread <= 2:
            confidence = "high"
        elif spread <= 4:
            confidence = "medium"
    elif consensus_high is not None:
        confidence = "medium"

    return {
        "predicted_high": round(predicted),
        "pred_low": round(pred_low),
        "pred_high": round(pred_high),
        "peak_hour": nws_peak_hour,
        "source": source,
        "confidence": confidence,
        "pattern_pred": round(pattern_pred),
        "nws_peak": nws_peak_temp,
        "consensus": consensus_high,
        "floor": floor,
    }


class BetEngine:
    """Triple-strategy bet decision engine.

    Play 1 (YES): When CI is tight enough, predict the bucket and buy YES.
    Play 2 (NO):  Buy NO on dead/overpriced buckets for daily base income.
    Play 3 (FAV): Buy YES on market favorite priced 50-65¢ (historical sweet spot).
    """

    def __init__(
        self,
        total_balance: float = 100.0,
        yes_pool_pct: float = 0.60,
        no_pool_pct: float = 0.40,
        execution_mode: str = "dry",
    ):
        self.total_balance = total_balance
        self.yes_pool_pct = yes_pool_pct
        self.no_pool_pct = no_pool_pct
        self.execution_mode = execution_mode  # "dry" or "live"

        # Daily state
        self.yes_bet_placed_today = False
        self.no_bets_placed_today = False
        self.favorite_bet_placed_today = False
        self.today_date: str = ""

        # Session log
        self.decisions: list[dict] = []

    @property
    def yes_pool(self) -> float:
        return self.total_balance * self.yes_pool_pct

    @property
    def no_pool(self) -> float:
        return self.total_balance * self.no_pool_pct

    def reset_daily(self, today: str):
        """Reset daily state for a new trading day."""
        self.today_date = today
        self.yes_bet_placed_today = False
        self.no_bets_placed_today = False
        self.favorite_bet_placed_today = False

    # ── Play 1: YES prediction ──────────────────────────────────────

    def evaluate_yes_play(
        self,
        current_temp: int,
        current_hour: int,
        month: int,
        sky_cover: str,
        buckets: list[dict],  # Kalshi bucket data with prices
        day_max: float | None = None,
        consensus_high: float | None = None,
        nws_hourly: list[dict] | None = None,
    ) -> dict:
        """Evaluate whether to place a YES bet on the predicted bucket.

        Uses combined prediction: consensus + NWS hourly + pattern + observed.

        Args:
            current_temp: Current temperature in °F
            current_hour: Current hour (10-16)
            month: Current month (1-12)
            sky_cover: NWS sky cover code (CLR, FEW, SCT, BKN, OVC)
            buckets: List of Kalshi bucket dicts with 'low_bound', 'yes_price', 'ticker'
            day_max: Highest temperature observed today so far
            consensus_high: Consensus forecast (NWS + models weighted avg)
            nws_hourly: NWS hourly forecast [{"hour": 15, "temp_f": 86}, ...]

        Returns:
            Decision dict with 'action', 'reasoning', and bet details if action='bet'
        """
        decision = {
            "play": "YES",
            "action": "skip",
            "reasoning": "",
            "ticker": None,
            "side": "yes",
            "bucket": None,
            "price": 0,
            "contracts": 0,
            "cost": 0,
        }

        # Rule 1: Only bet clear days
        if sky_cover not in ("CLR", "FEW"):
            decision["reasoning"] = f"Skip: sky={sky_cover}, need CLR/FEW"
            return decision

        # Rule 2: Check if we're past the earliest bet hour for this month
        earliest = EARLIEST_BET_HOUR.get(month, 14)
        if current_hour < earliest:
            decision["reasoning"] = (
                f"Skip: {current_hour}:00 < earliest bet hour {earliest}:00 for month {month}"
            )
            return decision

        # Rule 3: Already bet today
        if self.yes_bet_placed_today:
            decision["reasoning"] = "Skip: YES bet already placed today"
            return decision

        # Rule 4: Combined prediction (forecast + pattern + observed)
        prediction = predict_high_combined(
            current_temp=current_temp,
            day_max=day_max,
            current_hour=current_hour,
            month=month,
            consensus_high=consensus_high,
            nws_hourly=nws_hourly,
        )

        pred_low = prediction["pred_low"]
        pred_high = prediction["pred_high"]
        pred_mid = prediction["predicted_high"]

        # How many Kalshi buckets does this span?
        pred_buckets = set()
        for t in range(pred_low, pred_high + 1):
            pred_buckets.add(get_kalshi_bucket(t))
        n_buckets = len(pred_buckets)

        if n_buckets > 2:
            decision["reasoning"] = (
                f"Skip: CI too wide — {pred_low}-{pred_high}°F spans {n_buckets} buckets"
            )
            return decision

        # Rule 5: Check rounding crossing
        if should_skip_bucket(pred_mid) and n_buckets > 1:
            decision["reasoning"] = (
                f"Skip: predicted {pred_mid}°F is a rounding crossing temp "
                f"and CI spans {n_buckets} buckets"
            )
            return decision

        # Find the target bucket(s) on Kalshi
        target_bucket_keys = pred_buckets
        target_kalshi = []
        for bkt in buckets:
            bkt_key = (int(bkt.get("low_bound", 0)), int(bkt.get("high_bound", 0)))
            if bkt_key in target_bucket_keys:
                target_kalshi.append(bkt)

        if not target_kalshi:
            decision["reasoning"] = f"Skip: target bucket(s) not found on Kalshi"
            return decision

        # Pick the best bucket (cheapest, highest EV)
        best = min(target_kalshi, key=lambda b: b.get("yes_price", 1.0))
        yes_price = best.get("yes_price", 1.0)

        # Rule 6: Price check (positive EV)
        if yes_price > 0.70:
            decision["reasoning"] = (
                f"Skip: YES price {yes_price:.2f} > 0.70 for "
                f"{best.get('label', '???')}"
            )
            return decision

        # Calculate sizing (Half-Kelly)
        bet_pct = 0.30 if n_buckets == 1 else 0.20
        bet_amount = self.yes_pool * bet_pct
        contracts = int(bet_amount / yes_price)

        if contracts < 1:
            decision["reasoning"] = f"Skip: insufficient funds for 1 contract"
            return decision

        cost = contracts * yes_price

        decision.update({
            "action": "bet",
            "reasoning": (
                f"YES on {best.get('label', '???')} — "
                f"predicted {pred_low}-{pred_high}°F ({n_buckets} bucket(s)), "
                f"price={yes_price:.2f}, {contracts} contracts"
            ),
            "ticker": best.get("ticker"),
            "bucket": best.get("label"),
            "price": yes_price,
            "contracts": contracts,
            "cost": round(cost, 2),
            "predicted_range": f"{pred_low}-{pred_high}",
            "n_buckets": n_buckets,
        })

        return decision

    # ── Play 2: NO base income ──────────────────────────────────────

    def evaluate_no_plays(
        self,
        current_temp: int,
        current_hour: int,
        month: int,
        sky_cover: str,
        buckets: list[dict],
    ) -> list[dict]:
        """Evaluate NO bets on dead and overpriced buckets.

        Returns a list of NO bet decisions.
        """
        decisions = []

        if self.no_bets_placed_today:
            return decisions

        if current_hour < 12:
            return decisions  # Wait until noon for enough dead buckets

        # ── Dead buckets: temp already surpassed ──
        dead_bets = []
        for bkt in buckets:
            low_bound = bkt.get("low_bound")
            high_bound = bkt.get("high_bound")
            if low_bound is None or high_bound is None:
                continue

            # Dead = current temp is already above the top of this bucket
            if current_temp > high_bound + 1:
                yes_price = bkt.get("yes_price", 0)
                no_price = bkt.get("no_price", 1.0)

                # Only worth it if YES ≥ 3¢ (otherwise profit too thin)
                if yes_price >= 0.03:
                    dead_bets.append({
                        "play": "NO_DEAD",
                        "action": "bet",
                        "side": "no",
                        "ticker": bkt.get("ticker"),
                        "bucket": bkt.get("label"),
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "profit_per_contract": round(yes_price, 2),
                        "reasoning": (
                            f"Dead bucket: {bkt.get('label', '???')} "
                            f"(current temp {current_temp}°F > {high_bound}°F), "
                            f"NO costs {no_price:.2f}, profit {yes_price:.2f}/contract"
                        ),
                    })

        # Size dead bucket bets: 30% of NO pool, split evenly
        if dead_bets:
            dead_budget = self.no_pool * 0.30
            per_bucket = dead_budget / len(dead_bets) if dead_bets else 0

            for bet in dead_bets:
                n = int(per_bucket / bet["no_price"])
                if n < 1:
                    bet["action"] = "skip"
                    bet["reasoning"] += " — insufficient for 1 contract"
                    continue
                bet["contracts"] = n
                bet["cost"] = round(n * bet["no_price"], 2)

            decisions.extend(dead_bets)

        # ── Overpriced far buckets: >6°F below predicted high ──
        if sky_cover in ("CLR", "FEW"):
            pattern = CLEAR_DAY_PATTERNS.get(month, {}).get(current_hour)
            if pattern:
                avg_rise = pattern[0]
                predicted_high = current_temp + avg_rise

                far_bets = []
                for bkt in buckets:
                    low_bound = bkt.get("low_bound")
                    high_bound = bkt.get("high_bound")
                    if low_bound is None or high_bound is None:
                        continue

                    bucket_mid = (low_bound + high_bound) / 2
                    distance_below = predicted_high - bucket_mid

                    # Must be >6°F below predicted high
                    if distance_below < 6:
                        continue

                    # Must not already be a dead bucket
                    if current_temp > high_bound + 1:
                        continue

                    yes_price = bkt.get("yes_price", 0)
                    if yes_price >= 0.05:  # Only if YES is 5¢+
                        far_bets.append({
                            "play": "NO_FAR",
                            "action": "bet",
                            "side": "no",
                            "ticker": bkt.get("ticker"),
                            "bucket": bkt.get("label"),
                            "yes_price": yes_price,
                            "no_price": bkt.get("no_price", 1.0),
                            "profit_per_contract": round(yes_price, 2),
                            "reasoning": (
                                f"Overpriced far: {bkt.get('label', '???')} is "
                                f"{distance_below:.0f}°F below prediction, "
                                f"YES={yes_price:.2f} is too high"
                            ),
                        })

                # Size far bucket bets: 20% of NO pool
                if far_bets:
                    far_budget = self.no_pool * 0.20
                    per_bucket = far_budget / len(far_bets)

                    for bet in far_bets:
                        n = int(per_bucket / bet["no_price"])
                        if n < 1:
                            bet["action"] = "skip"
                            continue
                        bet["contracts"] = n
                        bet["cost"] = round(n * bet["no_price"], 2)

                    decisions.extend(far_bets)

        return decisions

    # ── Play 3: Market Favorite (Sweet Spot, Month-Aware) ──────────

    # Month → confidence tier based on remaining-rise std dev after noon
    # A = tight (std ≤ 2.0°F), B = moderate (≤ 2.8°F), C = wide (> 2.8°F)
    MONTH_TIER = {
        1: "B",  2: "C",  3: "C",  4: "B",  5: "C",  6: "B",
        7: "A",  8: "A",  9: "A", 10: "A", 11: "B", 12: "C",
    }

    # Tier → earliest evaluation hour and position sizing
    TIER_CONFIG = {
        "A": {"earliest_hour": 10, "size_pct": 0.30, "max_distance": 3},
        "B": {"earliest_hour": 11, "size_pct": 0.25, "max_distance": 4},
        "C": {"earliest_hour": 12, "size_pct": 0.20, "max_distance": 5},
    }

    def evaluate_favorite_play(
        self,
        current_temp: int,
        current_hour: int,
        month: int,
        sky_cover: str,
        buckets: list[dict],
        day_max: int | None = None,
        consensus_high: float | None = None,
        nws_hourly: list[dict] | None = None,
    ) -> dict:
        """Buy YES on the market favorite when priced in the 45-68¢ sweet spot.

        Uses CLEAR_DAY_PATTERNS for hour-specific remaining rise prediction.
        High-confidence months (Jul-Oct) start at 10 AM; low-confidence (Mar, May,
        Dec) wait until noon.

        Based on analysis of 6,249 settled KXHIGHAUS markets:
        - At $0.50 YES, buckets win 57% (+7% edge, +15% ROI)
        - At $0.65 YES, buckets win 78% (+13% edge, +20% ROI)
        """
        decision = {
            "play": "FAV",
            "action": "skip",
            "reasoning": "",
            "ticker": None,
            "side": "yes",
            "bucket": None,
            "price": 0,
            "contracts": 0,
            "cost": 0,
        }

        if self.favorite_bet_placed_today:
            decision["reasoning"] = "Skip: FAV bet already placed today"
            return decision

        # Get month confidence tier
        tier = self.MONTH_TIER.get(month, "C")
        tier_cfg = self.TIER_CONFIG[tier]
        earliest = tier_cfg["earliest_hour"]
        size_pct = tier_cfg["size_pct"]
        max_dist = tier_cfg["max_distance"]

        if current_hour < earliest:
            decision["reasoning"] = (
                f"Skip: {current_hour}:00 < {earliest}:00 "
                f"(tier {tier} for month {month})"
            )
            return decision

        # ── Predict daily high using combined forecast + pattern ──
        prediction = predict_high_combined(
            current_temp=current_temp,
            day_max=day_max,
            current_hour=current_hour,
            month=month,
            consensus_high=consensus_high,
            nws_hourly=nws_hourly,
        )

        predicted_high = prediction["predicted_high"]
        pred_low = prediction["pred_low"]
        pred_high_bound = prediction["pred_high"]

        # Prediction confidence: how many buckets does the range span?
        pred_range = pred_high_bound - pred_low

        # Find buckets in the sweet spot price range
        sweet_spot_min = 0.60
        sweet_spot_max = 0.68
        candidates = []

        for bkt in buckets:
            yes_price = bkt.get("yes_price", 0)
            if sweet_spot_min <= yes_price <= sweet_spot_max:
                candidates.append(bkt)

        if not candidates:
            prices_str = ", ".join(
                f"{b.get('label','?')}={b.get('yes_price',0):.2f}" for b in buckets
            )
            decision["reasoning"] = (
                f"Skip: no buckets in sweet spot "
                f"(${sweet_spot_min:.2f}-${sweet_spot_max:.2f}). "
                f"Predicted high: {predicted_high}°F (range {pred_low}-{pred_high_bound}°F, "
                f"tier {tier}). Prices: [{prices_str}]"
            )
            return decision

        # Score candidates: does our prediction agree with this bucket?
        best = None
        best_score = -999

        for bkt in candidates:
            low_bound = bkt.get("low_bound", 0)
            high_bound = bkt.get("high_bound", 999)
            yes_price = bkt.get("yes_price", 0)
            bucket_mid = (low_bound + high_bound) / 2

            # Does our trajectory predict this bucket?
            pred_score = 0
            if low_bound <= predicted_high <= high_bound:
                pred_score = 8  # Strong: predicted high falls in this bucket
            elif pred_low <= high_bound and pred_high_bound >= low_bound:
                pred_score = 4  # Prediction range overlaps this bucket
            elif abs(predicted_high - bucket_mid) <= max_dist:
                pred_score = 1  # Within max allowed distance
            else:
                pred_score = -5  # Disagree

            # Price attractiveness: prefer lower price within sweet spot
            price_score = (sweet_spot_max - yes_price) * 5

            total_score = pred_score + price_score

            if total_score > best_score:
                best_score = total_score
                best = bkt

        if best is None or best_score < 0:
            decision["reasoning"] = (
                f"Skip: no candidate agrees with prediction "
                f"({predicted_high}°F, range {pred_low}-{pred_high_bound}°F, tier {tier})"
            )
            return decision

        yes_price = best.get("yes_price", 0)
        low_bound = best.get("low_bound", 0)
        high_bound = best.get("high_bound", 999)
        bucket_mid = (low_bound + high_bound) / 2
        distance = abs(predicted_high - bucket_mid)

        # Final distance check
        if distance > max_dist:
            decision["reasoning"] = (
                f"Skip: FAV {best.get('label','?')} at ${yes_price:.2f} is "
                f"{distance:.0f}°F from prediction ({predicted_high}°F), "
                f"max allowed: {max_dist}°F for tier {tier}"
            )
            return decision

        # Size: tier-dependent %, boost if prediction range is narrow (≤ 4°F)
        effective_pct = size_pct
        if pred_range <= 4:
            effective_pct = min(size_pct * 1.25, 0.35)  # 25% boost, cap at 35%

        bet_amount = self.yes_pool * effective_pct
        contracts = int(bet_amount / yes_price)

        if contracts < 1:
            decision["reasoning"] = "Skip: insufficient funds for 1 FAV contract"
            return decision

        cost = contracts * yes_price
        narrow_str = " 🎯 NARROW" if pred_range <= 4 else ""

        decision.update({
            "action": "bet",
            "reasoning": (
                f"FAV: {best.get('label', '???')} at ${yes_price:.2f} "
                f"| pred={predicted_high}°F (range {pred_low}-{pred_high_bound}°F)"
                f"{narrow_str} "
                f"| tier {tier}, {contracts}x, ${cost:.2f}"
            ),
            "ticker": best.get("ticker"),
            "bucket": best.get("label"),
            "price": yes_price,
            "contracts": contracts,
            "cost": round(cost, 2),
            "predicted_high": predicted_high,
            "pred_range": f"{pred_low}-{pred_high_bound}",
            "confidence_tier": tier,
            "sweet_spot_edge": "+7-13% historical",
        })

        return decision

    # ── Logging ───────────────────────────────────────────────────

    def log_decision(self, decision: dict):
        """Log a bet decision to the session log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "date": self.today_date,
            **decision,
        }
        self.decisions.append(entry)

        action = decision.get("action", "skip")
        play = decision.get("play", "?")
        reasoning = decision.get("reasoning", "no reason")

        if action == "bet":
            logger.info(
                f"[BetEngine] ✅ {play} BET: {reasoning} "
                f"({decision.get('contracts', 0)} contracts, "
                f"${decision.get('cost', 0):.2f})"
            )
        else:
            logger.debug(f"[BetEngine] ⏭ {play} SKIP: {reasoning}")

    def save_session_log(self, path: Path | None = None):
        """Save the session's decisions to a JSON file."""
        if path is None:
            path = Path(__file__).parent / "data" / "bet_engine_log.json"

        # Load existing log if present
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, Exception):
                existing = []

        existing.extend(self.decisions)
        path.write_text(json.dumps(existing, indent=2, default=str))
        logger.info(f"[BetEngine] Session log saved: {len(self.decisions)} decisions")

    def get_daily_summary(self) -> dict:
        """Get a summary of today's activity."""
        today_decisions = [d for d in self.decisions if d.get("date") == self.today_date]
        bets = [d for d in today_decisions if d.get("action") == "bet"]
        skips = [d for d in today_decisions if d.get("action") == "skip"]

        yes_bets = [d for d in bets if d.get("play") == "YES"]
        no_bets = [d for d in bets if d.get("play", "").startswith("NO")]
        fav_bets = [d for d in bets if d.get("play") == "FAV"]

        total_cost = sum(d.get("cost", 0) for d in bets)
        total_contracts = sum(d.get("contracts", 0) for d in bets)

        return {
            "date": self.today_date,
            "total_bets": len(bets),
            "total_skips": len(skips),
            "yes_bets": len(yes_bets),
            "no_bets": len(no_bets),
            "fav_bets": len(fav_bets),
            "total_cost": round(total_cost, 2),
            "total_contracts": total_contracts,
            "balance_remaining": round(self.total_balance - total_cost, 2),
        }
