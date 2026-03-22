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

# ── Historical remaining-rise patterns (from 5yr KAUS data, clear days) ──
# month -> hour -> (avg_rise, p10_rise, p90_rise, n_samples)
CLEAR_DAY_PATTERNS = {
    1:  {10: (9.5, 5, 14, 82), 11: (6.1, 3, 10, 82), 12: (3.8, 2, 7, 82),
         13: (2.1, 0, 4, 82),  14: (0.9, 0, 2, 82),  15: (1.0, 0, 2, 81)},
    2:  {10: (10.4, 7, 15, 70), 11: (6.8, 4, 10, 70), 12: (4.1, 2, 6, 70),
         13: (2.2, 1, 4, 70),   14: (1.0, 0, 2, 70),  15: (0.6, 0, 1, 70)},
    3:  {10: (11.5, 6, 16, 75), 11: (8.2, 4, 12, 75), 12: (5.5, 2, 8, 75),
         13: (3.2, 1, 6, 75),   14: (1.7, 0, 3, 75),  15: (0.7, 0, 2, 75)},
    4:  {10: (11.1, 7, 15, 43), 11: (7.9, 4, 11, 43), 12: (5.2, 3, 8, 43),
         13: (2.9, 1, 5, 43),   14: (1.4, 0, 3, 43),  15: (0.6, 0, 2, 43)},
    5:  {10: (9.6, 6, 12, 34),  11: (7.1, 4, 9, 34),  12: (4.9, 2, 7, 34),
         13: (3.0, 1, 5, 34),   14: (1.4, 0, 3, 34),  15: (0.9, 0, 1, 34)},
    6:  {10: (9.5, 6, 13, 51),  11: (6.9, 4, 9, 51),  12: (4.8, 3, 7, 50),
         13: (2.8, 1, 5, 51),   14: (1.7, 0, 3, 51),  15: (0.8, 0, 2, 51)},
    7:  {10: (10.8, 8, 13, 51), 11: (7.7, 5, 10, 51), 12: (5.3, 3, 7, 51),
         13: (3.0, 1, 4, 51),   14: (1.4, 0, 3, 51),  15: (1.1, 0, 2, 51)},
    8:  {10: (10.6, 8, 13, 63), 11: (7.7, 6, 10, 63), 12: (5.3, 4, 7, 63),
         13: (3.3, 2, 5, 63),   14: (1.7, 0, 3, 63),  15: (0.8, 0, 2, 63)},
    9:  {10: (9.8, 8, 12, 72),  11: (6.8, 5, 9, 72),  12: (4.4, 3, 6, 72),
         13: (2.8, 1, 4, 72),   14: (1.2, 0, 2, 72),  15: (0.6, 0, 2, 72)},
    10: {10: (10.5, 8, 13, 79), 11: (6.9, 4, 9, 79),  12: (4.3, 2, 7, 79),
         13: (2.4, 1, 4, 79),   14: (1.0, 0, 2, 79),  15: (0.3, 0, 1, 79)},
    11: {10: (8.5, 5, 13, 67),  11: (5.4, 3, 8, 67),  12: (3.2, 1, 5, 67),
         13: (1.4, 0, 3, 67),   14: (0.4, 0, 1, 67),  15: (0.6, 0, 1, 67)},
    12: {10: (9.6, 5, 15, 59),  11: (6.1, 3, 10, 59), 12: (3.8, 1, 7, 59),
         13: (2.0, 0, 3, 59),   14: (1.1, 0, 2, 59),  15: (1.2, 0, 1, 59)},
}

# Month → earliest hour where p90-p10 range ≤ 4°F (2 Kalshi buckets)
EARLIEST_BET_HOUR = {
    1: 13, 2: 12, 3: 14, 4: 13, 5: 13, 6: 12,
    7: 12, 8: 11, 9: 10, 10: 13, 11: 12, 12: 13,
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
    ) -> dict:
        """Evaluate whether to place a YES bet on the predicted bucket.

        Args:
            current_temp: Current temperature in °F
            current_hour: Current hour (10-16)
            month: Current month (1-12)
            sky_cover: NWS sky cover code (CLR, FEW, SCT, BKN, OVC)
            buckets: List of Kalshi bucket dicts with 'low_bound', 'yes_price', 'ticker'

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

        # Rule 4: Get remaining rise pattern for this month/hour
        pattern = CLEAR_DAY_PATTERNS.get(month, {}).get(current_hour)
        if not pattern:
            decision["reasoning"] = f"Skip: no pattern data for month={month} hour={current_hour}"
            return decision

        avg_rise, p10_rise, p90_rise, n_samples = pattern

        # Predicted high range
        pred_low = current_temp + p10_rise
        pred_high = current_temp + p90_rise
        pred_mid = round(current_temp + avg_rise)

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

        # ── Predict daily high using hour-specific remaining rise ──
        pattern = CLEAR_DAY_PATTERNS.get(month, {}).get(current_hour)
        if not pattern:
            decision["reasoning"] = (
                f"Skip: no pattern data for month={month} hour={current_hour}"
            )
            return decision

        avg_rise, p10_rise, p90_rise, n_samples = pattern
        predicted_high = round(current_temp + avg_rise)
        pred_low = current_temp + p10_rise
        pred_high_bound = current_temp + p90_rise

        # Use day_max from Synoptic as a floor (high can't be below what we've seen)
        if day_max and day_max > predicted_high:
            predicted_high = day_max

        # Prediction confidence: how many buckets does the range span?
        pred_range = pred_high_bound - pred_low

        # Find buckets in the sweet spot price range
        sweet_spot_min = 0.45
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
