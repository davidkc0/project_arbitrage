"""Intraday Temperature Predictor — predicts today's daily high from current observations.

Uses historical METAR data to build statistical models of temperature progression
throughout the day, grouped by conditions (sky cover, wind, daylight hours, month).

The edge: by mid-afternoon, we can predict the daily high with much higher certainty
than the market prices, because we see the actual heating curve in progress.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

from weather_bets.iem_fetcher import (
    DAILY_SUMMARY_FILE,
    HISTORY_FILE,
    compute_daylight_hours,
    load_daily_summary,
)

logger = logging.getLogger(__name__)


# ── Condition grouping ─────────────────────────────────────────────────

def _daylight_bucket(hours: float) -> str:
    """Categorize daylight hours into buckets."""
    if hours < 11.0:
        return "short"    # ~Nov-Jan
    elif hours < 13.0:
        return "medium"   # ~Feb-Apr, Sep-Oct
    else:
        return "long"     # ~May-Aug


def _sky_bucket(sky: str) -> str:
    """Simplify sky cover into 3 categories."""
    if sky in ("CLR", "FEW"):
        return "clear"
    elif sky == "SCT":
        return "partly"
    else:  # BKN, OVC
        return "cloudy"


def _wind_bucket(wind_kt: float) -> str:
    """Categorize wind speed."""
    if wind_kt < 5:
        return "calm"
    elif wind_kt <= 15:
        return "light"
    else:
        return "windy"


def _make_group_key(sky: str, wind_kt: float, daylight: float, month: int) -> str:
    """Build a condition group key for lookup."""
    return f"{_sky_bucket(sky)}|{_daylight_bucket(daylight)}|{_wind_bucket(wind_kt)}"


# ── Model building ─────────────────────────────────────────────────────

class IntradayModel:
    """
    Statistical model: given current temp at hour H under conditions C,
    how much more will the temperature rise today?

    Built from historical hourly data. For each (condition_group, hour),
    stores the distribution of (daily_high - temp_at_hour).
    """

    def __init__(self):
        # group_key -> hour -> list of remaining_rise values
        self._rise_data: dict[str, dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # group_key -> list of peak_hour values
        self._peak_hours: dict[str, list[int]] = defaultdict(list)
        # Fallback: all data regardless of conditions
        self._global_rise: dict[int, list[int]] = defaultdict(list)
        self._global_peaks: list[int] = []
        self._n_days = 0

    def build(self, hourly_path: Path = HISTORY_FILE, daily_path: Path = DAILY_SUMMARY_FILE) -> None:
        """Build the model from cached CSV data."""
        daily_data = load_daily_summary(daily_path)
        if not daily_data:
            logger.warning("[Predictor] No daily summary data — cannot build model")
            return

        # Index daily summaries by date
        daily_by_date = {d["date"]: d for d in daily_data}

        # Load hourly data
        hourly_by_date: dict[str, dict[int, dict]] = defaultdict(dict)
        with open(hourly_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row["date"]
                h = int(row["hour"])
                hourly_by_date[d][h] = {
                    "temp_f": int(float(row["temp_f"])),
                    "wind_kt": float(row["wind_kt"]),
                    "sky_cover": row["sky_cover"],
                    "daylight_hours": float(row["daylight_hours"]),
                }

        # For each day, compute remaining rise at each hour
        for date_str, day_info in daily_by_date.items():
            hours = hourly_by_date.get(date_str, {})
            if not hours:
                continue

            daily_high = day_info["daily_high"]
            peak_hour = day_info["peak_hour"]
            sky = day_info["sky_dominant"]
            wind = day_info["wind_avg_kt"]
            daylight = day_info["daylight_hours"]
            month = day_info["month"]

            group_key = _make_group_key(sky, wind, daylight, month)
            self._peak_hours[group_key].append(peak_hour)
            self._global_peaks.append(peak_hour)
            self._n_days += 1

            # For each hour reading, compute how much more it will rise
            for hour, reading in hours.items():
                if hour < 8 or hour > 19:
                    continue  # Only daytime hours matter
                remaining_rise = daily_high - reading["temp_f"]
                self._rise_data[group_key][hour].append(remaining_rise)
                self._global_rise[hour].append(remaining_rise)

        n_groups = len(self._rise_data)
        logger.info(
            f"[Predictor] Model built: {self._n_days} days, "
            f"{n_groups} condition groups"
        )

        # Log group sizes for debugging
        for key, hours in sorted(self._rise_data.items()):
            sample = len(hours.get(12, []))
            logger.debug(f"  {key}: ~{sample} days with noon data")

    def predict(
        self,
        current_temp: int,
        current_hour: int,
        sky_cover: str,
        wind_kt: float,
        daylight_hours: float,
        month: int,
    ) -> dict:
        """
        Predict today's daily high given current conditions.

        Returns:
        - predicted_high: best estimate (mean)
        - confidence_80: (low, high) 80% confidence interval
        - confidence_90: (low, high) 90% confidence interval
        - std_dev: standard deviation of prediction
        - sample_size: how many historical days match this condition
        - peak_hour_estimate: when the high will likely occur
        - confidence_score: 0-100 (higher = more certain)
        """
        group_key = _make_group_key(sky_cover, wind_kt, daylight_hours, month)

        # Look up remaining rise for this group + hour
        rises = list(self._rise_data.get(group_key, {}).get(current_hour, []))

        # Need at least 30 samples for reliable stats; otherwise widen search
        fallback_used = False
        if len(rises) < 30:
            # Try with just sky + daylight (ignore wind)
            partial_key = f"{_sky_bucket(sky_cover)}|{_daylight_bucket(daylight_hours)}"
            rises = []  # Reset — don't mix small group with fallback
            for key, hours in self._rise_data.items():
                if key.startswith(partial_key):
                    rises.extend(hours.get(current_hour, []))
            fallback_used = True

        if len(rises) < 15:
            # Ultimate fallback: global data for this hour
            rises = list(self._global_rise.get(current_hour, []))
            fallback_used = True

        if not rises:
            return {
                "predicted_high": current_temp,
                "confidence_80": (current_temp - 3, current_temp + 3),
                "confidence_90": (current_temp - 5, current_temp + 5),
                "std_dev": 5.0,
                "sample_size": 0,
                "peak_hour_estimate": 16,
                "confidence_score": 0,
                "group_key": group_key,
                "fallback": True,
            }

        # ── Outlier rejection using IQR capping ──
        # A single +21°F rise at 3 PM shouldn't blow up the CI
        rises_sorted = sorted(rises)
        n = len(rises_sorted)
        q1 = rises_sorted[int(n * 0.25)]
        q3 = rises_sorted[int(n * 0.75)]
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        # Cap outliers to the fence values instead of removing them
        rises = [max(lower_fence, min(upper_fence, r)) for r in rises]

        # Statistics on remaining rise (after outlier capping)
        rises_sorted = sorted(rises)
        n = len(rises_sorted)
        mean_rise = sum(rises) / n
        variance = sum((r - mean_rise) ** 2 for r in rises) / n
        std_dev = math.sqrt(variance) if variance > 0 else 0.5

        predicted_high = round(current_temp + mean_rise)

        # Percentile-based confidence intervals (more robust than assuming normal)
        p10 = rises_sorted[max(0, int(n * 0.10))]
        p25 = rises_sorted[max(0, int(n * 0.25))]
        p75 = rises_sorted[min(n - 1, int(n * 0.75))]
        p90 = rises_sorted[min(n - 1, int(n * 0.90))]

        ci_80 = (int(current_temp + p10), int(current_temp + p90))
        ci_90 = (int(current_temp + rises_sorted[max(0, int(n * 0.05))]),
                 int(current_temp + rises_sorted[min(n - 1, int(n * 0.95))]))

        # Peak hour estimate for this condition group
        peaks = self._peak_hours.get(group_key, self._global_peaks)
        if peaks:
            avg_peak = sum(peaks) / len(peaks)
        else:
            avg_peak = 16.0

        # Confidence score: tighter CI = higher confidence
        ci_width = ci_80[1] - ci_80[0]
        # Score from 0-100: width of 0 = 100, width of 10+ = 0
        confidence_score = max(0, min(100, round(100 * (1 - ci_width / 10))))

        # Boost confidence if we're past the expected peak
        if current_hour >= avg_peak:
            confidence_score = min(100, confidence_score + 15)

        return {
            "predicted_high": predicted_high,
            "confidence_80": ci_80,
            "confidence_90": ci_90,
            "std_dev": round(std_dev, 2),
            "sample_size": n,
            "peak_hour_estimate": round(avg_peak, 1),
            "confidence_score": confidence_score,
            "mean_rise": round(mean_rise, 1),
            "group_key": group_key,
            "fallback": fallback_used,
        }


# ── Backtesting ────────────────────────────────────────────────────────

def backtest(
    model: IntradayModel,
    test_dates: list[str] | None = None,
    hours_to_test: list[int] | None = None,
) -> dict:
    """
    Backtest the model against historical data.

    For each test date, simulates predictions at each hour and compares
    against the actual daily high.

    Returns aggregated accuracy stats by hour.
    """
    if hours_to_test is None:
        hours_to_test = [10, 11, 12, 13, 14, 15, 16, 17]

    daily_data = load_daily_summary()
    daily_by_date = {d["date"]: d for d in daily_data}

    # Load hourly data
    hourly_by_date: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(HISTORY_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row["date"]
            h = int(row["hour"])
            hourly_by_date[d][h] = {
                "temp_f": int(float(row["temp_f"])),
                "wind_kt": float(row["wind_kt"]),
                "sky_cover": row["sky_cover"],
                "daylight_hours": float(row["daylight_hours"]),
            }

    if test_dates is None:
        test_dates = sorted(daily_by_date.keys())

    # Results by hour
    results: dict[int, dict] = {h: {
        "errors": [],
        "abs_errors": [],
        "in_ci80": 0,
        "in_ci90": 0,
        "total": 0,
        "confidence_scores": [],
    } for h in hours_to_test}

    for date_str in test_dates:
        day_info = daily_by_date.get(date_str)
        hours = hourly_by_date.get(date_str, {})
        if not day_info or not hours:
            continue

        actual_high = day_info["daily_high"]

        for test_hour in hours_to_test:
            reading = hours.get(test_hour)
            if not reading:
                continue

            prediction = model.predict(
                current_temp=reading["temp_f"],
                current_hour=test_hour,
                sky_cover=reading["sky_cover"],
                wind_kt=reading["wind_kt"],
                daylight_hours=reading["daylight_hours"],
                month=date.fromisoformat(date_str).month,
            )

            error = prediction["predicted_high"] - actual_high
            abs_error = abs(error)

            r = results[test_hour]
            r["errors"].append(error)
            r["abs_errors"].append(abs_error)
            r["total"] += 1
            r["confidence_scores"].append(prediction["confidence_score"])

            if prediction["confidence_80"][0] <= actual_high <= prediction["confidence_80"][1]:
                r["in_ci80"] += 1
            if prediction["confidence_90"][0] <= actual_high <= prediction["confidence_90"][1]:
                r["in_ci90"] += 1

    # Compute summary stats
    summary = {}
    for hour in hours_to_test:
        r = results[hour]
        if r["total"] == 0:
            continue
        n = r["total"]
        mae = sum(r["abs_errors"]) / n
        bias = sum(r["errors"]) / n
        within_1 = sum(1 for e in r["abs_errors"] if e <= 1) / n * 100
        within_2 = sum(1 for e in r["abs_errors"] if e <= 2) / n * 100
        ci80_hit = r["in_ci80"] / n * 100
        ci90_hit = r["in_ci90"] / n * 100
        avg_conf = sum(r["confidence_scores"]) / n

        summary[hour] = {
            "n": n,
            "mae": round(mae, 2),
            "bias": round(bias, 2),
            "within_1f": round(within_1, 1),
            "within_2f": round(within_2, 1),
            "ci80_hit_rate": round(ci80_hit, 1),
            "ci90_hit_rate": round(ci90_hit, 1),
            "avg_confidence": round(avg_conf, 1),
        }

    return summary
