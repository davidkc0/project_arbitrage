"""City-aware forecast accuracy tracker.

Extends forecast_tracker.py with per-city tracking and dynamic model weighting.
Logs NWS + multi-model predictions vs actuals, and computes per-city weights
so cities with different forecast difficulty get appropriate model blending.

Data lives in: weather_bets/data/forecast_accuracy_by_city.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CITY_ACCURACY_FILE = DATA_DIR / "forecast_accuracy_by_city.json"

# Minimum settled data points before we use dynamic weights
MIN_SAMPLES = 5

# Default equal weights when not enough data
DEFAULT_WEIGHTS = {"nws": 0.25, "gfs": 0.25, "icon": 0.25, "gem": 0.25}


def _load() -> dict:
    """Load the city-keyed accuracy store."""
    if CITY_ACCURACY_FILE.exists():
        try:
            with open(CITY_ACCURACY_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[AccuracyTracker] Could not load {CITY_ACCURACY_FILE}: {e}")
    return {}


def _save(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CITY_ACCURACY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def log_forecast(
    city_code: str,
    target_date: str,
    days_out: int,
    nws: Optional[float],
    gfs: Optional[float],
    icon: Optional[float],
    gem: Optional[float],
    consensus: Optional[float] = None,
) -> None:
    """
    Log a forecast made today for target_date.

    Called after each scan cycle for each city+date combination.
    Upserts: if a forecast for this (city, target_date, forecast_date) already exists, updates it.
    """
    data = _load()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    timestamp = datetime.now(timezone.utc).isoformat()

    city_data = data.setdefault(city_code, {})
    date_entry = city_data.setdefault(target_date, {
        "target_date": target_date,
        "city": city_code,
        "forecasts": {},
        "actual_high": None,
        "settled": False,
    })

    date_entry["forecasts"][today] = {
        "nws": nws,
        "gfs": gfs,
        "icon": icon,
        "gem": gem,
        "consensus": consensus,
        "days_out": days_out,
        "logged_at": timestamp,
    }

    _save(data)
    logger.debug(
        f"[AccuracyTracker] {city_code}/{target_date} ({days_out}d out): "
        f"NWS={nws}, GFS={gfs}, ICON={icon}, GEM={gem}, consensus={consensus}"
    )


def record_actual(city_code: str, target_date: str, actual_high: float) -> None:
    """
    Record the actual observed high for a city+date and compute errors.

    Called when historical data comes in for past dates.
    """
    data = _load()

    city_data = data.setdefault(city_code, {})
    date_entry = city_data.get(target_date)
    if date_entry is None:
        logger.debug(f"[AccuracyTracker] No forecasts tracked for {city_code}/{target_date} — creating stub")
        date_entry = {
            "target_date": target_date,
            "city": city_code,
            "forecasts": {},
            "actual_high": None,
            "settled": False,
        }
        city_data[target_date] = date_entry

    if date_entry.get("settled"):
        logger.debug(f"[AccuracyTracker] {city_code}/{target_date} already settled — skipping")
        return

    date_entry["actual_high"] = actual_high
    date_entry["settled"] = True
    date_entry["settled_at"] = datetime.now(timezone.utc).isoformat()

    # Compute errors for each forecast
    sources = ["nws", "gfs", "icon", "gem", "consensus"]
    for fc_date, fc in date_entry["forecasts"].items():
        errors = {}
        for src in sources:
            val = fc.get(src)
            if val is not None:
                errors[src] = round(val - actual_high, 2)
        fc["errors"] = errors

    _save(data)
    logger.info(
        f"[AccuracyTracker] ✓ Settled {city_code}/{target_date}: actual={actual_high}°F "
        f"({len(date_entry['forecasts'])} forecast snapshots)"
    )


def get_accuracy_stats(city_code: str) -> dict:
    """
    Return MAE/bias stats per source for a city, split by days_out.

    Returns:
        {
            "1": {"nws_mae": 2.1, "gfs_mae": 1.8, ..., "n": 14},
            "2": {"nws_mae": 3.4, "gfs_mae": 2.9, ..., "n": 12},
        }
    """
    data = _load()
    city_data = data.get(city_code, {})
    sources = ["nws", "gfs", "icon", "gem", "consensus"]

    buckets: dict[str, list[tuple]] = {}

    for _date, entry in city_data.items():
        if not entry.get("settled") or entry.get("actual_high") is None:
            continue
        actual = entry["actual_high"]
        for _fc_date, fc in entry["forecasts"].items():
            days_out = str(fc.get("days_out", 1))
            if days_out not in buckets:
                buckets[days_out] = []
            buckets[days_out].append((actual, fc))

    result = {}
    for days_key, entries in buckets.items():
        n = len(entries)
        stat = {"n": n}
        bias_sums: dict[str, list[float]] = {s: [] for s in sources}

        for actual, fc in entries:
            errs = fc.get("errors", {})
            for src in sources:
                err = errs.get(src)
                if err is None and fc.get(src) is not None:
                    err = fc[src] - actual
                if err is not None:
                    bias_sums[src].append(err)

        for src in sources:
            errs = bias_sums[src]
            if errs:
                stat[f"{src}_mae"] = round(sum(abs(e) for e in errs) / len(errs), 2)
                stat[f"{src}_bias"] = round(sum(errs) / len(errs), 2)
            else:
                stat[f"{src}_mae"] = None
                stat[f"{src}_bias"] = None

        result[days_key] = stat

    return result


def get_model_weights(city_code: str, days_out: int = 1) -> dict:
    """
    Return optimal model weights for a specific city based on recent accuracy.

    Uses inverse-MAE weighting. Falls back to DEFAULT_WEIGHTS if < MIN_SAMPLES settled.
    This is the city-aware version — different cities may favor different models
    (e.g., coastal cities may have better GFS forecasts, mountain cities may favor GEM).

    Returns: {"nws": 0.30, "gfs": 0.25, "icon": 0.25, "gem": 0.20}
    """
    sources = ["nws", "gfs", "icon", "gem"]
    stats = get_accuracy_stats(city_code)
    key = str(days_out)
    s = stats.get(key, {})
    n = s.get("n", 0)

    if n < MIN_SAMPLES:
        logger.debug(
            f"[AccuracyTracker] {city_code}: only {n} settled points for {days_out}d-out "
            f"— using equal weights"
        )
        return dict(DEFAULT_WEIGHTS)

    maes = {}
    for src in sources:
        mae = s.get(f"{src}_mae")
        if mae is None or mae <= 0:
            mae = 2.5  # fallback
        maes[src] = mae

    # Weight inversely proportional to MAE
    inv = {src: 1.0 / maes[src] for src in sources}
    total = sum(inv.values())
    weights = {src: round(inv[src] / total, 4) for src in sources}

    # Log notable per-city biases
    for src in sources:
        shift = abs(weights[src] - 0.25)
        if shift > 0.08:
            logger.info(
                f"[AccuracyTracker] {city_code} {days_out}d-out: "
                f"{src} weight={weights[src]:.3f} (shift={shift:+.3f}, mae={maes[src]:.1f}°F, n={n})"
            )

    return weights


def get_all_city_summaries() -> dict:
    """Return a summary of accuracy stats for all tracked cities."""
    data = _load()
    result = {}
    for city_code in data:
        stats = get_accuracy_stats(city_code)
        settled_count = sum(
            1 for entry in data[city_code].values()
            if entry.get("settled")
        )
        result[city_code] = {
            "settled_dates": settled_count,
            "stats_by_days_out": stats,
        }
    return result
