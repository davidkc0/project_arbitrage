"""Persistent forecast accuracy tracker.

Saves NWS + model forecasts when made, matches against actual highs when
they arrive, and computes dynamic weights for the consensus model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent / "data" / "forecast_accuracy.json"


def _load() -> list[dict]:
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[Tracker] Could not load {DATA_FILE}: {e}")
    return []


def _save(data: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_forecast(
    target_date: str,
    forecast_date: str,
    nws: Optional[float],
    models_dict: dict,
    days_out: int,
) -> None:
    """
    Upsert a forecast entry for target_date, made on forecast_date.

    models_dict should have keys: gfs, icon, gem (all optional floats).
    """
    data = _load()

    # Find or create entry for target_date
    entry = next((e for e in data if e["target_date"] == target_date), None)
    if entry is None:
        entry = {
            "target_date": target_date,
            "forecasts": {},
            "actual_high": None,
            "settled": False,
        }
        data.append(entry)

    # Upsert forecast for this forecast_date
    entry["forecasts"][forecast_date] = {
        "nws": nws,
        "gfs": models_dict.get("gfs"),
        "icon": models_dict.get("icon"),
        "gem": models_dict.get("gem"),
        "days_out": days_out,
    }

    _save(data)
    logger.info(
        f"[Tracker] Saved forecast for {target_date} "
        f"(made {forecast_date}, {days_out}d out): "
        f"NWS={nws}, GFS={models_dict.get('gfs')}, "
        f"ICON={models_dict.get('icon')}, GEM={models_dict.get('gem')}"
    )


def record_actual(target_date: str, actual_high: float) -> None:
    """Record the actual observed high for target_date and mark as settled."""
    data = _load()

    entry = next((e for e in data if e["target_date"] == target_date), None)
    if entry is None:
        # No forecast was tracked for this date — create a stub so we don't lose the actual
        logger.info(f"[Tracker] No forecast entry for {target_date} — creating stub with actual")
        entry = {
            "target_date": target_date,
            "forecasts": {},
            "actual_high": None,
            "settled": False,
        }
        data.append(entry)

    if not entry.get("settled"):
        entry["actual_high"] = actual_high
        entry["settled"] = True
        _save(data)
        logger.info(f"[Tracker] Recorded actual high for {target_date}: {actual_high}°F ✓")
    else:
        logger.debug(f"[Tracker] {target_date} already settled at {entry['actual_high']}°F — skipping")


def get_accuracy_stats() -> dict:
    """
    Return bias stats split by days_out.

    Returns:
        {
            "1_day_out": {"nws_bias": 2.1, "gfs_bias": -0.3, ..., "n": 14},
            "2_day_out": {"nws_bias": 3.4, "gfs_bias": -0.1, ..., "n": 12},
        }
    Bias = forecast - actual (positive = model runs hot).
    """
    data = _load()
    settled = [e for e in data if e.get("settled") and e.get("actual_high") is not None]

    sources = ["nws", "gfs", "icon", "gem"]
    buckets: dict[str, list[tuple]] = {"1": [], "2": []}

    for entry in settled:
        actual = entry["actual_high"]
        for _fc_date, fc in entry["forecasts"].items():
            days_out = fc.get("days_out")
            if days_out not in (1, 2):
                continue
            buckets[str(days_out)].append((actual, fc))

    result = {}
    for days_key, entries in buckets.items():
        stat_key = f"{days_key}_day_out"
        if not entries:
            result[stat_key] = {
                f"{s}_bias": 0.0 for s in sources
            }
            result[stat_key].update({f"{s}_mae": 0.0 for s in sources})
            result[stat_key]["n"] = 0
            continue

        bias_errors: dict[str, list[float]] = {s: [] for s in sources}
        for actual, fc in entries:
            for src in sources:
                val = fc.get(src)
                if val is not None:
                    bias_errors[src].append(val - actual)

        n = len(entries)
        stat = {"n": n}
        for src in sources:
            errs = bias_errors[src]
            if errs:
                stat[f"{src}_bias"] = round(sum(errs) / len(errs), 2)
                stat[f"{src}_mae"] = round(sum(abs(e) for e in errs) / len(errs), 2)
            else:
                stat[f"{src}_bias"] = 0.0
                stat[f"{src}_mae"] = 0.0
        result[stat_key] = stat

    return result


def get_optimal_weights(days_out: int = 1) -> dict:
    """
    Return optimal weights for each source based on historical MAE.

    Falls back to equal weights (0.25 each) if < 5 settled data points.
    Weights inversely proportional to MAE as data accumulates.
    """
    equal_weights = {"nws": 0.25, "gfs": 0.25, "icon": 0.25, "gem": 0.25}
    sources = ["nws", "gfs", "icon", "gem"]

    stats = get_accuracy_stats()
    key = f"{days_out}_day_out"
    s = stats.get(key, {})
    n = s.get("n", 0)

    if n < 5:
        logger.debug(f"[Tracker] Only {n} settled points for {days_out}d-out — using equal weights")
        return equal_weights

    maes = {}
    for src in sources:
        mae = s.get(f"{src}_mae")
        if mae is None or mae <= 0:
            mae = 2.5  # sensible default
        maes[src] = mae

    # Weight inversely proportional to MAE
    inv = {src: 1.0 / maes[src] for src in sources}
    total = sum(inv.values())
    weights = {src: round(inv[src] / total, 4) for src in sources}

    # Log significant shifts from equal weighting
    for src in sources:
        shift = abs(weights[src] - 0.25)
        if shift > 0.10:
            logger.info(
                f"[Tracker] Significant weight shift: {src}={weights[src]:.3f} "
                f"(equal=0.25, shift={shift:+.3f}) for {days_out}d-out (n={n})"
            )

    return weights
