"""Validate hardcoded σ values in config.py.

config.py uses:
  FORECAST_STDEV_24H = 3.0  (NWS 24h forecast error)
  FORECAST_STDEV_48H = 4.0  (NWS 48h forecast error)

This script computes the actual forecast error distribution using:
  1. forecast_accuracy.json — NWS + model forecasts matched with actuals
  2. settled_markets.json — expiration_value gives the actual observed high

Output: Observed σ, MAE, bias for each source and lead time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

FORECAST_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "forecast_accuracy.json"
SETTLED_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "settled_markets.json"


def load_forecast_accuracy() -> list[dict]:
    """Load forecast accuracy tracker entries."""
    if not FORECAST_FILE.exists():
        return []
    with open(FORECAST_FILE) as f:
        return json.load(f)


def load_actual_highs_from_settlements() -> dict[str, float]:
    """Extract actual high temp per date from settled markets.
    
    Each event has an expiration_value that is the actual observed high.
    Returns {date_str: actual_high}.
    """
    with open(SETTLED_FILE) as f:
        raw = json.load(f)
    
    actuals: dict[str, float] = {}
    for m in raw:
        ticker = m.get("event_ticker", "")
        if "HIGHAUS" not in ticker:
            continue
        ev = m.get("expiration_value")
        if ev is None:
            continue
        try:
            actual = float(ev)
        except (ValueError, TypeError):
            continue
        
        # Parse date from event_ticker: KXHIGHAUS-26MAR21 -> 2026-03-21
        # or HIGHAUS-24MAR21 -> 2024-03-21
        parts = ticker.split("-")
        if len(parts) < 2:
            continue
        date_part = parts[-1]  # e.g. "26MAR21"
        
        try:
            # Extract YY, MMM, DD
            if len(date_part) >= 7:
                yy = int(date_part[:2])
                mmm = date_part[2:5]
                dd = int(date_part[5:])
                
                months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
                mm = months.get(mmm.upper())
                if mm is None:
                    continue
                year = 2000 + yy
                date_str = f"{year}-{mm:02d}-{dd:02d}"
                actuals[date_str] = actual
        except (ValueError, IndexError):
            continue
    
    return actuals


def stats(errors: list[float]) -> dict:
    """Compute summary statistics for a list of forecast errors."""
    if not errors:
        return {"n": 0, "mean": 0, "std": 0, "mae": 0, "rmse": 0}
    n = len(errors)
    mean = sum(errors) / n
    var = sum((e - mean) ** 2 for e in errors) / max(n - 1, 1)
    std = math.sqrt(var)
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    return {
        "n": n,
        "mean_bias": round(mean, 2),
        "std": round(std, 2),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
    }


def run_analysis():
    print("=" * 70)
    print("PHASE 1B: VALIDATE HARDCODED σ VALUES")
    print("=" * 70)
    print()
    print("config.py claims:")
    print("  FORECAST_STDEV_24H = 3.0  (NWS 24h forecast error)")
    print("  FORECAST_STDEV_48H = 4.0  (NWS 48h forecast error)")
    print()
    
    # --- Method 1: From forecast_accuracy.json ---
    print("-" * 70)
    print("METHOD 1: From forecast_accuracy.json (NWS + model forecasts)")
    print("-" * 70)
    print()
    
    entries = load_forecast_accuracy()
    settled = [e for e in entries if e.get("settled") and e.get("actual_high") is not None]
    print(f"Settled forecast entries: {len(settled)}")
    
    sources = ["nws", "gfs", "icon", "gem"]
    by_days_out: dict[int, dict[str, list[float]]] = {1: {s: [] for s in sources}, 2: {s: [] for s in sources}}
    
    for entry in settled:
        actual = entry["actual_high"]
        for fc_date, fc in entry.get("forecasts", {}).items():
            days_out = fc.get("days_out")
            if days_out not in (1, 2):
                continue
            for src in sources:
                val = fc.get(src)
                if val is not None:
                    by_days_out[days_out][src].append(val - actual)
    
    for days_out in [1, 2]:
        print(f"\n  {days_out}-day-out forecasts:")
        for src in sources:
            errors = by_days_out[days_out][src]
            s = stats(errors)
            if s["n"] > 0:
                print(f"    {src.upper():>4}: n={s['n']:>3}, bias={s['mean_bias']:>+5.1f}°F, "
                      f"σ={s['std']:.2f}°F, MAE={s['mae']:.2f}°F, RMSE={s['rmse']:.2f}°F")
            else:
                print(f"    {src.upper():>4}: NO DATA")
    
    # --- Method 2: From settled_markets (much larger sample) ---
    print()
    print("-" * 70)
    print("METHOD 2: Actual high temperature distribution from settled markets")
    print("-" * 70)
    print()
    
    actuals = load_actual_highs_from_settlements()
    print(f"Unique dates with actual highs from settlements: {len(actuals)}")
    
    if actuals:
        highs = sorted(actuals.values())
        mean_high = sum(highs) / len(highs)
        std_high = math.sqrt(sum((h - mean_high) ** 2 for h in highs) / max(len(highs) - 1, 1))
        print(f"  Mean actual high: {mean_high:.1f}°F")
        print(f"  Std of actual highs: {std_high:.1f}°F")
        print(f"  Range: {min(highs):.0f}°F to {max(highs):.0f}°F")
        
        # Day-to-day temperature variability gives upper bound on forecast error
        diffs = []
        sorted_dates = sorted(actuals.keys())
        for i in range(1, len(sorted_dates)):
            d1, d2 = sorted_dates[i-1], sorted_dates[i]
            diffs.append(actuals[d2] - actuals[d1])
        
        if diffs:
            s = stats(diffs)
            print(f"\n  Day-to-day high temp change:")
            print(f"    Mean change: {s['mean_bias']:+.1f}°F")
            print(f"    σ of change: {s['std']:.2f}°F")
            print(f"    MAE of change: {s['mae']:.2f}°F")
    
    # --- Verdict ---
    print()
    print("=" * 70)
    print("VERDICT ON σ VALUES:")
    print()
    
    # Use NWS forecast error if available
    nws_1d_errors = by_days_out[1].get("nws", [])
    nws_2d_errors = by_days_out[2].get("nws", [])
    
    if nws_1d_errors:
        s1 = stats(nws_1d_errors)
        print(f"  NWS 1-day-out: observed σ = {s1['std']:.2f}°F  (config says 3.00°F)")
        if abs(s1['std'] - 3.0) < 0.5:
            print(f"    => REASONABLE (within 0.5°F of claimed)")
        else:
            print(f"    => MISMATCH: should update config to σ={s1['std']:.1f}")
    else:
        print("  NWS 1-day-out: INSUFFICIENT DATA to validate σ=3.0")
        print("  (forecast_accuracy.json needs more settled entries)")
    
    if nws_2d_errors:
        s2 = stats(nws_2d_errors)
        print(f"  NWS 2-day-out: observed σ = {s2['std']:.2f}°F  (config says 4.00°F)")
        if abs(s2['std'] - 4.0) < 0.5:
            print(f"    => REASONABLE (within 0.5°F of claimed)")
        else:
            print(f"    => MISMATCH: should update config to σ={s2['std']:.1f}")
    else:
        print("  NWS 2-day-out: INSUFFICIENT DATA to validate σ=4.0")
    
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
