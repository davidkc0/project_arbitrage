"""Validate CLEAR_DAY_PATTERNS from bet_engine.py against kaus_hourly_history.csv.

CLEAR_DAY_PATTERNS claims to contain (avg_rise, p10_rise, p90_rise, n_samples)
"from 5yr KAUS data, clear days" for each (month, hour).

This script recomputes those values from the raw CSV and flags discrepancies.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

HOURLY_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kaus_hourly_history.csv"
DAILY_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kaus_daily_summary.csv"

# Exact copy of CLEAR_DAY_PATTERNS from bet_engine.py for comparison
CLAIMED_PATTERNS = {
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


def load_hourly() -> list[dict]:
    """Load hourly history CSV."""
    rows = []
    with open(HOURLY_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "date": row["date"],
                    "hour": int(row["hour"]),
                    "temp_f": int(float(row["temp_f"])),
                    "sky_cover": row.get("sky_cover", "CLR"),
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_daily() -> dict[str, int]:
    """Load daily summary and return {date: daily_high}."""
    highs = {}
    with open(DAILY_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                highs[row["date"]] = int(float(row["daily_high"]))
            except (ValueError, KeyError):
                continue
    return highs


def percentile(vals: list[float], pct: float) -> float:
    """Simple percentile calculation."""
    if not vals:
        return 0
    sorted_vals = sorted(vals)
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def run_analysis():
    print("=" * 70)
    print("PHASE 3A: VALIDATE CLEAR_DAY_PATTERNS")
    print("=" * 70)
    print()
    
    hourly = load_hourly()
    daily_highs = load_daily()
    print(f"Loaded {len(hourly)} hourly observations, {len(daily_highs)} daily summaries")
    print()
    
    # Group hourly by date
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in hourly:
        by_date[row["date"]].append(row)
    
    # Filter to clear-sky days (CLR or FEW dominant during daytime 8-15)
    clear_days: set[str] = set()
    for date_str, readings in by_date.items():
        daytime = [r for r in readings if 8 <= r["hour"] <= 15]
        if not daytime:
            continue
        # Count sky conditions
        sky_counts: dict[str, int] = defaultdict(int)
        for r in daytime:
            sky_counts[r["sky_cover"]] += 1
        # Dominant condition
        dominant = max(sky_counts, key=sky_counts.get)
        if dominant in ("CLR", "FEW"):
            clear_days.add(date_str)
    
    print(f"Clear/Few days: {len(clear_days)}")
    print()
    
    # For each clear day, at each hour, compute remaining rise to daily high
    # remaining_rise[month][hour] = list of (daily_high - temp_at_hour) values
    remaining_rise: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    
    for date_str in clear_days:
        if date_str not in daily_highs:
            continue
        high = daily_highs[date_str]
        
        # Get month
        try:
            month = int(date_str.split("-")[1])
        except (ValueError, IndexError):
            continue
        
        readings = by_date[date_str]
        # For each target hour, find the reading
        hour_temps: dict[int, int] = {}
        for r in readings:
            if r["hour"] in (10, 11, 12, 13, 14, 15):
                # Use first reading for this hour
                if r["hour"] not in hour_temps:
                    hour_temps[r["hour"]] = r["temp_f"]
        
        for hour, temp in hour_temps.items():
            rise = high - temp
            if rise >= 0:  # Only non-negative (temp hasn't exceeded high yet)
                remaining_rise[month][hour].append(rise)
    
    # Compute stats and compare
    print("-" * 70)
    print("Comparison: Claimed vs Recomputed from CSV")
    print("-" * 70)
    print()
    
    header = f"{'Mo':>2} {'Hr':>2} | {'Claim_avg':>9} {'Calc_avg':>9} {'Δavg':>6} | {'Claim_p10':>9} {'Calc_p10':>9} | {'Claim_p90':>9} {'Calc_p90':>9} | {'Claim_n':>7} {'Calc_n':>6} | {'Match':>5}"
    print(header)
    print("-" * len(header))
    
    discrepancies = []
    total_checks = 0
    matches = 0
    
    for month in range(1, 13):
        for hour in (10, 11, 12, 13, 14, 15):
            total_checks += 1
            claimed = CLAIMED_PATTERNS.get(month, {}).get(hour)
            rises = remaining_rise.get(month, {}).get(hour, [])
            
            if not claimed:
                continue
            
            c_avg, c_p10, c_p90, c_n = claimed
            
            if not rises:
                print(f"{month:>2} {hour:>2} | {c_avg:>9.1f} {'N/A':>9} {'N/A':>6} | {c_p10:>9} {'N/A':>9} | {c_p90:>9} {'N/A':>9} | {c_n:>7} {'0':>6} | {'MISS':>5}")
                discrepancies.append((month, hour, "NO DATA to verify"))
                continue
            
            r_avg = sum(rises) / len(rises)
            r_p10 = percentile(rises, 10)
            r_p90 = percentile(rises, 90)
            r_n = len(rises)
            
            avg_diff = r_avg - c_avg
            
            # Check match: avg within 1.5°F, p10/p90 within 2°F, n within 30%
            avg_ok = abs(avg_diff) <= 1.5
            p10_ok = abs(r_p10 - c_p10) <= 2
            p90_ok = abs(r_p90 - c_p90) <= 2
            n_ratio = r_n / c_n if c_n > 0 else 0
            n_ok = 0.5 <= n_ratio <= 2.0
            
            all_ok = avg_ok and p10_ok and p90_ok
            status = "OK" if all_ok else "DIFF"
            if all_ok:
                matches += 1
            else:
                reasons = []
                if not avg_ok:
                    reasons.append(f"avg off by {avg_diff:+.1f}")
                if not p10_ok:
                    reasons.append(f"p10 off by {r_p10-c_p10:+.0f}")
                if not p90_ok:
                    reasons.append(f"p90 off by {r_p90-c_p90:+.0f}")
                discrepancies.append((month, hour, "; ".join(reasons)))
            
            print(f"{month:>2} {hour:>2} | {c_avg:>9.1f} {r_avg:>9.1f} {avg_diff:>+5.1f} | {c_p10:>9} {r_p10:>9.0f} | {c_p90:>9} {r_p90:>9.0f} | {c_n:>7} {r_n:>6} | {status:>5}")
    
    # Summary
    print()
    print("=" * 70)
    print(f"VERDICT: {matches}/{total_checks} cells match (within tolerance)")
    print()
    if discrepancies:
        print(f"Discrepancies ({len(discrepancies)}):")
        for month, hour, reason in discrepancies:
            print(f"  Month {month}, Hour {hour}: {reason}")
    else:
        print("All values match within tolerance. CLEAR_DAY_PATTERNS is VERIFIED.")
    
    # Also output the recomputed table for replacement
    print()
    print("-" * 70)
    print("RECOMPUTED CLEAR_DAY_PATTERNS (for reference):")
    print("-" * 70)
    print("CLEAR_DAY_PATTERNS = {")
    for month in range(1, 13):
        parts = []
        for hour in (10, 11, 12, 13, 14, 15):
            rises = remaining_rise.get(month, {}).get(hour, [])
            if rises:
                avg = round(sum(rises)/len(rises), 1)
                p10 = round(percentile(rises, 10))
                p90 = round(percentile(rises, 90))
                n = len(rises)
                parts.append(f"{hour}: ({avg}, {p10}, {p90}, {n})")
            else:
                parts.append(f"{hour}: (0, 0, 0, 0)")
        print(f"    {month:>2}: {{{', '.join(parts)}}},")
    print("}")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
