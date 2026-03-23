"""Hypothesis 1: Price-Threshold Edge Analysis

Tests: "If a Kalshi KXHIGHAUS bucket was last trading at price P,
how often did it actually win? Is the realized win rate significantly
different from the implied probability P?"

Dataset: data/kalshi_history/settled_markets.json
Output:  analysis/results/h1_results.json + console tables

Success metric: Any price bin where win_rate - implied_prob > 0
with p < 0.05 and n >= 30.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "settled_markets.json"
RESULTS_FILE = Path(__file__).resolve().parent / "results" / "h1_results.json"

# Price bins: 0.00-0.04, 0.05-0.09, 0.10-0.14, ... 0.95-1.00
BIN_WIDTH = 0.05


def load_settled_markets() -> list[dict]:
    """Load and filter to finalized KXHIGHAUS/HIGHAUS markets only."""
    with open(DATA_FILE) as f:
        raw = json.load(f)

    markets = []
    for m in raw:
        ticker = m.get("ticker", "")
        status = m.get("status", "")
        result = m.get("result", "")

        # Only settled Austin high-temp markets
        if status != "finalized":
            continue
        if "HIGHAUS" not in ticker:
            continue
        if result not in ("yes", "no"):
            continue

        last_price_str = m.get("last_price_dollars", "0")
        volume_str = m.get("volume_fp", "0")

        try:
            last_price = float(last_price_str)
            volume = float(volume_str)
        except (ValueError, TypeError):
            continue

        # Skip markets with zero volume (never traded)
        if volume <= 0:
            continue

        markets.append({
            "ticker": ticker,
            "event_ticker": m.get("event_ticker", ""),
            "result": result,
            "won": result == "yes",
            "last_price": last_price,
            "volume": volume,
            "open_interest": float(m.get("open_interest_fp", "0")),
            "strike_type": m.get("strike_type", ""),
            "floor_strike": m.get("floor_strike"),
            "cap_strike": m.get("cap_strike"),
            "expiration_value": m.get("expiration_value"),
        })

    return markets


def price_to_bin(price: float) -> float:
    """Map a price to the midpoint of its bin. E.g. 0.53 -> 0.525"""
    bin_index = int(price / BIN_WIDTH)
    bin_low = bin_index * BIN_WIDTH
    bin_high = bin_low + BIN_WIDTH
    return round((bin_low + bin_high) / 2, 4)


def price_to_bin_label(price: float) -> str:
    """Human-readable bin label. E.g. 0.53 -> '0.50-0.54'"""
    bin_index = int(price / BIN_WIDTH)
    bin_low = bin_index * BIN_WIDTH
    bin_high = bin_low + BIN_WIDTH - 0.01
    return f"${bin_low:.2f}-${bin_high:.2f}"


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0, center - spread), min(1, center + spread))


def compute_edge_significance(win_rate: float, implied_prob: float, n: int) -> tuple[float, float]:
    """
    Test H0: true win rate = implied_prob.
    Returns (z_score, p_value_one_sided).
    """
    if n == 0 or implied_prob <= 0 or implied_prob >= 1:
        return (0.0, 1.0)
    se = math.sqrt(implied_prob * (1 - implied_prob) / n)
    if se == 0:
        return (0.0, 1.0)
    z = (win_rate - implied_prob) / se
    # One-sided p-value (we care about win_rate > implied_prob)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return (round(z, 3), round(p, 6))


def run_analysis():
    """Main analysis pipeline."""
    print("=" * 70)
    print("HYPOTHESIS 1: PRICE-THRESHOLD EDGE ANALYSIS")
    print("=" * 70)
    print()

    # Load data
    markets = load_settled_markets()
    print(f"Loaded {len(markets)} settled HIGHAUS markets with volume > 0")

    # Count unique events (dates)
    events = set(m["event_ticker"] for m in markets)
    print(f"Across {len(events)} unique event dates")
    print()

    # --- Analysis 1: Win rate by price bin ---
    print("-" * 70)
    print("ANALYSIS 1: Win Rate by Last-Traded Price Bin")
    print("-" * 70)
    print()

    bins: dict[float, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
    for m in markets:
        b = price_to_bin(m["last_price"])
        bins[b]["wins"] += int(m["won"])
        bins[b]["total"] += 1

    results_table = []
    for bin_mid in sorted(bins.keys()):
        data = bins[bin_mid]
        n = data["total"]
        wins = data["wins"]
        win_rate = wins / n if n > 0 else 0
        edge = win_rate - bin_mid
        z, p = compute_edge_significance(win_rate, bin_mid, n)
        ci_lo, ci_hi = wilson_ci(wins, n)

        results_table.append({
            "bin_midpoint": bin_mid,
            "bin_label": price_to_bin_label(bin_mid),
            "n": n,
            "wins": wins,
            "win_rate": round(win_rate, 4),
            "implied_prob": round(bin_mid, 4),
            "edge": round(edge, 4),
            "edge_pct": round(edge * 100, 2),
            "z_score": z,
            "p_value": p,
            "ci_95_low": round(ci_lo, 4),
            "ci_95_high": round(ci_hi, 4),
            "significant": p < 0.05 and n >= 30 and edge > 0,
        })

    # Print table
    header = f"{'Price Bin':<14} {'N':>6} {'Wins':>6} {'WinRate':>8} {'Implied':>8} {'Edge':>8} {'Z':>7} {'p':>8} {'Sig':>4}"
    print(header)
    print("-" * len(header))
    for r in results_table:
        sig_marker = " ***" if r["significant"] else ""
        print(
            f"{r['bin_label']:<14} {r['n']:>6} {r['wins']:>6} "
            f"{r['win_rate']:>7.1%} {r['implied_prob']:>7.1%} "
            f"{r['edge_pct']:>+7.1f}% {r['z_score']:>7.2f} "
            f"{r['p_value']:>8.4f}{sig_marker}"
        )

    # --- Analysis 2: Test the specific claimed sweet-spot ---
    print()
    print("-" * 70)
    print("ANALYSIS 2: Testing Claimed Sweet-Spot ($0.50-$0.68)")
    print("-" * 70)
    print()
    print("Claim from bet_engine.py line 411-413:")
    print('  "At $0.50 YES, buckets win 57% (+7% edge, +15% ROI)"')
    print('  "At $0.65 YES, buckets win 78% (+13% edge, +20% ROI)"')
    print()

    # Test specific price ranges
    sweet_spot_tests = [
        ("$0.45-$0.55", 0.45, 0.55),
        ("$0.50-$0.55", 0.50, 0.55),
        ("$0.55-$0.65", 0.55, 0.65),
        ("$0.60-$0.68", 0.60, 0.68),
        ("$0.65-$0.75", 0.65, 0.75),
    ]

    for label, lo, hi in sweet_spot_tests:
        subset = [m for m in markets if lo <= m["last_price"] < hi]
        if not subset:
            print(f"  {label}: NO DATA")
            continue
        n = len(subset)
        wins = sum(1 for m in subset if m["won"])
        wr = wins / n
        mid = (lo + hi) / 2
        z, p = compute_edge_significance(wr, mid, n)
        ci_lo_val, ci_hi_val = wilson_ci(wins, n)

        # ROI if we bought at average price and won $1
        avg_price = sum(m["last_price"] for m in subset) / n
        roi = (wr * 1.0 - avg_price) / avg_price if avg_price > 0 else 0

        print(f"  {label}: n={n}, wins={wins}, win_rate={wr:.1%}, "
              f"avg_price=${avg_price:.3f}, ROI={roi:+.1%}, "
              f"z={z:.2f}, p={p:.4f}, 95%CI=[{ci_lo_val:.1%}, {ci_hi_val:.1%}]")

    # --- Analysis 3: Expected value by price threshold ---
    print()
    print("-" * 70)
    print("ANALYSIS 3: Expected Value If You Bought YES at Each Price")
    print("-" * 70)
    print()
    print("If you bought YES at last-traded price, your EV per $1 risked:")
    print()

    ev_header = f"{'Price Range':<14} {'N':>6} {'Wins':>6} {'WinRate':>8} {'AvgCost':>8} {'EV/trade':>10} {'ROI':>8}"
    print(ev_header)
    print("-" * len(ev_header))

    for r in results_table:
        if r["n"] < 10:
            continue
        avg_cost = r["bin_midpoint"]  # approximately
        ev_per_trade = r["win_rate"] * 1.0 - avg_cost  # win $1, lose cost
        roi = ev_per_trade / avg_cost if avg_cost > 0 else 0
        print(
            f"{r['bin_label']:<14} {r['n']:>6} {r['wins']:>6} "
            f"{r['win_rate']:>7.1%} ${avg_cost:>6.2f} "
            f"${ev_per_trade:>+8.4f} {roi:>+7.1%}"
        )

    # --- Analysis 4: Calibration check ---
    print()
    print("-" * 70)
    print("ANALYSIS 4: Overall Market Calibration")
    print("-" * 70)
    print()
    print("Perfect calibration = win rate equals implied probability at all price levels.")
    print()

    total_bins_tested = sum(1 for r in results_table if r["n"] >= 30)
    sig_positive = sum(1 for r in results_table if r["significant"])
    sig_negative = sum(1 for r in results_table if r["n"] >= 30 and r["edge"] < 0 and r["p_value"] < 0.05)
    well_calibrated = sum(1 for r in results_table if r["n"] >= 30 and abs(r["edge"]) < 0.05)

    print(f"  Price bins with n >= 30: {total_bins_tested}")
    print(f"  Bins with significant POSITIVE edge (win_rate > implied, p<0.05): {sig_positive}")
    print(f"  Bins with significant NEGATIVE edge (win_rate < implied, p<0.05): {sig_negative}")
    print(f"  Bins well-calibrated (|edge| < 5%): {well_calibrated}")

    # --- Verdict ---
    print()
    print("=" * 70)
    if sig_positive > 0:
        print("VERDICT: POTENTIAL EDGE FOUND")
        print(f"  {sig_positive} price bin(s) show statistically significant positive edge.")
        print("  However, multiple comparison correction may reduce significance.")
        print("  Bonferroni-adjusted threshold: p < " + f"{0.05/max(total_bins_tested,1):.4f}")

        # Re-check with Bonferroni
        bonf_threshold = 0.05 / max(total_bins_tested, 1)
        bonf_sig = sum(1 for r in results_table if r["n"] >= 30 and r["edge"] > 0 and r["p_value"] < bonf_threshold)
        if bonf_sig > 0:
            print(f"  After Bonferroni correction: {bonf_sig} bin(s) still significant.")
            print("  => EDGE SURVIVES MULTIPLE COMPARISON CORRECTION")
        else:
            print(f"  After Bonferroni correction: 0 bins remain significant.")
            print("  => EDGE DOES NOT SURVIVE MULTIPLE COMPARISON CORRECTION")
    else:
        print("VERDICT: NO SIGNIFICANT EDGE FOUND")
        print("  Market prices appear well-calibrated against realized outcomes.")
        print("  The price-threshold edge hypothesis is NOT SUPPORTED by this data.")
    print("=" * 70)

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "n_markets": len(markets),
        "n_events": len(events),
        "bins": results_table,
        "sweet_spot_claim_verified": any(
            r["significant"] and 0.475 <= r["bin_midpoint"] <= 0.675
            for r in results_table
        ),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_analysis()
