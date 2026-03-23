"""Hypothesis 2: Volume / Liquidity Edge Analysis

Tests: "Does high relative volume on a bucket predict it winning,
after controlling for price?"

Dataset: data/kalshi_history/settled_markets.json
Output:  analysis/results/h2_results.json + console tables

Success metric: After controlling for price, does top-volume status
add significant predictive power? (p < 0.05, n >= 50)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "settled_markets.json"
RESULTS_FILE = Path(__file__).resolve().parent / "results" / "h2_results.json"


def load_and_group() -> dict[str, list[dict]]:
    """Load markets, group by event_ticker."""
    with open(DATA_FILE) as f:
        raw = json.load(f)
    
    events: dict[str, list[dict]] = defaultdict(list)
    for m in raw:
        ticker = m.get("ticker", "")
        status = m.get("status", "")
        result = m.get("result", "")
        
        if status != "finalized" or "HIGHAUS" not in ticker or result not in ("yes", "no"):
            continue
        
        try:
            last_price = float(m.get("last_price_dollars", "0"))
            volume = float(m.get("volume_fp", "0"))
            open_interest = float(m.get("open_interest_fp", "0"))
        except (ValueError, TypeError):
            continue
        
        if volume <= 0:
            continue
        
        events[m["event_ticker"]].append({
            "ticker": ticker,
            "event_ticker": m["event_ticker"],
            "result": result,
            "won": result == "yes",
            "last_price": last_price,
            "volume": volume,
            "open_interest": open_interest,
            "strike_type": m.get("strike_type", ""),
        })
    
    return events


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0, center - spread), min(1, center + spread))


def significance_test(win_rate: float, baseline: float, n: int) -> tuple[float, float]:
    """Z-test of win_rate vs baseline."""
    if n == 0 or baseline <= 0 or baseline >= 1:
        return (0.0, 1.0)
    se = math.sqrt(baseline * (1 - baseline) / n)
    if se == 0:
        return (0.0, 1.0)
    z = (win_rate - baseline) / se
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return (round(z, 3), round(p, 6))


def run_analysis():
    print("=" * 70)
    print("HYPOTHESIS 2: VOLUME / LIQUIDITY EDGE ANALYSIS")
    print("=" * 70)
    print()
    
    events = load_and_group()
    print(f"Loaded {sum(len(v) for v in events.values())} markets across {len(events)} events")
    print()
    
    # Enrich each market with volume rank and share within its event
    enriched = []
    for event_ticker, markets in events.items():
        event_volume = sum(m["volume"] for m in markets)
        n_buckets = len(markets)
        avg_share = 1.0 / n_buckets if n_buckets > 0 else 0
        
        # Sort by volume descending to get ranks
        sorted_by_vol = sorted(markets, key=lambda m: m["volume"], reverse=True)
        for rank, m in enumerate(sorted_by_vol, 1):
            m["volume_rank"] = rank
            m["volume_share"] = m["volume"] / event_volume if event_volume > 0 else 0
            m["avg_share"] = avg_share
            m["relative_share"] = m["volume_share"] / avg_share if avg_share > 0 else 0
            m["n_siblings"] = n_buckets
            m["event_volume"] = event_volume
            enriched.append(m)
    
    print(f"Enriched {len(enriched)} markets with volume rank/share")
    print()
    
    # --- Analysis A: Highest-volume bucket per event ---
    print("-" * 70)
    print("ANALYSIS A: Does the highest-volume bucket win more often?")
    print("-" * 70)
    print()
    
    top_vol = [m for m in enriched if m["volume_rank"] == 1]
    not_top = [m for m in enriched if m["volume_rank"] > 1]
    
    top_n = len(top_vol)
    top_wins = sum(1 for m in top_vol if m["won"])
    top_wr = top_wins / top_n if top_n > 0 else 0
    top_avg_price = sum(m["last_price"] for m in top_vol) / top_n if top_n > 0 else 0
    
    not_n = len(not_top)
    not_wins = sum(1 for m in not_top if m["won"])
    not_wr = not_wins / not_n if not_n > 0 else 0
    not_avg_price = sum(m["last_price"] for m in not_top) / not_n if not_n > 0 else 0
    
    print(f"  Top-volume bucket: n={top_n}, wins={top_wins}, win_rate={top_wr:.1%}, avg_price=${top_avg_price:.3f}")
    print(f"  Other buckets:     n={not_n}, wins={not_wins}, win_rate={not_wr:.1%}, avg_price=${not_avg_price:.3f}")
    print()
    print(f"  Raw difference: {top_wr - not_wr:+.1%}")
    print(f"  But top-volume buckets also have higher avg price (${top_avg_price:.3f} vs ${not_avg_price:.3f})")
    print(f"  => Must control for price before concluding anything.")
    
    # --- Analysis B: Volume share > 2x average ---
    print()
    print("-" * 70)
    print("ANALYSIS B: Buckets with disproportionate volume share (> 2x average)")
    print("-" * 70)
    print()
    
    high_share = [m for m in enriched if m["relative_share"] > 2.0]
    low_share = [m for m in enriched if m["relative_share"] <= 2.0]
    
    hs_n = len(high_share)
    hs_wins = sum(1 for m in high_share if m["won"])
    hs_wr = hs_wins / hs_n if hs_n > 0 else 0
    hs_avg_price = sum(m["last_price"] for m in high_share) / hs_n if hs_n > 0 else 0
    
    ls_n = len(low_share)
    ls_wins = sum(1 for m in low_share if m["won"])
    ls_wr = ls_wins / ls_n if ls_n > 0 else 0
    ls_avg_price = sum(m["last_price"] for m in low_share) / ls_n if ls_n > 0 else 0
    
    print(f"  High share (>2x): n={hs_n}, wins={hs_wins}, win_rate={hs_wr:.1%}, avg_price=${hs_avg_price:.3f}")
    print(f"  Normal share:     n={ls_n}, wins={ls_wins}, win_rate={ls_wr:.1%}, avg_price=${ls_avg_price:.3f}")
    
    # --- Analysis C: Control for price ---
    print()
    print("-" * 70)
    print("ANALYSIS C: Volume edge CONTROLLING FOR PRICE")
    print("-" * 70)
    print()
    print("Within each price bin, compare high-volume vs low-volume buckets.")
    print("This isolates the volume signal from the price signal.")
    print()
    
    # Bin by price, then split by volume rank
    BIN_WIDTH = 0.10  # Wider bins for more samples per cell
    
    price_bins: dict[float, dict[str, list[dict]]] = defaultdict(lambda: {"high_vol": [], "low_vol": []})
    
    for m in enriched:
        bin_mid = round((int(m["last_price"] / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2), 2)
        if m["volume_rank"] <= 2:  # Top-2 volume
            price_bins[bin_mid]["high_vol"].append(m)
        else:
            price_bins[bin_mid]["low_vol"].append(m)
    
    header = f"{'Price Bin':<12} {'HiVol_N':>8} {'HiVol_WR':>9} {'LoVol_N':>8} {'LoVol_WR':>9} {'Diff':>8} {'Z':>7} {'p':>8} {'Sig':>4}"
    print(header)
    print("-" * len(header))
    
    total_hi_n = 0
    total_hi_wins = 0
    total_lo_n = 0
    total_lo_wins = 0
    sig_bins = 0
    
    results_by_bin = []
    
    for bin_mid in sorted(price_bins.keys()):
        hi = price_bins[bin_mid]["high_vol"]
        lo = price_bins[bin_mid]["low_vol"]
        
        if len(hi) < 10 or len(lo) < 10:
            continue
        
        hi_n = len(hi)
        hi_wins = sum(1 for m in hi if m["won"])
        hi_wr = hi_wins / hi_n
        
        lo_n = len(lo)
        lo_wins = sum(1 for m in lo if m["won"])
        lo_wr = lo_wins / lo_n
        
        diff = hi_wr - lo_wr
        
        # Pooled proportion test
        pooled_p = (hi_wins + lo_wins) / (hi_n + lo_n)
        if pooled_p <= 0 or pooled_p >= 1:
            z, p = 0.0, 1.0
        else:
            se = math.sqrt(pooled_p * (1 - pooled_p) * (1/hi_n + 1/lo_n))
            z = diff / se if se > 0 else 0
            p = 0.5 * math.erfc(z / math.sqrt(2))
        
        sig = p < 0.05 and diff > 0 and hi_n >= 30 and lo_n >= 30
        sig_marker = " ***" if sig else ""
        if sig:
            sig_bins += 1
        
        total_hi_n += hi_n
        total_hi_wins += hi_wins
        total_lo_n += lo_n
        total_lo_wins += lo_wins
        
        bin_label = f"${bin_mid - BIN_WIDTH/2:.2f}-${bin_mid + BIN_WIDTH/2 - 0.01:.2f}"
        print(f"{bin_label:<12} {hi_n:>8} {hi_wr:>8.1%} {lo_n:>8} {lo_wr:>8.1%} {diff:>+7.1%} {z:>7.2f} {p:>8.4f}{sig_marker}")
        
        results_by_bin.append({
            "bin_midpoint": bin_mid,
            "high_vol_n": hi_n,
            "high_vol_win_rate": round(hi_wr, 4),
            "low_vol_n": lo_n,
            "low_vol_win_rate": round(lo_wr, 4),
            "diff": round(diff, 4),
            "z_score": round(z, 3),
            "p_value": round(p, 6),
            "significant": sig,
        })
    
    # Overall aggregate
    print()
    if total_hi_n > 0 and total_lo_n > 0:
        agg_hi_wr = total_hi_wins / total_hi_n
        agg_lo_wr = total_lo_wins / total_lo_n
        print(f"Aggregate (within price-controlled bins):")
        print(f"  High-volume: {total_hi_wins}/{total_hi_n} = {agg_hi_wr:.1%}")
        print(f"  Low-volume:  {total_lo_wins}/{total_lo_n} = {agg_lo_wr:.1%}")
        print(f"  Difference:  {agg_hi_wr - agg_lo_wr:+.1%}")
    
    # --- Verdict ---
    print()
    print("=" * 70)
    if sig_bins > 0:
        print(f"VERDICT: POTENTIAL VOLUME EDGE IN {sig_bins} PRICE BIN(S)")
        print("  However, consider multiple comparison correction.")
    else:
        print("VERDICT: NO SIGNIFICANT VOLUME EDGE FOUND")
        print("  After controlling for price, high relative volume does NOT")
        print("  significantly predict winning. The volume edge hypothesis")
        print("  is NOT SUPPORTED by this data.")
    print("=" * 70)
    
    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "n_markets": len(enriched),
        "n_events": len(events),
        "analysis_c_bins": results_by_bin,
        "significant_bins": sig_bins,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_analysis()
