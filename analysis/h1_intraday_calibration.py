"""Intraday Price Calibration: "When a bucket trades at $X at ANY point, how often does it win?"

Uses bins_hit data from intraday_analysis.json and unbiased_analysis.json.
These files record every price level (in $0.05 increments) that each ticker 
traded at during its lifetime, along with whether it ultimately won.

This answers:
- "When the market hits $0.80 on a YES, how often does that bucket win?"
- "Is there a proven edge at the $0.50-$0.68 sweet spot?"

With ~800+ tickers and multiple price levels each, we get 5,000+ data points 
instead of the 10-50 we had from last_price analysis.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

INTRADAY_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "intraday_analysis.json"
UNBIASED_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "unbiased_analysis.json"


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0, center - spread), min(1, center + spread))


def load_data() -> list[dict]:
    """Load and merge both analysis files. Deduplicate by ticker."""
    tickers_seen = set()
    records = []
    
    for path in [INTRADAY_FILE, UNBIASED_FILE]:
        if not path.exists():
            continue
        with open(path) as f:
            raw = json.load(f)
        for entry in raw:
            ticker = entry.get("ticker", "")
            if ticker in tickers_seen:
                continue
            tickers_seen.add(ticker)
            
            result = entry.get("result", "")
            won = entry.get("won", False)
            bins_hit = entry.get("bins_hit", [])
            
            if result not in ("yes", "no") or not bins_hit:
                continue
            
            records.append({
                "ticker": ticker,
                "won": won,
                "bins_hit": [round(b, 2) for b in bins_hit],
                "n_trades": entry.get("n_trades", 0),
            })
    
    return records


def run_analysis():
    print("=" * 70)
    print("INTRADAY PRICE CALIBRATION")
    print("When a bucket trades at $X at ANY point, how often does it win?")
    print("=" * 70)
    print()
    
    records = load_data()
    print(f"Loaded {len(records)} unique tickers with bins_hit data")
    winners = sum(1 for r in records if r["won"])
    print(f"  Winners: {winners}, Losers: {len(records) - winners}")
    print()
    
    # For each price level, count how many tickers touched it and how many won
    # bins_hit values are in $0.05 increments (0.0, 0.05, 0.10, ...)
    price_bins: dict[float, dict] = defaultdict(lambda: {"touched": 0, "won": 0})
    
    for r in records:
        for price in set(r["bins_hit"]):  # deduplicate within same ticker
            price_bins[price]["touched"] += 1
            if r["won"]:
                price_bins[price]["won"] += 1
    
    # Print full calibration table
    print("-" * 70)
    print("FULL CALIBRATION TABLE")
    print("'Touched' = number of tickers that traded at this price level")
    print("'Won' = how many of those tickers ultimately settled YES")
    print("-" * 70)
    print()
    
    header = f"{'Price':>7} {'Touched':>8} {'Won':>6} {'WinRate':>8} {'Implied':>8} {'Edge':>8} {'95% CI':>16} {'Sig':>4}"
    print(header)
    print("-" * len(header))
    
    for price in sorted(price_bins.keys()):
        d = price_bins[price]
        n = d["touched"]
        w = d["won"]
        wr = w / n if n > 0 else 0
        implied = price
        edge = wr - implied
        ci_lo, ci_hi = wilson_ci(w, n)
        
        # Significance test
        if n >= 30 and implied > 0 and implied < 1:
            se = math.sqrt(implied * (1 - implied) / n)
            z = (wr - implied) / se if se > 0 else 0
            p = 0.5 * math.erfc(z / math.sqrt(2))
            sig = "***" if p < 0.05 and edge > 0 else ""
        else:
            sig = ""
        
        print(f"  ${price:.2f} {n:>8} {w:>6} {wr:>7.1%} {implied:>7.1%} {edge:>+7.1%}  [{ci_lo:.1%}, {ci_hi:.1%}] {sig:>4}")
    
    # Specific questions
    print()
    print("=" * 70)
    print("SPECIFIC QUESTIONS")
    print("=" * 70)
    
    # Q1: When market hits $0.80, how often does it win?
    print()
    print("Q: When the market hits $0.80 YES, how often does the bucket win?")
    p80 = price_bins.get(0.80, {"touched": 0, "won": 0})
    if p80["touched"] > 0:
        wr = p80["won"] / p80["touched"]
        ci_lo, ci_hi = wilson_ci(p80["won"], p80["touched"])
        print(f"A: {p80['won']}/{p80['touched']} = {wr:.1%} win rate")
        print(f"   95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")
        print(f"   Implied probability at $0.80 = 80.0%")
        print(f"   Edge: {wr - 0.80:+.1%}")
        if wr > 0.80:
            print(f"   => Buckets that reach $0.80 win MORE than 80% of the time")
        else:
            print(f"   => Buckets that reach $0.80 win LESS than 80% of the time")
    
    # Also check $0.75 and $0.85 neighbors
    for p in [0.75, 0.85, 0.90, 0.95]:
        d = price_bins.get(p, {"touched": 0, "won": 0})
        if d["touched"] > 0:
            wr = d["won"] / d["touched"]
            ci_lo, ci_hi = wilson_ci(d["won"], d["touched"])
            print(f"   At ${p:.2f}: {d['won']}/{d['touched']} = {wr:.1%} [{ci_lo:.1%}, {ci_hi:.1%}]")
    
    # Q2: Sweet spot $0.50-$0.68
    print()
    print("Q: Is there a proven edge in the $0.50-$0.68 sweet spot?")
    sweet_prices = [p for p in sorted(price_bins.keys()) if 0.45 <= p <= 0.70]
    
    total_touched = 0
    total_won = 0
    for p in sweet_prices:
        d = price_bins[p]
        total_touched += d["touched"]
        total_won += d["won"]
    
    if total_touched > 0:
        wr = total_won / total_touched
        ci_lo, ci_hi = wilson_ci(total_won, total_touched)
        avg_implied = sum(sweet_prices) / len(sweet_prices)
        print(f"A: Across ${min(sweet_prices):.2f}-${max(sweet_prices):.2f}:")
        print(f"   {total_won}/{total_touched} = {wr:.1%} win rate")
        print(f"   95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")
        print(f"   Average implied probability: {avg_implied:.1%}")
        print(f"   Edge vs average implied: {wr - avg_implied:+.1%}")
    
    # Detail by price within sweet spot
    print()
    print("   Breakdown:")
    for p in sweet_prices:
        d = price_bins[p]
        n = d["touched"]
        w = d["won"]
        wr = w / n if n > 0 else 0
        edge = wr - p
        ci_lo, ci_hi = wilson_ci(w, n)
        print(f"   ${p:.2f}: {w:>4}/{n:<4} = {wr:>5.1%}  (implied {p:.0%}, edge {edge:+.1%})  [{ci_lo:.1%}, {ci_hi:.1%}]")
    
    # Q3: How to get more data / prove it
    print()
    print("=" * 70)
    print("HOW TO PROVE THE SWEET-SPOT EDGE")
    print("=" * 70)
    print()
    print("Current dataset has historical trade data from the Kalshi API.")
    print("The bins_hit data gives us much larger samples than last_price.")
    print()
    print("To get definitive proof, you need:")
    print("  1. MORE SETTLED MARKETS — the Kalshi API can fetch more historical")
    print("     events beyond what's in settled_markets.json")
    print("  2. INTRADAY PRICE SNAPSHOTS — polling Kalshi prices every 30-60 min")
    print("     during trading hours creates time-series data that reveals")
    print("     when during the day the edge is largest")
    print("  3. FORWARD TRACKING — start logging every trade decision + outcome")
    print("     to build a prospective (not retrospective) dataset")
    print()
    print("Rule of thumb for sample size:")
    print("  To detect a 5% edge at p<0.05 with 80% power, you need ~780 observations")
    print("  To detect a 10% edge at p<0.05 with 80% power, you need ~200 observations")
    print("  To detect a 15% edge at p<0.05 with 80% power, you need ~90 observations")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
