"""Liquidity analysis: Can you actually buy at $0.85 on Kalshi?

Examines:
1. Volume at the $0.85 price level from historical data
2. Bid-ask spread data from settled_markets.json
3. How many contracts per event trade near $0.85
4. How often a $0.85 opportunity even appears
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

SETTLED_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "settled_markets.json"
INTRADAY_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "intraday_analysis.json"
UNBIASED_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "unbiased_analysis.json"


def load_settled() -> list[dict]:
    with open(SETTLED_FILE) as f:
        raw = json.load(f)
    
    markets = []
    for m in raw:
        ticker = m.get("ticker", "")
        if "HIGHAUS" not in ticker or m.get("status") != "finalized":
            continue
        
        try:
            last_price = float(m.get("last_price_dollars", "0"))
            volume = float(m.get("volume_fp", "0"))
            open_interest = float(m.get("open_interest_fp", "0"))
            yes_bid = float(m.get("yes_bid_dollars", "0"))
            yes_ask = float(m.get("yes_ask_dollars", "0"))
        except (ValueError, TypeError):
            continue
        
        markets.append({
            "ticker": ticker,
            "event_ticker": m.get("event_ticker", ""),
            "result": m.get("result", ""),
            "won": m.get("result") == "yes",
            "last_price": last_price,
            "volume": volume,
            "open_interest": open_interest,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "floor_strike": m.get("floor_strike"),
            "cap_strike": m.get("cap_strike"),
            "strike_type": m.get("strike_type", ""),
        })
    
    return markets


def load_bins_hit() -> list[dict]:
    """Load tickers with bins_hit data."""
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
            records.append(entry)
    return records


def run_analysis():
    print("=" * 78)
    print("LIQUIDITY ANALYSIS: Can You Actually Buy at $0.85 on Kalshi?")
    print("=" * 78)
    print()
    
    markets = load_settled()
    bins_data = load_bins_hit()
    
    # ================================================================
    # 1. How often does a bucket reach $0.85?
    # ================================================================
    print("-" * 78)
    print("1. HOW OFTEN DOES A $0.85 OPPORTUNITY APPEAR?")
    print("-" * 78)
    print()
    
    tickers_at_85 = [r for r in bins_data if 0.85 in [round(b, 2) for b in r.get("bins_hit", [])]]
    total_tickers = len(bins_data)
    
    print(f"Total tickers in dataset: {total_tickers}")
    print(f"Tickers that ever traded at $0.85: {len(tickers_at_85)} ({len(tickers_at_85)/total_tickers:.1%})")
    
    # How many unique events?
    events_at_85 = set()
    for r in tickers_at_85:
        # Extract event from ticker
        parts = r["ticker"].rsplit("-", 1)
        if len(parts) >= 2:
            # The event ticker is everything after removing the bucket suffix
            # KXHIGHAUS-26MAR22-B81-T84 -> KXHIGHAUS-26MAR22
            t = r["ticker"]
            # Find the date part
            for i, p in enumerate(t.split("-")):
                if any(m in p for m in ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]):
                    events_at_85.add("-".join(t.split("-")[:i+1]))
                    break
    
    print(f"Unique event dates with a $0.85 bucket: {len(events_at_85)}")
    
    # Group bins_data by event to count events total
    all_events = set()
    for r in bins_data:
        t = r["ticker"]
        for i, p in enumerate(t.split("-")):
            if any(m in p for m in ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]):
                all_events.add("-".join(t.split("-")[:i+1]))
                break
    
    print(f"Total unique event dates: {len(all_events)}")
    if all_events:
        print(f"Events with a $0.85 bucket: {len(events_at_85)}/{len(all_events)} = {len(events_at_85)/len(all_events):.1%}")
    print()
    
    # Also check nearby prices
    for target_price in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        count = sum(1 for r in bins_data if target_price in [round(b, 2) for b in r.get("bins_hit", [])])
        print(f"  Tickers touching ${target_price:.2f}: {count}/{total_tickers} = {count/total_tickers:.1%}")
    
    # ================================================================
    # 2. Volume on contracts near $0.85
    # ================================================================
    print()
    print("-" * 78)
    print("2. VOLUME ON CONTRACTS NEAR $0.85 (from settled_markets.json)")
    print("-" * 78)
    print()
    
    # Markets whose last_price was near $0.85
    near_85 = [m for m in markets if 0.80 <= m["last_price"] <= 0.90]
    print(f"Markets with last_price $0.80-$0.90: {len(near_85)}")
    
    if near_85:
        volumes = [m["volume"] for m in near_85]
        oi = [m["open_interest"] for m in near_85]
        avg_vol = sum(volumes) / len(volumes)
        median_vol = sorted(volumes)[len(volumes) // 2]
        max_vol = max(volumes)
        min_vol = min(volumes)
        
        print(f"  Volume stats:")
        print(f"    Mean:   {avg_vol:.0f} contracts")
        print(f"    Median: {median_vol:.0f} contracts")
        print(f"    Min:    {min_vol:.0f} contracts")
        print(f"    Max:    {max_vol:.0f} contracts")
        
        if oi:
            avg_oi = sum(oi) / len(oi)
            print(f"  Open Interest: avg {avg_oi:.0f}")
    
    # All markets overall volume distribution
    all_volumes = [m["volume"] for m in markets if m["volume"] > 0]
    print()
    print(f"Overall volume distribution across ALL {len(all_volumes)} markets:")
    pcts = [10, 25, 50, 75, 90, 95, 99]
    sorted_vols = sorted(all_volumes)
    for p in pcts:
        idx = int(len(sorted_vols) * p / 100)
        print(f"  P{p}: {sorted_vols[min(idx, len(sorted_vols)-1)]:.0f} contracts")
    
    # ================================================================
    # 3. Bid-ask spread analysis
    # ================================================================
    print()
    print("-" * 78)
    print("3. BID-ASK SPREAD (snapshot at settlement)")
    print("-" * 78)
    print()
    print("Note: yes_bid/yes_ask in settled_markets.json is a SNAPSHOT at or near")
    print("settlement — not necessarily representative of intraday spreads.")
    print()
    
    # Analyze spread for various price ranges
    price_ranges = [
        ("$0.60-$0.70", 0.60, 0.70),
        ("$0.70-$0.80", 0.70, 0.80),
        ("$0.80-$0.90", 0.80, 0.90),
        ("$0.85-$0.95", 0.85, 0.95),
    ]
    
    for label, lo, hi in price_ranges:
        subset = [m for m in markets if lo <= m["last_price"] < hi and m["yes_bid"] > 0 and m["yes_ask"] > 0]
        if not subset:
            print(f"  {label}: No bid-ask data available")
            continue
        
        spreads = [m["yes_ask"] - m["yes_bid"] for m in subset]
        avg_spread = sum(spreads) / len(spreads)
        print(f"  {label} (n={len(subset)}): avg spread ${avg_spread:.3f}, "
              f"avg bid ${sum(m['yes_bid'] for m in subset)/len(subset):.3f}, "
              f"avg ask ${sum(m['yes_ask'] for m in subset)/len(subset):.3f}")
    
    # Look at ALL markets for spread analysis
    all_with_spread = [m for m in markets if m["yes_bid"] > 0 and m["yes_ask"] > 0]
    if all_with_spread:
        spreads = [m["yes_ask"] - m["yes_bid"] for m in all_with_spread]
        print(f"\n  Overall (n={len(all_with_spread)}): avg spread ${sum(spreads)/len(spreads):.3f}")
    else:
        print("\n  No bid-ask spread data available in settled_markets.json")
        print("  (yes_bid_dollars and yes_ask_dollars may be zeros or absent)")
    
    # ================================================================
    # 4. Volume per event for the winning bucket
    # ================================================================
    print()
    print("-" * 78)
    print("4. VOLUME ON THE WINNING BUCKET PER EVENT")
    print("-" * 78)
    print()
    
    events: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        events[m["event_ticker"]].append(m)
    
    winner_volumes = []
    winner_oi = []
    for event, buckets in events.items():
        winners = [b for b in buckets if b["won"]]
        if winners:
            winner_volumes.append(winners[0]["volume"])
            winner_oi.append(winners[0]["open_interest"])
    
    if winner_volumes:
        sorted_wv = sorted(winner_volumes)
        print(f"Volume on the winning bucket (n={len(winner_volumes)} events):")
        print(f"  Mean:   {sum(winner_volumes)/len(winner_volumes):.0f} contracts")
        print(f"  Median: {sorted_wv[len(sorted_wv)//2]:.0f} contracts")
        print(f"  P10:    {sorted_wv[int(len(sorted_wv)*0.1)]:.0f} contracts")
        print(f"  P25:    {sorted_wv[int(len(sorted_wv)*0.25)]:.0f} contracts")
        print(f"  P75:    {sorted_wv[int(len(sorted_wv)*0.75)]:.0f} contracts")
        print(f"  P90:    {sorted_wv[int(len(sorted_wv)*0.90)]:.0f} contracts")
    
    # ================================================================
    # 5. Contract size analysis — how much can you deploy?
    # ================================================================
    print()
    print("-" * 78)
    print("5. HOW MUCH CAPITAL CAN YOU DEPLOY?")
    print("-" * 78)
    print()
    
    # If median volume at winning bucket is X, you can probably buy 5-10% 
    # of volume without moving the market
    if winner_volumes:
        median_vol = sorted_wv[len(sorted_wv)//2]
        print(f"Winning bucket median volume: {median_vol:.0f} contracts")
        print()
        print("Rule of thumb: you can buy 5-10% of daily volume without")
        print("significantly moving the price.")
        print()
        for pct in [0.01, 0.05, 0.10]:
            contracts = int(median_vol * pct)
            capital = contracts * 0.85
            if contracts > 0:
                print(f"  {pct:.0%} of volume = {contracts} contracts = ${capital:.2f} deployed at $0.85")
                print(f"    Expected profit (95.1% WR): ${contracts * 0.1012:.2f}")
    
    # ================================================================
    # 6. Kalshi market structure
    # ================================================================
    print()
    print("-" * 78)
    print("6. KALSHI MARKET STRUCTURE CONSIDERATIONS")
    print("-" * 78)
    print()
    print("Key facts about Kalshi temperature markets:")
    print()
    print("  • Contracts settle at $1.00 (YES) or $0.00 (NO)")
    print("  • Minimum trade: 1 contract")
    print("  • Tick size: $0.01")
    print("  • No maker fees; taker fee is typically $0.02-$0.04 per contract")
    print("  • Austin HIGHAUS markets have 2°F bucket widths")
    print("  • Multiple buckets per event (5-7 typically)")
    print("  • Markets open ~48h before settlement")
    print()
    print("  CRITICAL: The $0.85 strategy's 10¢ edge must survive fees.")
    print()
    
    # Fee impact
    fees = [0.00, 0.02, 0.03, 0.04]
    ev_raw = 0.1012  # from profitability analysis
    print(f"  Raw EV per contract: ${ev_raw:.4f}")
    for fee in fees:
        ev_net = ev_raw - fee
        print(f"  After ${fee:.2f} fee: ${ev_net:.4f} ({ev_net/0.85:+.1%} ROI) {'✓ STILL PROFITABLE' if ev_net > 0 else '✗ WIPED OUT'}")
    
    # ================================================================
    # 7. Practical strategy sizing
    # ================================================================
    print()
    print("-" * 78)
    print("7. PRACTICAL STRATEGY SIZING")
    print("-" * 78)
    print()
    
    # Assume you want to risk no more than 5% of bankroll per trade
    for bankroll in [100, 500, 1000, 5000]:
        max_risk = bankroll * 0.05
        contracts_at_85 = int(max_risk / 0.85)
        ev_per_trade = contracts_at_85 * (0.1012 - 0.03)  # subtract typical fee
        
        # Number of trades: based on how often $0.85 appears
        if all_events:
            freq = len(events_at_85) / len(all_events) if all_events else 0
            trades_per_month = freq * 30  # rough
        else:
            trades_per_month = 3  # guess
        
        monthly_ev = ev_per_trade * trades_per_month
        
        print(f"  Bankroll: ${bankroll}")
        print(f"    Max 5% risk per trade: ${max_risk:.2f}")
        print(f"    Contracts at $0.85: {contracts_at_85}")
        print(f"    Net EV per trade (after $0.03 fee): ${ev_per_trade:.2f}")
        print(f"    Est. events with $0.85 bucket: ~{trades_per_month:.0f}/month")
        print(f"    Est. monthly EV: ${monthly_ev:.2f}")
        print()
    
    # ================================================================
    # VERDICT
    # ================================================================
    print("=" * 78)
    print("VERDICT: CAN YOU REGULARLY BUY AT $0.85?")
    print("=" * 78)
    print()
    
    if events_at_85 and all_events:
        freq = len(events_at_85) / len(all_events)
        print(f"Opportunity frequency: {len(events_at_85)}/{len(all_events)} events")
        print(f"  = {freq:.1%} of event dates have a bucket touching $0.85")
        print(f"  = roughly {freq * 30:.0f} opportunities per month")
        print()
    
    print("Volume constraints:")
    if winner_volumes:
        med = sorted(winner_volumes)[len(winner_volumes)//2]
        print(f"  Median winning bucket volume: {med:.0f} contracts")
        safe = int(med * 0.05)
        print(f"  Safe position size (5% of volume): {safe} contracts = ${safe * 0.85:.2f}")
        print()
    
    print("Fee impact:")
    print(f"  Raw edge: ~10¢/contract")
    print(f"  After $0.03 fee: ~7¢/contract — STILL STRONGLY POSITIVE")
    print()
    print("Liquidity risk:")
    print("  • Temperature markets are niche — thinner than politics/macro")
    print("  • The $0.85 price level requires the market to be fairly certain")
    print("    about the outcome — this typically happens later in the day")
    print("  • You may need to place limit orders and wait for fills")
    print("  • Slippage: buying at $0.86 or $0.87 instead of $0.85 shrinks your edge")
    print("    ($0.87 with 95.1% WR → EV = $0.0812 = 9.3% ROI, still profitable)")
    print("=" * 78)


if __name__ == "__main__":
    run_analysis()
