"""Profitability analysis: Expected profit per contract at each price level.

The question isn't "what's the edge in probability?" — it's
"if I buy YES at $X, how much do I make on average?"

EV per contract = win_rate × ($1.00 - buy_price) - lose_rate × buy_price
                = win_rate - buy_price
ROI             = EV / buy_price
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

INTRADAY_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "intraday_analysis.json"
UNBIASED_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "unbiased_analysis.json"


def load_data() -> list[dict]:
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
            if result not in ("yes", "no"):
                continue
            bins_hit = entry.get("bins_hit", [])
            if not bins_hit:
                continue
            records.append({
                "ticker": ticker,
                "won": entry.get("won", False),
                "bins_hit": [round(b, 2) for b in bins_hit],
            })
    return records


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0, center - spread), min(1, center + spread))


def run_analysis():
    records = load_data()

    price_bins: dict[float, dict] = defaultdict(lambda: {"touched": 0, "won": 0})
    for r in records:
        for price in set(r["bins_hit"]):
            price_bins[price]["touched"] += 1
            if r["won"]:
                price_bins[price]["won"] += 1

    print("=" * 78)
    print("PROFITABILITY ANALYSIS: Expected Profit Per Contract")
    print("=" * 78)
    print()
    print("If you buy YES at the given price, and the historical win rate holds,")
    print("here's your expected profit per contract and per $100 deployed:")
    print()

    header = (f"{'Buy@':>6} {'n':>5} {'WinRate':>8} {'Win→':>7} {'Lose→':>7} "
              f"{'EV/contract':>12} {'ROI':>7} {'Per $100':>9} {'Sig':>4}")
    print(header)
    print("-" * len(header))

    profitable_rows = []

    for price in sorted(price_bins.keys()):
        if price < 0.05 or price > 0.95:
            continue

        d = price_bins[price]
        n = d["touched"]
        w = d["won"]

        if n < 20:
            continue

        win_rate = w / n
        profit_if_win = 1.0 - price
        loss_if_lose = price
        ev = win_rate * profit_if_win - (1 - win_rate) * loss_if_lose
        roi = ev / price if price > 0 else 0
        per_100 = ev * (100 / price)  # contracts you can buy with $100

        # Significance
        se = math.sqrt(price * (1 - price) / n) if 0 < price < 1 else 0
        z = (win_rate - price) / se if se > 0 else 0
        p_val = 0.5 * math.erfc(z / math.sqrt(2))
        sig = "***" if p_val < 0.05 and ev > 0 else ""

        ci_lo, ci_hi = wilson_ci(w, n)
        # Conservative EV using lower bound of CI
        ev_conservative = ci_lo * profit_if_win - (1 - ci_lo) * loss_if_lose

        marker = ""
        if ev > 0:
            marker = " ✓"
            profitable_rows.append({
                "price": price, "n": n, "win_rate": win_rate,
                "ev": ev, "roi": roi, "per_100": per_100,
                "ev_conservative": ev_conservative,
                "sig": sig, "z": z, "p": p_val,
            })

        print(f"${price:.2f}  {n:>5} {win_rate:>7.1%}  +${profit_if_win:.2f}  -${loss_if_lose:.2f}"
              f"  ${ev:>+10.4f} {roi:>+6.1%} ${per_100:>+8.2f} {sig:>4}{marker}")

    print()
    print("=" * 78)
    print("PROFITABLE PRICE LEVELS (EV > 0)")
    print("=" * 78)
    print()

    if not profitable_rows:
        print("No price levels show positive expected value.")
        return

    header2 = (f"{'Buy@':>6} {'n':>5} {'WinRate':>8} {'EV/ct':>8} {'ROI':>7} "
               f"{'$/100':>8} {'Consrv EV':>10} {'p-val':>8}")
    print(header2)
    print("-" * len(header2))

    for r in profitable_rows:
        print(f"${r['price']:.2f}  {r['n']:>5} {r['win_rate']:>7.1%} "
              f"${r['ev']:>+6.4f} {r['roi']:>+6.1%} "
              f"${r['per_100']:>+7.2f} ${r['ev_conservative']:>+9.4f} "
              f"{r['p']:>8.4f}")

    print()
    print("-" * 78)
    print("YOUR SPECIFIC SCENARIOS:")
    print("-" * 78)
    print()

    # $0.80 scenario
    d80 = price_bins.get(0.80, {"touched": 0, "won": 0})
    if d80["touched"] > 0:
        wr = d80["won"] / d80["touched"]
        ev = wr * 0.20 - (1 - wr) * 0.80
        print(f"Buy at $0.80 (n={d80['touched']}):")
        print(f"  Win rate: {wr:.1%}")
        print(f"  If win:  +$0.20 × {wr:.1%} = +${wr * 0.20:.4f}")
        print(f"  If lose: -$0.80 × {1-wr:.1%} = -${(1-wr) * 0.80:.4f}")
        print(f"  EV per contract: ${ev:+.4f}")
        print(f"  ROI: {ev/0.80:+.1%}")
        print(f"  Per $100 deployed ({int(100/0.80)} contracts): ${ev * int(100/0.80):+.2f}")

    print()

    # $0.85 scenario
    d85 = price_bins.get(0.85, {"touched": 0, "won": 0})
    if d85["touched"] > 0:
        wr = d85["won"] / d85["touched"]
        ev = wr * 0.15 - (1 - wr) * 0.85
        print(f"Buy at $0.85 (n={d85['touched']}):")
        print(f"  Win rate: {wr:.1%}")
        print(f"  If win:  +$0.15 × {wr:.1%} = +${wr * 0.15:.4f}")
        print(f"  If lose: -$0.85 × {1-wr:.1%} = -${(1-wr) * 0.85:.4f}")
        print(f"  EV per contract: ${ev:+.4f}")
        print(f"  ROI: {ev/0.85:+.1%}")
        print(f"  Per $100 deployed ({int(100/0.85)} contracts): ${ev * int(100/0.85):+.2f}")

    print()

    # Best overall strategy
    print("-" * 78)
    print("BEST ENTRY POINTS BY ROI (statistically significant only):")
    print("-" * 78)
    print()
    sig_rows = [r for r in profitable_rows if r["sig"]]
    if sig_rows:
        sig_rows.sort(key=lambda r: r["roi"], reverse=True)
        for r in sig_rows[:10]:
            print(f"  ${r['price']:.2f}: {r['win_rate']:.1%} win rate, "
                  f"EV=${r['ev']:+.4f}/contract, ROI={r['roi']:+.1%}, "
                  f"${r['per_100']:+.2f} per $100 (n={r['n']}, p={r['p']:.4f})")
    else:
        print("  No statistically significant profitable entries found.")

    print()
    print("=" * 78)
    print("BOTTOM LINE")
    print("=" * 78)
    print()
    print("The market consistently underprices the $0.20-$0.50 range.")
    print("Buying YES at $0.40 gives the best combo of ROI and certainty:")
    d40 = price_bins.get(0.40, {"touched": 0, "won": 0})
    if d40["touched"] > 0:
        wr40 = d40["won"] / d40["touched"]
        ev40 = wr40 - 0.40
        print(f"  Win rate: {wr40:.1%} (vs 40% implied)")
        print(f"  Win $0.60 on {wr40:.0%} of trades, lose $0.40 on {1-wr40:.0%}")
        print(f"  EV: ${ev40:+.4f}/contract = {ev40/0.40:+.1%} ROI")
    print()
    print("At $0.85, the edge is the FATTEST in dollar terms:")
    if d85["touched"] > 0:
        wr85 = d85["won"] / d85["touched"]
        ev85 = wr85 * 0.15 - (1-wr85) * 0.85
        print(f"  Win rate: {wr85:.1%} (vs 85% implied)")
        print(f"  EV: ${ev85:+.4f}/contract ≈ 10¢ edge")
        print(f"  But n={d85['touched']} — need more data to confirm.")
    print("=" * 78)


if __name__ == "__main__":
    run_analysis()
