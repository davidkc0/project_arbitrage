"""Hypothesis 3: Intraday Temperature Progression Edge

Tests: "If I use hourly METAR data to predict the daily high at hour H,
and that prediction disagrees with Kalshi's market price, can I profit?"

Joins:
  - kaus_hourly_history.csv (5yr METAR, hourly temp/sky/wind)
  - kaus_daily_summary.csv (daily highs)
  - settled_markets.json (Kalshi outcomes with bucket structures)

Design:
  For each settled event date that overlaps with METAR data:
    For each hour (10, 11, 12, 13, 14, 15):
      1. Compute remaining-rise distribution from METAR historical
      2. Map distribution onto Kalshi bucket structure
      3. Compare model probability vs. market price for each bucket
      4. Simulate: "If model_prob > market_price + threshold, buy YES"
      5. Track PnL

Success metric: ROI > 0 with n >= 50 simulated trades and p < 0.05
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HOURLY_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kaus_hourly_history.csv"
DAILY_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kaus_daily_summary.csv"
SETTLED_FILE = Path(__file__).resolve().parent.parent / "weather_bets" / "data" / "kalshi_history" / "settled_markets.json"
RESULTS_FILE = Path(__file__).resolve().parent / "results" / "h3_results.json"


def load_hourly() -> dict[str, dict[int, int]]:
    """Load hourly METAR and return {date: {hour: temp_f}}."""
    result: dict[str, dict[int, int]] = defaultdict(dict)
    with open(HOURLY_FILE) as f:
        for row in csv.DictReader(f):
            try:
                date = row["date"]
                hour = int(row["hour"])
                temp = int(float(row["temp_f"]))
                if hour not in result[date]:
                    result[date][hour] = temp
            except (ValueError, KeyError):
                continue
    return dict(result)


def load_daily() -> dict[str, dict]:
    """Load daily summaries: {date: {daily_high, sky_dominant, month}}."""
    result = {}
    with open(DAILY_FILE) as f:
        for row in csv.DictReader(f):
            try:
                result[row["date"]] = {
                    "daily_high": int(float(row["daily_high"])),
                    "sky_dominant": row.get("sky_dominant", "CLR"),
                    "month": int(row["month"]),
                }
            except (ValueError, KeyError):
                continue
    return result


def load_settled_events() -> dict[str, dict]:
    """Load settled markets, group by date. 
    Returns {date_str: {actual_high, buckets: [{floor, cap, result, last_price, volume}]}}.
    """
    with open(SETTLED_FILE) as f:
        raw = json.load(f)
    
    events: dict[str, dict] = {}
    
    for m in raw:
        ticker = m.get("ticker", "")
        status = m.get("status", "")
        result = m.get("result", "")
        
        if status != "finalized" or "HIGHAUS" not in ticker or result not in ("yes", "no"):
            continue
        
        event_ticker = m.get("event_ticker", "")
        
        # Parse date from event_ticker
        parts = event_ticker.split("-")
        if len(parts) < 2:
            continue
        date_part = parts[-1]
        
        try:
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
            else:
                continue
        except (ValueError, IndexError):
            continue
        
        try:
            last_price = float(m.get("last_price_dollars", "0"))
            volume = float(m.get("volume_fp", "0"))
        except (ValueError, TypeError):
            continue
        
        if date_str not in events:
            ev = m.get("expiration_value")
            actual = float(ev) if ev else None
            events[date_str] = {"actual_high": actual, "buckets": []}
        
        bucket = {
            "ticker": ticker,
            "result": result,
            "won": result == "yes",
            "last_price": last_price,
            "volume": volume,
            "strike_type": m.get("strike_type", ""),
            "floor_strike": m.get("floor_strike"),
            "cap_strike": m.get("cap_strike"),
        }
        events[date_str]["buckets"].append(bucket)
    
    return events


def build_remaining_rise_model(hourly: dict, daily: dict) -> dict[int, dict[int, list[float]]]:
    """Build remaining-rise distribution by (month, hour) from all historical data.
    
    Returns {month: {hour: [list of remaining_rise values]}}.
    Only uses days where an observation exists at the target hour.
    """
    model: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    
    for date_str, day_info in daily.items():
        high = day_info["daily_high"]
        month = day_info["month"]
        
        if date_str not in hourly:
            continue
        
        hour_temps = hourly[date_str]
        
        for hour in (10, 11, 12, 13, 14, 15):
            if hour in hour_temps:
                rise = high - hour_temps[hour]
                if rise >= 0:
                    model[month][hour].append(rise)
    
    return model


def gaussian_bucket_prob(mean: float, std: float, low: float | None, high: float | None) -> float:
    """Compute probability that a Gaussian(mean, std) falls within [low, high].
    
    For 'between' type: P(low <= X < high)
    For 'less' type: P(X < high) (low is None)
    For 'greater' type: P(X >= low) (high is None)
    """
    def phi(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    if std <= 0:
        std = 0.5  # avoid division by zero
    
    if low is not None and high is not None:
        return phi((high - mean) / std) - phi((low - mean) / std)
    elif high is not None:  # less than
        return phi((high - mean) / std)
    elif low is not None:  # greater than
        return 1 - phi((low - mean) / std)
    return 0.0


def run_analysis():
    print("=" * 70)
    print("HYPOTHESIS 3: INTRADAY TEMPERATURE PROGRESSION EDGE")
    print("=" * 70)
    print()
    
    hourly = load_hourly()
    daily = load_daily()
    events = load_settled_events()
    
    print(f"METAR hourly data: {len(hourly)} dates")
    print(f"Daily summaries: {len(daily)} dates")
    print(f"Settled Kalshi events: {len(events)} dates")
    
    # Find overlap
    overlap_dates = set(hourly.keys()) & set(daily.keys()) & set(events.keys())
    print(f"Overlapping dates (METAR + daily + Kalshi): {len(overlap_dates)}")
    print()
    
    # Build the remaining-rise model from ALL historical METAR (not just overlap)
    model = build_remaining_rise_model(hourly, daily)
    
    # Simulate trading at each hour with different edge thresholds
    thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    hours = [10, 11, 12, 13, 14, 15]
    
    # Results structure: {threshold: {hour: {trades, wins, pnl, cost}}}
    results: dict[float, dict[int, dict]] = {
        t: {h: {"trades": 0, "wins": 0, "pnl": 0.0, "cost": 0.0} for h in hours}
        for t in thresholds
    }
    
    for date_str in sorted(overlap_dates):
        event = events[date_str]
        actual_high = event["actual_high"]
        if actual_high is None:
            continue
        
        day_info = daily[date_str]
        month = day_info["month"]
        hour_temps = hourly[date_str]
        
        for hour in hours:
            if hour not in hour_temps:
                continue
            
            current_temp = hour_temps[hour]
            
            # Get remaining-rise distribution for this (month, hour)
            rises = model.get(month, {}).get(hour, [])
            if len(rises) < 20:
                continue
            
            # Predict final high: current_temp + remaining_rise
            predicted_highs = [current_temp + r for r in rises]
            pred_mean = sum(predicted_highs) / len(predicted_highs)
            pred_std = math.sqrt(sum((h - pred_mean)**2 for h in predicted_highs) / max(len(predicted_highs) - 1, 1))
            
            # For each bucket in this event, compute model probability
            for bucket in event["buckets"]:
                strike_type = bucket["strike_type"]
                floor = bucket.get("floor_strike")
                cap = bucket.get("cap_strike")
                market_price = bucket["last_price"]
                won = bucket["won"]
                
                # Skip very low-volume buckets (likely untradeable)
                if bucket["volume"] < 100:
                    continue
                
                # Compute model probability
                if strike_type == "between" and floor is not None and cap is not None:
                    model_prob = gaussian_bucket_prob(pred_mean, pred_std, floor, cap + 1)
                elif strike_type == "less" and cap is not None:
                    model_prob = gaussian_bucket_prob(pred_mean, pred_std, None, cap)
                elif strike_type == "greater" and floor is not None:
                    model_prob = gaussian_bucket_prob(pred_mean, pred_std, floor + 1, None)
                else:
                    continue
                
                # For each threshold, decide whether to trade
                for threshold in thresholds:
                    edge = model_prob - market_price
                    if edge > threshold and market_price < 0.95:  # Don't buy near-certain contracts
                        r = results[threshold][hour]
                        r["trades"] += 1
                        r["cost"] += market_price
                        if won:
                            r["wins"] += 1
                            r["pnl"] += (1.0 - market_price)  # profit
                        else:
                            r["pnl"] -= market_price  # loss
    
    # Print results
    print("-" * 70)
    print("SIMULATED TRADING RESULTS")
    print("-" * 70)
    print()
    print("Edge threshold = model_prob - market_price must exceed this to trade.")
    print("Volume filter: only buckets with >= 100 contracts traded.")
    print("Price filter: skip market_price >= $0.95.")
    print()
    
    all_results = []
    
    for threshold in thresholds:
        print(f"\n  THRESHOLD: model_prob > market_price + {threshold:.0%}")
        print(f"  {'Hour':>6} {'Trades':>7} {'Wins':>6} {'WinRate':>8} {'TotalPnL':>10} {'AvgCost':>8} {'ROI':>8}")
        print(f"  {'-'*55}")
        
        total_trades = 0
        total_wins = 0
        total_pnl = 0.0
        total_cost = 0.0
        
        for hour in hours:
            r = results[threshold][hour]
            n = r["trades"]
            w = r["wins"]
            wr = w / n if n > 0 else 0
            pnl = r["pnl"]
            cost = r["cost"]
            avg_cost = cost / n if n > 0 else 0
            roi = pnl / cost if cost > 0 else 0
            
            total_trades += n
            total_wins += w
            total_pnl += pnl
            total_cost += cost
            
            if n > 0:
                print(f"  {hour:>6} {n:>7} {w:>6} {wr:>7.1%} ${pnl:>+9.2f} ${avg_cost:>7.3f} {roi:>+7.1%}")
        
        total_wr = total_wins / total_trades if total_trades > 0 else 0
        total_roi = total_pnl / total_cost if total_cost > 0 else 0
        print(f"  {'TOTAL':>6} {total_trades:>7} {total_wins:>6} {total_wr:>7.1%} ${total_pnl:>+9.2f} {'':>8} {total_roi:>+7.1%}")
        
        all_results.append({
            "threshold": threshold,
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_win_rate": round(total_wr, 4),
            "total_pnl": round(total_pnl, 2),
            "total_cost": round(total_cost, 2),
            "total_roi": round(total_roi, 4),
        })
    
    # Verdict
    print()
    print("=" * 70)
    
    best = max(all_results, key=lambda r: r["total_roi"] if r["total_trades"] >= 50 else -999)
    
    if best["total_roi"] > 0 and best["total_trades"] >= 50:
        # Simple significance test
        n = best["total_trades"]
        wr = best["total_win_rate"]
        avg_price = best["total_cost"] / n if n > 0 else 0
        se = math.sqrt(wr * (1 - wr) / n) if n > 0 else 0
        z = (wr - avg_price) / se if se > 0 else 0
        p = 0.5 * math.erfc(z / math.sqrt(2))
        
        print(f"VERDICT: POTENTIAL INTRADAY EDGE at threshold={best['threshold']:.0%}")
        print(f"  Trades: {best['total_trades']}, Win rate: {best['total_win_rate']:.1%}, ROI: {best['total_roi']:+.1%}")
        print(f"  PnL: ${best['total_pnl']:+.2f}, z={z:.2f}, p={p:.4f}")
        if p < 0.05:
            print("  => STATISTICALLY SIGNIFICANT (p < 0.05)")
        else:
            print("  => NOT statistically significant (p >= 0.05)")
    else:
        print("VERDICT: NO SIGNIFICANT INTRADAY EDGE FOUND")
        print("  The intraday temperature progression model does NOT produce")
        print("  profitable trades against Kalshi last-traded prices.")
    print("=" * 70)
    
    # Save
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "overlap_dates": len(overlap_dates),
            "results_by_threshold": all_results,
        }, f, indent=2)
    print(f"\nFull results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_analysis()
