"""Main entry point for the weather betting system."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path

from weather_bets import config
from weather_bets.edge_calculator import calculate_edges, calculate_spreads
from weather_bets.executor import WeatherExecutor
from weather_bets.historical import fetch_historical_temps, analyze_forecast_bias
from weather_bets.kalshi_weather import fetch_weather_markets
from weather_bets.llm_analyst import analyze_spread
from weather_bets.models import BetDecision, ForecastData
from weather_bets.nws_forecast import fetch_forecast, fetch_hourly_forecast
from weather_bets.open_meteo import fetch_multi_model_forecast, build_consensus
from weather_bets.trade_log import (
    load_trades, save_trade, save_scan_result, get_trade_summary,
    get_open_tickers,
)

logger = logging.getLogger(__name__)

# ── Global State ────────────────────────────────────────────────────────
scan_state = {
    "last_scan": None,
    "forecasts": [],
    "models": [],
    "consensus": {},
    "buckets": {},
    "opportunities": [],
    "spreads": [],
    "recommendations": [],
    "placed_bets": load_trades(),  # Load from disk on startup
    "historical": [],
    "bias": {},
    "scan_log": [],
    "trade_summary": get_trade_summary(),
}

executor = WeatherExecutor()


async def run_scan_cycle():
    """Run one full scan cycle for all active cities."""
    logger.info("=" * 60)
    logger.info("[Scan] Starting weather betting scan cycle")
    logger.info("=" * 60)

    scan_state["last_scan"] = datetime.now(timezone.utc).isoformat()
    scan_state["opportunities"] = []
    scan_state["spreads"] = []
    scan_state["recommendations"] = []
    scan_state["scan_log"] = []

    def log(msg: str):
        scan_state["scan_log"].append(msg)
        logger.info(msg)

    for city_code in config.ACTIVE_CITIES:
        city = config.CITIES.get(city_code)
        if not city:
            continue

        log(f"\n📍 Scanning {city.name}...")

        # Step 1: Fetch historical observations for bias analysis
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_observed_high = None
        try:
            historical = await fetch_historical_temps(city)
            scan_state["historical"] = [
                {"date": h["date"], "high": h["tmax"], "low": h["tmin"]}
                for h in historical
            ]
            # Extract today's current observed high as a floor for same-day bets
            for h in historical:
                if h["date"] == today_str:
                    current_observed_high = h["tmax"]
                    log(f"📡 Current observed high today: {current_observed_high}°F")
                    break
        except Exception as e:
            log(f"⚠️ Historical data error: {e}")
            historical = []

        # Step 2: Fetch NWS forecast
        try:
            forecasts = await fetch_forecast(city)
            scan_state["forecasts"] = [
                {"city": f.city, "date": f.date, "high": f.high_temp_f,
                 "detail": f.short_forecast, "source": "NWS"}
                for f in forecasts
            ]
        except Exception as e:
            log(f"❌ NWS forecast error: {e}")
            continue

        # Step 3: Fetch multi-model forecasts (Open-Meteo)
        model_forecasts = []
        try:
            model_forecasts = await fetch_multi_model_forecast(city)
            scan_state["models"] = model_forecasts
        except Exception as e:
            log(f"⚠️ Open-Meteo error: {e}")

        # Step 4: Compare forecast vs recent actuals for bias
        if historical and forecasts:
            fc_dicts = [{"date": f.date, "high": f.high_temp_f} for f in forecasts]
            bias = analyze_forecast_bias(historical, fc_dicts)
            scan_state["bias"] = bias
            if bias["sample_size"] > 0:
                log(f"📊 NWS bias: {bias['mean_error']:+.1f}°F avg error, "
                    f"MAE={bias['abs_error']:.1f}°F (n={bias['sample_size']})")

        # Step 5: For each upcoming day, build consensus and find spreads
        for i, fc in enumerate(forecasts[:3]):
            # Build consensus from NWS + Open-Meteo models
            consensus = build_consensus(fc.high_temp_f, model_forecasts, fc.date)
            scan_state["consensus"][fc.date] = consensus

            consensus_high = consensus["consensus_high"]
            nws_high = consensus["nws_high"]
            alt_avg = consensus.get("alt_avg")
            spread_deg = consensus["spread"]

            if alt_avg:
                log(f"\n📅 {fc.date} — NWS: {nws_high}°F | Models avg: {alt_avg}°F "
                    f"| Consensus: {consensus_high}°F (spread: {spread_deg}°F) "
                    f"({fc.short_forecast})")
            else:
                log(f"\n📅 {fc.date} — NWS: {nws_high}°F ({fc.short_forecast})")

            # Use consensus temp for edge/spread calculations
            consensus_fc = ForecastData(
                city=fc.city, date=fc.date,
                high_temp_f=round(consensus_high),
                short_forecast=fc.short_forecast,
                detailed_forecast=fc.detailed_forecast,
            )

            try:
                buckets = await fetch_weather_markets(city, fc.date)
                scan_state["buckets"][f"{city_code}-{fc.date}"] = [
                    {"label": b.label, "yes": b.yes_price, "no": b.no_price,
                     "ticker": b.ticker}
                    for b in buckets
                ]
            except Exception as e:
                log(f"❌ Kalshi markets error for {fc.date}: {e}")
                continue

            if not buckets:
                log(f"  No Kalshi markets found for {fc.date}")
                continue

            days_ahead = i + 1

            # Fetch hourly forecast early so we can use it for same-day calcs
            hourly = None
            try:
                hourly = await fetch_hourly_forecast(city, fc.date)
            except Exception:
                pass

            # Calculate per-bucket edges using CONSENSUS temp
            # Pass current observed high + hourly for same-day bets
            obs_high = current_observed_high if days_ahead == 1 else None
            edges = calculate_edges(consensus_fc, buckets, days_ahead, obs_high, hourly)
            scan_state["opportunities"].extend([
                {
                    "city": o.city, "date": o.date, "bucket": o.bucket.label,
                    "ticker": o.bucket.ticker,
                    "forecast": o.forecast_high,
                    "consensus": round(consensus_high, 1),
                    "our_prob": round(o.our_probability * 100, 1),
                    "market_prob": round(o.market_probability * 100, 1),
                    "edge": round(o.edge_percent, 1),
                    "ev": round(o.expected_value, 3),
                    "yes_price": o.bucket.yes_price,
                }
                for o in edges
            ])

            # Build spread bets using CONSENSUS temp
            spreads = calculate_spreads(consensus_fc, buckets, days_ahead, obs_high, hourly)

            for sp in spreads:
                labels = " + ".join(b.label for b in sp.buckets)
                scan_state["spreads"].append({
                    "city": sp.city, "date": sp.date,
                    "buckets": labels,
                    "forecast": sp.forecast_high,
                    "probability": round(sp.total_probability * 100, 1),
                    "cost": round(sp.total_cost, 2),
                    "profit": round(sp.profit_if_hit, 2),
                    "ev": round(sp.expected_profit, 3),
                    "roi": round(sp.roi_percent, 1),
                    "n_buckets": len(sp.buckets),
                })

            bucket_summary = ""
            for b in buckets:
                bucket_summary += f"  {b.label:15} YES=${b.yes_price:.2f} NO=${b.no_price:.2f}\n"

            # Find the best spread with positive EV and send to Claude
            best_spreads = [s for s in spreads if s.expected_profit > 0]
            if not best_spreads:
                log(f"  No positive-EV spreads found")
                continue

            # Analyze top 2 spreads
            for sp in best_spreads[:2]:
                labels = " + ".join(b.label for b in sp.buckets)
                log(f"\n🤖 Asking Claude about spread [{labels}] "
                    f"(prob={sp.total_probability:.0%}, cost=${sp.total_cost:.2f}, "
                    f"EV=${sp.expected_profit:+.3f})...")

                try:
                    rec = await analyze_spread(sp, bucket_summary, hourly)
                except Exception as e:
                    log(f"  ❌ LLM error: {e}")
                    continue

                labels = " + ".join(b.label for b in sp.buckets)
                scan_state["recommendations"].append({
                    "spread": labels,
                    "decision": rec.decision.value,
                    "confidence": round(rec.confidence * 100),
                    "bet_size": rec.bet_size_usd,
                    "probability": round(sp.total_probability * 100, 1),
                    "cost": round(sp.total_cost, 2),
                    "ev": round(sp.expected_profit, 3),
                    "reasoning": rec.reasoning,
                })

                if rec.decision == BetDecision.BET:
                    log(f"  ✅ Claude says BET: {rec.reasoning}")
                    # Check for duplicate bets
                    open_tickers = get_open_tickers()
                    # Place individual orders for each leg
                    for j, (bucket, alloc) in enumerate(zip(sp.buckets, rec.allocations)):
                        if alloc <= 0:
                            continue
                        if bucket.ticker in open_tickers:
                            log(f"  ⚠️ Already have open bet on {bucket.ticker} — skipping")
                            continue
                        from weather_bets.models import BetRecommendation, EdgeOpportunity
                        single_rec = BetRecommendation(
                            decision=BetDecision.BET,
                            opportunity=EdgeOpportunity(
                                city=sp.city, date=sp.date, bucket=bucket,
                                forecast_high=sp.forecast_high,
                                our_probability=sp.bucket_probabilities[j],
                                market_probability=bucket.yes_price,
                                edge=sp.bucket_probabilities[j] - bucket.yes_price,
                                edge_percent=(sp.bucket_probabilities[j] - bucket.yes_price) * 100,
                                expected_value=0, forecast_detail="",
                            ),
                            confidence=rec.confidence,
                            bet_size_usd=alloc,
                            reasoning=f"Spread leg: {rec.reasoning}",
                            side="yes",
                            ticker=bucket.ticker,
                        )
                        bet = await executor.place_bet(single_rec)
                        trade_record = {
                            "id": bet.id, "ticker": bet.ticker,
                            "city": bet.city, "date": bet.date,
                            "bucket": bet.bucket_label, "side": bet.side,
                            "price": bet.price, "qty": bet.quantity,
                            "cost": round(bet.cost_usd, 2),
                            "edge": round(bet.edge_percent, 1),
                            "our_prob": round(bet.our_probability * 100, 1),
                            "market_prob": round(bet.market_probability * 100, 1),
                            "spread": labels,
                            "reasoning": rec.reasoning,
                            "mode": config.EXECUTION_MODE,
                            "time": bet.placed_at.isoformat(),
                            "settled": False, "won": None, "pnl": None,
                        }
                        save_trade(trade_record)
                        scan_state["placed_bets"].append(trade_record)
                else:
                    log(f"  ⏭️  Claude says SKIP: {rec.reasoning}")

    log(f"\n{'=' * 60}")
    bets_this_scan = len(scan_state['placed_bets']) - len(load_trades()) + len(scan_state['placed_bets'])
    log(f"✅ Scan complete.")

    # Save scan summary
    save_scan_result({
        "forecasts": scan_state.get("forecasts", []),
        "consensus": scan_state.get("consensus", {}),
        "spreads_found": len(scan_state.get("spreads", [])),
        "recommendations": len(scan_state.get("recommendations", [])),
        "bets_placed": len([r for r in scan_state.get("recommendations", []) if r.get("decision") == "bet"]),
    })
    scan_state["trade_summary"] = get_trade_summary()


# ── Scan Loop ───────────────────────────────────────────────────────────

async def scan_loop():
    await asyncio.sleep(2)
    await executor.initialize()
    while True:
        try:
            await run_scan_cycle()
        except Exception as e:
            logger.error(f"[Scan] Error: {e}", exc_info=True)
        logger.info(f"[Scan] Next scan in {config.SCAN_INTERVAL}s")
        await asyncio.sleep(config.SCAN_INTERVAL)


# ── FastAPI Server ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scan_loop())
    yield
    task.cancel()
    await executor.close()

app = FastAPI(title="Weather Bets", lifespan=lifespan)
DASHBOARD_DIR = Path(__file__).parent / "dashboard"

@app.get("/")
async def index():
    return FileResponse(DASHBOARD_DIR / "index.html")

@app.get("/api/state")
async def get_state():
    return scan_state

@app.get("/api/rescan")
async def trigger_rescan():
    asyncio.create_task(run_scan_cycle())
    return {"status": "scan_triggered"}

@app.get("/{filename}")
async def serve_static(filename: str):
    filepath = DASHBOARD_DIR / filename
    if filepath.exists():
        return FileResponse(filepath)
    return HTMLResponse("Not found", status_code=404)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, log_level="info")

if __name__ == "__main__":
    main()
