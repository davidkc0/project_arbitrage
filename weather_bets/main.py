"""Main entry point for the weather betting system."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pathlib import Path

from weather_bets import config
from weather_bets.executor import WeatherExecutor
from weather_bets.historical import fetch_historical_temps, analyze_forecast_bias
from weather_bets.kalshi_weather import fetch_weather_markets
from weather_bets import bucket_cache
from weather_bets.nws_sniper import sniper_loop
from weather_bets.weather_summary import build_weather_summary
from weather_bets.claude_predictor import predict_high
from weather_bets.models import BetDecision, BetRecommendation, EdgeOpportunity, ForecastData
from weather_bets.nws_forecast import fetch_forecast, fetch_hourly_forecast
from weather_bets.open_meteo import fetch_multi_model_forecast, build_consensus
from weather_bets.forecast_tracker import save_forecast, record_actual
from weather_bets.accuracy_tracker import (
    log_forecast as log_city_forecast,
    record_actual as record_city_actual,
)
from weather_bets.price_watcher import PriceWatcher
from weather_bets.position_manager import PositionManager
from weather_bets.trade_log import (
    load_trades, save_trade, save_scan_result, get_trade_summary,
    get_open_tickers, already_bet_on,
)
from weather_bets.bet_engine import BetEngine
from weather_bets.synoptic_poller import SynopticPoller
from weather_bets.iem_poller import IEMPoller
from weather_bets.rounding_map import is_crossing_temp, get_bucket_label

logger = logging.getLogger(__name__)

# ── Global State ────────────────────────────────────────────────────────
scan_state = {
    "last_scan": None,
    "forecasts": [],
    "models": [],
    "consensus": {},
    "buckets": {},
    "opportunities": [],
    "placed_bets": load_trades(),  # Load from disk on startup
    "historical": [],
    "bias": {},
    "scan_log": [],
    "trade_summary": get_trade_summary(),
    "hourly_today": [],  # NWS hourly forecast for today [{"hour": N, "temp_f": X}]
}

executor = WeatherExecutor()
position_manager = PositionManager()
price_watcher: PriceWatcher | None = None  # initialized in lifespan

# ── Intraday System ─────────────────────────────────────────────────────
bet_engine = BetEngine(
    total_balance=35.91,
    execution_mode=config.EXECUTION_MODE,
)
synoptic = SynopticPoller()
iem = IEMPoller(station="AUS", network="TX_ASOS")

intraday_state = {
    "active": False,
    "last_poll": None,
    "latest_temp": None,
    "trajectory": {},
    "iem_high": None,
    "iem_source": "IEM_DSM",
    "today_decisions": [],
    "yes_bet_placed": False,
    "no_bets_placed": False,
    "favorite_bet_placed": False,
    "daily_summary": {},
}


async def run_scan_cycle():
    """Run one full scan cycle for all active cities."""
    logger.info("=" * 60)
    logger.info("[Scan] Starting weather betting scan cycle")
    logger.info("=" * 60)

    scan_state["last_scan"] = datetime.now(timezone.utc).isoformat()
    scan_state["opportunities"] = []
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

            # Record actuals for past dates so forecast_tracker can settle them
            for h in historical:
                obs_date = h["date"]
                if obs_date < today_str:  # only past dates have final actuals
                    try:
                        record_actual(obs_date, h["tmax"])
                        # Also record in city-aware accuracy tracker
                        record_city_actual(city_code, obs_date, h["tmax"])
                    except Exception as te:
                        logger.warning(f"[Tracker] record_actual error for {obs_date}: {te}")
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
            days_out = i + 1  # i=0 → 1d out, i=1 → 2d out, etc.

            # Build consensus from NWS + Open-Meteo models (city-aware weights when available)
            consensus = build_consensus(fc.high_temp_f, model_forecasts, fc.date, days_out=days_out, city_code=city_code)
            scan_state["consensus"][fc.date] = consensus

            # Save forecast for accuracy tracking
            try:
                model_name_map = {
                    "GFS (NOAA)": "gfs",
                    "ICON (German)": "icon",
                    "GEM (Canadian)": "gem",
                }
                models_dict = {}
                for mf in model_forecasts:
                    if mf.get("date") == fc.date:
                        src_key = model_name_map.get(mf.get("model", ""))
                        if src_key:
                            models_dict[src_key] = mf["high_f"]
                save_forecast(
                    target_date=fc.date,
                    forecast_date=today_str,
                    nws=fc.high_temp_f,
                    models_dict=models_dict,
                    days_out=days_out,
                )
                # Also log in city-aware accuracy tracker (consensus is already computed above)
                log_city_forecast(
                    city_code=city_code,
                    target_date=fc.date,
                    days_out=days_out,
                    nws=fc.high_temp_f,
                    gfs=models_dict.get("gfs"),
                    icon=models_dict.get("icon"),
                    gem=models_dict.get("gem"),
                    consensus=consensus.get("consensus_high"),
                )
            except Exception as te:
                logger.warning(f"[Tracker] save_forecast error for {fc.date}: {te}")

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

            days_ahead = days_out  # same value, keep variable name for compatibility

            # Fetch hourly forecast early so we can use it for same-day calcs
            hourly = None
            try:
                hourly = await fetch_hourly_forecast(city, fc.date)
            except Exception:
                pass

            # ── Weather Summary + Claude Prediction ──
            # Only do this for same-day (days_ahead==1) since that's when we trade
            if days_ahead == 1 and buckets:
                # Determine current hour in CT
                ct_now = datetime.now(timezone(timedelta(hours=-5)))
                current_hour_ct = ct_now.hour
                month = ct_now.month

                # Build model forecasts dict for weather summary
                model_temps = {}
                for mf in model_forecasts:
                    if mf.get("date") == fc.date:
                        model_temps[mf.get("model", "?")] = mf["high_f"]

                # Get yesterday's high from historical data
                yesterday_high = None
                for h in (scan_state.get("historical") or []):
                    if h["date"] < today_str:
                        yesterday_high = h["high"]

                # Get hourly data as list of dicts and store for intraday use
                hourly_dicts = []
                if hourly:
                    hourly_dicts = hourly  # already list of {"hour": N, "temp_f": X}
                    scan_state["hourly_today"] = hourly_dicts

                # Use Synoptic day_max (direct °F, no C→F rounding issues)
                # Fall back to IEM DSM (settlement source), then NWS obs as last resort
                synoptic_max = synoptic.get_trajectory().get("day_max") if synoptic else None
                iem_max = intraday_state.get("iem_high")
                best_observed_high = synoptic_max or iem_max or current_observed_high or intraday_state.get("latest_temp")

                # Build and print the weather summary
                summary_text = build_weather_summary(
                    city_name=city.name,
                    current_temp_f=best_observed_high,
                    day_max_f=best_observed_high,
                    current_hour_ct=current_hour_ct,
                    month=month,
                    nws_forecast_high=fc.high_temp_f,
                    model_forecasts=model_temps,
                    consensus_high=consensus_high,
                    hourly_forecast=hourly_dicts,
                    yesterday_high=yesterday_high,
                    sky_conditions=fc.short_forecast,
                    buckets=buckets,
                    iem_high=intraday_state.get("iem_high"),
                )

                # Ask Claude for a prediction
                log("\n🤖 Asking Claude for high temperature prediction...")
                prediction = await predict_high(summary_text)
                if prediction:
                    scan_state["claude_prediction"] = prediction
                    log(f"  🌡️  Claude predicts: {prediction.get('predicted_high_f')}°F "
                        f"({prediction.get('confidence_low_f')}-{prediction.get('confidence_high_f')}°F)")
                    log(f"  🏷️  Winning bucket: {prediction.get('winning_bucket')}")
                    log(f"  💡 Trade: {prediction.get('trade_suggestion')}")
                else:
                    log("  ⚠️ Claude prediction unavailable")

    log(f"\n{'=' * 60}")
    bets_this_scan = len(scan_state['placed_bets']) - len(load_trades()) + len(scan_state['placed_bets'])
    log(f"✅ Scan complete.")

    # Save scan summary
    save_scan_result({
        "forecasts": scan_state.get("forecasts", []),
        "consensus": scan_state.get("consensus", {}),
        "claude_prediction": scan_state.get("claude_prediction"),
        "bets_placed": len(scan_state.get("placed_bets", [])),
    })
    scan_state["trade_summary"] = get_trade_summary()


# ── Momentum signal handler ─────────────────────────────────────────────

async def handle_momentum_signal(ticker: str, signal: str) -> None:
    """
    React immediately to a price momentum signal from PriceWatcher.

    buy  → evaluate and place bet if we don't already hold it (50% budget rule applies)
    sell → close the position via PositionManager
    """
    if signal == "sell":
        logger.info(f"[Momentum] SELL signal for {ticker} — attempting to close position")
        result = await position_manager.sell_position(ticker, reason="momentum take-profit")
        logger.info(f"[Momentum] Sell result: {result}")
        if price_watcher:
            price_watcher.clear_signal(ticker)
        return

    if signal == "buy":
        if already_bet_on(ticker):
            logger.info(f"[Momentum] BUY signal for {ticker} — already holding, skipping")
            if price_watcher:
                price_watcher.clear_signal(ticker)
            return

        # Find the opportunity data from latest scan state
        opportunities = scan_state.get("opportunities", [])
        opp_data = next((o for o in opportunities if o.get("ticker") == ticker), None)
        if not opp_data:
            logger.info(f"[Momentum] BUY signal for {ticker} — no opportunity data from scan, skipping")
            if price_watcher:
                price_watcher.clear_signal(ticker)
            return

        # Respect budget rule
        remaining = await executor.get_betting_budget()
        if remaining <= 0:
            logger.warning(f"[Momentum] BUY signal for {ticker} — 50% budget exhausted, skipping")
            if price_watcher:
                price_watcher.clear_signal(ticker)
            return

        bet_size = min(config.BET_SIZE_USD, remaining)
        logger.info(
            f"[Momentum] BUY signal for {ticker} — our_prob={opp_data.get('our_prob')}% "
            f"market={opp_data.get('market_prob')}% "
            f"edge={opp_data.get('edge')}% — placing bet of ${bet_size:.2f}"
        )

        # Build a minimal BetRecommendation
        from weather_bets.models import TemperatureBucket
        bucket = TemperatureBucket(
            ticker=ticker,
            label=opp_data.get("bucket", "?"),
            low_bound=None,
            high_bound=None,
            yes_price=opp_data.get("yes_price", 0.5),
            no_price=1.0 - opp_data.get("yes_price", 0.5),
            volume=0,
            close_time="",
        )
        opp = EdgeOpportunity(
            city=opp_data.get("city", "AUS"),
            date=opp_data.get("date", ""),
            bucket=bucket,
            forecast_high=opp_data.get("forecast", 0),
            our_probability=opp_data.get("our_prob", 50) / 100.0,
            market_probability=opp_data.get("market_prob", 50) / 100.0,
            edge=opp_data.get("edge", 0) / 100.0,
            edge_percent=opp_data.get("edge", 0),
            expected_value=0,
            forecast_detail="momentum buy",
        )
        rec = BetRecommendation(
            decision=BetDecision.BET,
            opportunity=opp,
            confidence=0.65,
            bet_size_usd=bet_size,
            reasoning=f"Momentum buy: price dropped >15% over 3 readings, consensus still bullish",
            side="yes",
            ticker=ticker,
        )

        bet = await executor.place_bet(rec)
        trade_record = {
            "id": bet.id,
            "ticker": bet.ticker,
            "city": bet.city,
            "date": bet.date,
            "bucket": bet.bucket_label,
            "side": bet.side,
            "price": bet.price,
            "qty": bet.quantity,
            "cost": round(bet.cost_usd, 2),
            "edge": round(bet.edge_percent, 1),
            "our_prob": round(bet.our_probability * 100, 1),
            "market_prob": round(bet.market_probability * 100, 1),
            "spread": "momentum",
            "reasoning": rec.reasoning,
            "mode": config.EXECUTION_MODE,
            "time": bet.placed_at.isoformat(),
            "settled": False,
            "won": None,
            "pnl": None,
        }
        if config.EXECUTION_MODE == "dry" or len(bet.id) > 10:
            save_trade(trade_record)
            scan_state["placed_bets"].append(trade_record)
            scan_state["trade_summary"] = get_trade_summary()
        else:
            logger.warning(f"[Momentum] Order not confirmed — not logging to trade log")

        if price_watcher:
            price_watcher.clear_signal(ticker)


# ── Momentum watcher loop ───────────────────────────────────────────────

async def momentum_loop() -> None:
    """
    Polls PriceWatcher state and fires handle_momentum_signal() on new signals.
    Runs every 10s — lightweight since PriceWatcher does the heavy polling.
    """
    seen_signals: dict[str, str] = {}  # ticker -> last signal we acted on
    while True:
        await asyncio.sleep(10)
        if price_watcher is None:
            continue
        try:
            state = price_watcher.get_state()
            for ticker, signal in state.get("signals", {}).items():
                if signal and seen_signals.get(ticker) != signal:
                    seen_signals[ticker] = signal
                    asyncio.create_task(handle_momentum_signal(ticker, signal))
        except Exception as e:
            logger.error(f"[MomentumLoop] Error: {e}", exc_info=True)


# ── Scan Loop ───────────────────────────────────────────────────────────

async def settlement_loop():
    """
    Check Kalshi for settled positions every 30 minutes.
    Runs independently of the main scan loop so settlements are caught promptly.
    """
    # Initial delay to let executor initialize first
    await asyncio.sleep(30)
    while True:
        try:
            newly_settled = await executor.check_settlements()
            if newly_settled:
                scan_state["trade_summary"] = get_trade_summary()
                logger.info(f"[SettlementLoop] {len(newly_settled)} new settlements: "
                            f"{[s['ticker'] for s in newly_settled]}")
        except Exception as e:
            logger.error(f"[SettlementLoop] Error: {e}", exc_info=True)
        await asyncio.sleep(1800)  # 30 minutes


async def scan_loop():
    # NOTE: PriceWatcher momentum trading DISABLED.
    # In weather markets, price drops = forecast changed, not a buy signal.
    # Re-enable after base strategy is proven profitable.
    await asyncio.sleep(2)
    await executor.initialize()
    await position_manager.initialize()
    while True:
        try:
            await run_scan_cycle()
        except Exception as e:
            logger.error(f"[Scan] Error: {e}", exc_info=True)
        logger.info(f"[Scan] Next scan in {config.SCAN_INTERVAL}s")
        await asyncio.sleep(config.SCAN_INTERVAL)


# ── Intraday Loop (Synoptic + Bet Engine) ──────────────────────────────

async def intraday_loop():
    """Poll Synoptic for live temperature data and run the bet engine.

    Runs every 60 seconds during trading hours (8 AM - 6 PM CDT).
    At the right hour per month, evaluates YES and NO plays.
    """
    await asyncio.sleep(5)  # Let other services start first
    logger.info("[Intraday] Starting intraday scan loop")

    last_date = ""

    while True:
        try:
            # Get current time in CDT (UTC - 5)
            now_utc = datetime.now(timezone.utc)
            now_cdt = now_utc - timedelta(hours=5)
            today_str = now_cdt.strftime("%Y-%m-%d")
            hour = now_cdt.hour
            minute = now_cdt.minute

            # Reset daily state at midnight
            if today_str != last_date:
                logger.info(f"[Intraday] New day: {today_str}")
                bet_engine.reset_daily(today_str)
                synoptic.reset_daily()
                last_date = today_str
                intraday_state["active"] = False
                intraday_state["today_decisions"] = []
                intraday_state["yes_bet_placed"] = False
                intraday_state["no_bets_placed"] = False
                intraday_state["favorite_bet_placed"] = False

            # Only poll during daytime hours (8 AM - 6 PM CDT)
            if hour < 8 or hour >= 18:
                await asyncio.sleep(60)
                continue

            intraday_state["active"] = True

            # ── Poll Synoptic ──
            reading = await synoptic.poll_once()
            if reading:
                intraday_state["last_poll"] = reading["timestamp_cdt"]
                intraday_state["latest_temp"] = reading["temp_f"]
                intraday_state["trajectory"] = synoptic.get_trajectory()

            # ── Poll IEM DSM (settlement source) ──
            iem_data = await iem.poll_once(today_str)
            if iem_data:
                intraday_state["iem_high"] = iem_data["max_tmpf_int"]
                # Log discrepancy between Synoptic and IEM
                syn_max = synoptic.get_trajectory().get("day_max")
                iem_high = iem_data["max_tmpf_int"]
                if syn_max and iem_high and abs(syn_max - iem_high) >= 2:
                    logger.warning(
                        "[Intraday] ⚠️ DATA DISCREPANCY: Synoptic max=%d°F vs IEM DSM=%d°F "
                        "(IEM is settlement source)",
                        syn_max, iem_high,
                    )

            # ── Run bet engine from 10 AM+ (Play 3 may start at 10 in tier A months) ──
            if reading and hour >= 10:
                current_temp = reading["temp_f_int"]
                month = now_cdt.month

                # Determine sky cover from NWS forecast (rough)
                sky_cover = "CLR"  # Default; will be refined when NWS data is integrated
                if scan_state.get("forecasts"):
                    latest_fc = scan_state["forecasts"][0]
                    detail = latest_fc.get("detail", "").lower()
                    if "cloud" in detail or "overcast" in detail:
                        sky_cover = "BKN"
                    elif "partly" in detail or "mostly" in detail:
                        sky_cover = "SCT"

                # Fetch live Kalshi bucket data
                city = config.CITIES.get("AUS")
                if city:
                    try:
                        buckets = await bucket_cache.get_or_fetch(
                            "AUS", city, today_str
                        )
                        bucket_dicts = [
                            {
                                "ticker": b.ticker,
                                "label": b.label,
                                "low_bound": b.low_bound,
                                "high_bound": b.high_bound,
                                "yes_price": b.yes_price,
                                "no_price": b.no_price,
                                "volume": b.volume,
                            }
                            for b in buckets
                        ]
                    except Exception as e:
                        logger.warning(f"[Intraday] Kalshi fetch error: {e}")
                        bucket_dicts = []

                    if bucket_dicts:
                        # Get forecast data for bet engine
                        today_consensus = scan_state.get("consensus", {}).get(today_str, {})
                        consensus_val = today_consensus.get("consensus_high") if today_consensus else None
                        hourly_data = scan_state.get("hourly_today", [])
                        day_max_val = synoptic.get_trajectory().get("day_max")

                        # ── Play 1: YES prediction ──
                        if not bet_engine.yes_bet_placed_today:
                            yes_decision = bet_engine.evaluate_yes_play(
                                current_temp=current_temp,
                                current_hour=hour,
                                month=month,
                                sky_cover=sky_cover,
                                buckets=bucket_dicts,
                                day_max=day_max_val,
                                consensus_high=consensus_val,
                                nws_hourly=hourly_data,
                            )
                            bet_engine.log_decision(yes_decision)
                            intraday_state["today_decisions"].append(yes_decision)

                            if yes_decision["action"] == "bet":
                                bet_engine.yes_bet_placed_today = True
                                intraday_state["yes_bet_placed"] = True
                                logger.info(
                                    f"[Intraday] 🎯 YES BET: {yes_decision['bucket']} "
                                    f"@ {yes_decision['price']:.2f} x{yes_decision['contracts']}"
                                )
                                # In dry mode, just log. In live mode, would place via executor.

                        # ── Play 2: NO base income ──
                        if not bet_engine.no_bets_placed_today:
                            no_decisions = bet_engine.evaluate_no_plays(
                                current_temp=current_temp,
                                current_hour=hour,
                                month=month,
                                sky_cover=sky_cover,
                                buckets=bucket_dicts,
                            )
                            for nd in no_decisions:
                                bet_engine.log_decision(nd)
                                intraday_state["today_decisions"].append(nd)

                            active_nos = [d for d in no_decisions if d.get("action") == "bet"]
                            if active_nos:
                                bet_engine.no_bets_placed_today = True
                                intraday_state["no_bets_placed"] = True
                                total_cost = sum(d.get("cost", 0) for d in active_nos)
                                logger.info(
                                    f"[Intraday] 💰 NO BETS: {len(active_nos)} buckets, "
                                    f"${total_cost:.2f} total"
                                )

                        intraday_state["daily_summary"] = bet_engine.get_daily_summary()

                        # ── Play 3: Market Favorite (Sweet Spot) ──
                        if not bet_engine.favorite_bet_placed_today:
                            fav_decision = bet_engine.evaluate_favorite_play(
                                current_temp=current_temp,
                                current_hour=hour,
                                month=month,
                                sky_cover=sky_cover,
                                buckets=bucket_dicts,
                                day_max=day_max_val,
                                consensus_high=consensus_val,
                                nws_hourly=hourly_data,
                            )
                            bet_engine.log_decision(fav_decision)
                            intraday_state["today_decisions"].append(fav_decision)

                            if fav_decision["action"] == "bet":
                                bet_engine.favorite_bet_placed_today = True
                                intraday_state["favorite_bet_placed"] = True
                                logger.info(
                                    f"[Intraday] ⭐ FAV BET: {fav_decision['bucket']} "
                                    f"@ {fav_decision['price']:.2f} x{fav_decision['contracts']} "
                                    f"({fav_decision.get('sweet_spot_edge', '')})"
                                )

                        intraday_state["daily_summary"] = bet_engine.get_daily_summary()

                # Log rounding crossings
                if reading and is_crossing_temp(reading["temp_f_int"]):
                    logger.warning(
                        f"[Intraday] ⚠️ ROUNDING CROSSING: {reading['temp_f_int']}°F "
                        f"({reading['temp_c']}°C) — bucket may be ambiguous"
                    )

        except Exception as e:
            logger.error(f"[Intraday] Error: {e}", exc_info=True)

        await asyncio.sleep(60)  # Poll every 60 seconds


# ── FastAPI Server ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    scan_task = asyncio.create_task(scan_loop())
    settlement_task = asyncio.create_task(settlement_loop())
    intraday_task = asyncio.create_task(intraday_loop())
    sniper_task = asyncio.create_task(sniper_loop(execution_mode=config.EXECUTION_MODE))
    # PriceWatcher + momentum loop DISABLED — see scan_loop comment
    yield
    scan_task.cancel()
    settlement_task.cancel()
    intraday_task.cancel()
    sniper_task.cancel()
    bet_engine.save_session_log()
    await executor.close()
    await position_manager.close()

app = FastAPI(title="Weather Bets", lifespan=lifespan)
DASHBOARD_DIR = Path(__file__).parent / "dashboard"

@app.get("/")
async def index():
    return FileResponse(DASHBOARD_DIR / "index.html")

@app.get("/api/state")
async def get_state():
    return scan_state

@app.get("/api/prices")
async def get_prices():
    """Expose PriceWatcher state: price history, signals, momentum."""
    if price_watcher is None:
        return {"status": "not_started", "tickers": {}, "signals": {}}
    return price_watcher.get_state()

@app.get("/api/positions")
async def get_positions():
    """Fetch open Kalshi portfolio positions."""
    return await position_manager.get_open_positions()

@app.get("/api/rescan")
async def trigger_rescan():
    asyncio.create_task(run_scan_cycle())
    return {"status": "scan_triggered"}

@app.get("/api/pending")
async def get_pending_bets():
    """Return all bets awaiting DC approval (semi-auto mode)."""
    import json as _json
    from pathlib import Path
    queue_file = Path(__file__).parent / "data" / "pending_bets.json"
    if not queue_file.exists():
        return {"pending": []}
    try:
        queue = _json.loads(queue_file.read_text())
        return {"pending": [b for b in queue if b.get("status") == "pending"]}
    except Exception:
        return {"pending": []}

@app.post("/api/approve/{ticker}")
async def approve_bet(ticker: str):
    """DC approves a pending bet — executes it immediately."""
    success = await executor.execute_approved_bet(ticker)
    if success:
        return {"status": "executed", "ticker": ticker}
    return {"status": "failed", "ticker": ticker}

@app.post("/api/reject/{ticker}")
async def reject_bet(ticker: str):
    """DC rejects a pending bet — removes it from the queue."""
    import json as _json
    from pathlib import Path
    queue_file = Path(__file__).parent / "data" / "pending_bets.json"
    if not queue_file.exists():
        return {"status": "not_found"}
    queue = _json.loads(queue_file.read_text())
    for b in queue:
        if b["ticker"] == ticker and b["status"] == "pending":
            b["status"] = "rejected"
    queue_file.write_text(_json.dumps(queue, indent=2))
    logger.info(f"[Semi-auto] Bet rejected by DC: {ticker}")
    return {"status": "rejected", "ticker": ticker}

@app.get("/api/intraday")
async def get_intraday():
    """Expose intraday bet engine state."""
    return {
        "active": intraday_state["active"],
        "last_poll": intraday_state["last_poll"],
        "latest_temp": intraday_state["latest_temp"],
        "trajectory": intraday_state["trajectory"],
        "yes_bet_placed": intraday_state["yes_bet_placed"],
        "no_bets_placed": intraday_state["no_bets_placed"],
        "favorite_bet_placed": intraday_state["favorite_bet_placed"],
        "daily_summary": intraday_state["daily_summary"],
        "decisions_today": len(intraday_state["today_decisions"]),
        "bet_decisions": [
            d for d in intraday_state["today_decisions"]
            if d.get("action") == "bet"
        ],
        "engine_balance": bet_engine.total_balance,
        "execution_mode": bet_engine.execution_mode,
    }

@app.get("/api/intraday/decisions")
async def get_intraday_decisions():
    """All decisions made by the bet engine today."""
    return {"decisions": intraday_state["today_decisions"]}

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
