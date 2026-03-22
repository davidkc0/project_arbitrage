---
description: How to run and monitor the Kalshi temperature betting system
---

# Weather Betting System — Operations Guide

## Overview

This is a dual-strategy automated temperature betting system for Kalshi. It runs as a FastAPI server that:

1. **Polls Synoptic Data API** every 60 seconds for live KAUS (Austin) temperature
2. **Runs a bet engine** at noon+ CDT that evaluates two plays:
   - **Play 1 (YES):** Predicts which 2°F bucket the daily high lands in → buys YES
   - **Play 2 (NO):** Identifies "dead" buckets (temp already above that range) → buys NO for base income
3. **Archives all data** to CSV for post-trial analysis
4. **Logs all decisions** with full reasoning

## Quick Start

// turbo-all

1. Navigate to the project directory:
```bash
cd "/Volumes/1TB SSD/project_arbitrage"
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Start the server:
```bash
PYTHONPATH="/Volumes/1TB SSD/project_arbitrage" python3 -m weather_bets.main
```

The server runs on `http://127.0.0.1:8001`.

## What Happens Automatically

The server runs three background loops:

| Loop | Interval | Purpose |
|---|---|---|
| `scan_loop` | Every 30 min | Fetches NWS/model forecasts, calculates spreads |
| `settlement_loop` | Every 30 min | Checks Kalshi for settled bets |
| `intraday_loop` | Every 60 sec | Polls Synoptic temp, runs bet engine |

### Intraday Loop Schedule (All times CDT = UTC-5)

- **Before 8 AM CDT:** Sleeps
- **8 AM – 11:59 AM:** Polls Synoptic for live temp, builds trajectory, archives data
- **12:00 PM – 5:59 PM:** Polls + runs bet engine (evaluates YES and NO plays using live Kalshi prices)
- **6 PM+:** Sleeps until next day

At midnight CDT, the intraday system resets for the new day.

## Key API Endpoints

| Endpoint | What It Shows |
|---|---|
| `GET /api/intraday` | Live intraday state: current temp, trajectory, bet decisions, balance |
| `GET /api/intraday/decisions` | All bet engine decisions made today (bets + skips with reasoning) |
| `GET /api/state` | Full scan state: forecasts, models, consensus, Kalshi buckets |
| `GET /api/positions` | Open Kalshi portfolio positions |
| `GET /api/pending` | Bets awaiting approval (semi-auto mode) |

## Current Configuration

### Execution Mode

The system is in **dry mode** (`WEATHER_EXECUTION_MODE=dry` in `.env`). All bet decisions are logged but no real orders are placed. To go live:

1. Open `/Volumes/1TB SSD/project_arbitrage/.env`
2. Change `WEATHER_EXECUTION_MODE=dry` to `WEATHER_EXECUTION_MODE=live`
3. Restart the server

> **WARNING:** Only switch to live after reviewing 2+ weeks of dry run results.

### Bet Engine Settings

- **Total balance:** $100 (set in `main.py` when initializing `BetEngine`)
- **YES pool:** 60% ($60) — for prediction bets on clear days
- **NO pool:** 40% ($40) — for daily base income
- **YES sizing:** 30% of YES pool for 1-bucket CI, 20% for 2-bucket
- **NO sizing:** 30% of NO pool for dead buckets, 20% for overpriced far buckets

### Synoptic API

- **Token:** Stored in `.env` as `SYNOPTIC_API_TOKEN`
- **Trial:** 14-day trial started 2026-03-21, expires ~2026-04-04
- **Station:** KAUS (Austin-Bergstrom Airport, standard 5-min updates)
- **Archive:** All readings saved to `weather_bets/data/synoptic_archive/kaus_YYYY-MM-DD.csv`

## Key Files

| File | Purpose |
|---|---|
| `weather_bets/main.py` | FastAPI server, all background loops, API endpoints |
| `weather_bets/bet_engine.py` | Dual-strategy decision engine (YES + NO plays) |
| `weather_bets/synoptic_poller.py` | Synoptic Data API poller with CSV archiving |
| `weather_bets/rounding_map.py` | NWS F→C→F conversion lookup (crossing detection) |
| `weather_bets/executor.py` | Kalshi order placement (RSA-signed API requests) |
| `weather_bets/kalshi_weather.py` | Fetches Kalshi temperature bucket prices |
| `weather_bets/intraday_predictor.py` | Statistical lookup model for remaining temp rise |
| `weather_bets/config.py` | All config, city definitions, API keys from .env |
| `.env` | API keys, execution mode, bet limits |

## Understanding the Bet Engine

### Play 1 (YES) — Prediction

The engine uses hardcoded historical patterns (5 years of KAUS data) to predict remaining temperature rise. For each month and hour, it knows the average, p10, and p90 remaining rise on clear days.

**Rules:**
- Only bets on clear days (CLR/FEW sky cover)
- Must be past the earliest bet hour for the current month (e.g., 2 PM in March, 10 AM in Sep)
- 80% confidence interval must fit in ≤ 2 Kalshi buckets
- Kalshi YES price must be ≤ 70¢
- Max 1 YES bet per day
- Skips if predicted high is at a rounding crossing temperature

### Play 2 (NO) — Base Income

Buys NO on buckets that are clearly wrong:

1. **Dead buckets:** Current temp already exceeds the bucket's range (100% safe)
2. **Overpriced far buckets:** Buckets >6°F below predicted high with YES ≥ 5¢ (clear days only)

**Rules:**
- Runs every day at noon+, any sky condition
- Only buys NO when YES price ≥ 3¢ (otherwise profit too thin)
- Runs once per day

### NWS Rounding Awareness

The system knows that KAUS temperatures undergo F→C→F double-conversion. 20 specific temperatures (69, 71, 74, 76, 87, 89, 92, 94°F, etc.) cross Kalshi bucket boundaries after this conversion. The system:
- Logs a ⚠️ warning when the current temp is at a crossing point
- Skips YES bets when the predicted high is a crossing temp and CI spans 2 buckets

## Monitoring & Troubleshooting

### Check if system is running:
```bash
curl -s http://127.0.0.1:8001/api/intraday | python3 -m json.tool
```

### Check today's decisions:
```bash
curl -s http://127.0.0.1:8001/api/intraday/decisions | python3 -m json.tool
```

### View Synoptic archive for today:
```bash
cat "/Volumes/1TB SSD/project_arbitrage/weather_bets/data/synoptic_archive/kaus_$(date +%Y-%m-%d).csv"
```

### View bet engine log:
```bash
cat "/Volumes/1TB SSD/project_arbitrage/weather_bets/data/bet_engine_log.json" | python3 -m json.tool | tail -50
```

### Restart the server:
```bash
lsof -ti:8001 | xargs kill -9 2>/dev/null
cd "/Volumes/1TB SSD/project_arbitrage"
source venv/bin/activate
PYTHONPATH="/Volumes/1TB SSD/project_arbitrage" python3 -m weather_bets.main
```

### Common Issues

- **Port already in use:** Kill the old process with `lsof -ti:8001 | xargs kill -9`
- **Synoptic "Invalid token":** The public token may have expired. Generate a new one:
  ```bash
  curl -s -X POST "https://api.synopticdata.com/auth/v2/tokens" \
    -H "Authorization: Bearer $(grep SYNOPTIC_API_KEY .env | cut -d= -f2)" \
    -H "Content-Type: application/json" \
    -d '{"name": "weather_bets_poller"}' | python3 -m json.tool
  ```
  Then update `SYNOPTIC_API_TOKEN` in `.env`.
- **No intraday data:** The intraday loop only runs 8 AM – 6 PM CDT. Outside those hours it sleeps.
- **No bets placed:** The bet engine only runs at noon+ CDT, and YES bets only on clear days. Check the decisions endpoint to see skip reasons.
