#!/bin/bash
# Weather Betting Server — auto-start wrapper
# Used by launchd to start and keep the server running

cd "/Volumes/1TB SSD/project_arbitrage"
source venv/bin/activate
export PYTHONPATH="/Volumes/1TB SSD/project_arbitrage"

# Log startup
echo "$(date) — Starting weather betting server" >> "/Volumes/1TB SSD/project_arbitrage/weather_bets/data/server.log"

exec python3 -m weather_bets.main
