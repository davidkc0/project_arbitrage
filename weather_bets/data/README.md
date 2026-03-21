# ⚠️ CRITICAL — DO NOT DELETE OR MODIFY THESE FILES MANUALLY

## trades.json
This is the SINGLE SOURCE OF TRUTH for all placed bets.
- NEVER clear or overwrite this file during testing/debugging
- NEVER reset this file on restart
- On startup, the bot syncs FROM Kalshi to this file, not the other way around
- Deleting this caused duplicate orders and $11.79 in losses on 2026-03-21

## scans.json
Historical scan log. Safe to read, do not delete.

## forecast_accuracy*.json
Model accuracy tracking. Do not delete — takes days to rebuild.
