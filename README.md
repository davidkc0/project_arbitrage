# Polymarket ↔ Kalshi Arbitrage Scanner

A cross-platform prediction market arbitrage tool that scans Polymarket and Kalshi for pricing discrepancies, calculates risk-adjusted edges, and enables semi-automated trade execution.

## Quick Start

### 1. Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your Polymarket private key and Kalshi API credentials
```

### 2. Run
```bash
# Start the backend server
python -m src.main
```
The server starts at `http://127.0.0.1:8000`.

### 3. Dashboard
Open `dashboard/index.html` in your browser (or serve it with any static server).
The dashboard connects via WebSocket to the backend at `ws://127.0.0.1:8000/ws`.

```bash
# Option: serve dashboard with Python
cd dashboard && python -m http.server 3000
```

## How It Works

1. **Scan** — Pulls active markets from both Polymarket (Gamma + CLOB API) and Kalshi (Trade API v2)
2. **Match** — Uses fuzzy string matching to pair the same event across platforms
3. **Calculate** — Finds cross-platform spreads where YES + NO < $1.00 (guaranteed profit)
4. **Execute** — Queue trades for 1-click confirmation, with concurrent order placement on both platforms

## Architecture

```
src/
├── main.py              # FastAPI server + orchestration loop
├── config.py            # Environment-based configuration
├── models.py            # Pydantic data models
├── database.py          # SQLite persistence
├── scanners/
│   ├── base.py          # Abstract scanner
│   ├── polymarket.py    # Polymarket CLOB scanner
│   └── kalshi.py        # Kalshi API scanner
├── matching/
│   ├── matcher.py       # Fuzzy event matcher
│   └── arbitrage.py     # Edge calculator
└── execution/
    ├── executor.py      # Trade execution (DRY RUN by default)
    └── risk.py          # Position limits + kill switch

dashboard/
├── index.html           # Dashboard UI
├── style.css            # Dark theme styling
└── main.js              # WebSocket client + rendering
```

## Safety

⚠️ **The executor runs in DRY RUN mode by default.** No real trades are placed until you uncomment the API calls in `src/execution/executor.py`. This lets you verify the scanner and strategy before committing real capital.

## Configuration

All settings are in `.env`. Key parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_POSITION_SIZE` | $50 | Max USD per trade leg |
| `MAX_TOTAL_EXPOSURE` | $500 | Max total USD at risk |
| `MIN_EDGE_PERCENT` | 1.0% | Minimum net edge to show |
| `SCAN_INTERVAL_SECONDS` | 10 | Scan frequency |
| `EXECUTION_MODE` | semi | `semi` or `auto` |
