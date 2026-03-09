"""FastAPI server — orchestrates scanning, matching, and trade execution."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import settings
from src.database import Database
from src.execution.executor import TradeExecutor
from src.execution.risk import RiskManager
from src.matching.arbitrage import find_opportunities
from src.matching.matcher import EventMatcher
from src.models import ArbitrageOpportunity, Platform
from src.scanners.kalshi import KalshiScanner
from src.scanners.polymarket import PolymarketScanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Global State ────────────────────────────────────────────────────────

db = Database()
polymarket_scanner = PolymarketScanner()
kalshi_scanner = KalshiScanner()
matcher = EventMatcher()
risk_manager = RiskManager()
executor = TradeExecutor(risk_manager)

# Active WebSocket connections for the dashboard
ws_clients: list[WebSocket] = []
# Current opportunities (updated each scan cycle)
current_opportunities: list[ArbitrageOpportunity] = []
# Background task handle
scan_task: asyncio.Task | None = None


# ── Lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop background services with the app."""
    global scan_task

    # Initialize
    await db.initialize()
    await executor.initialize()

    # Load cached matched pairs
    cached_pairs = await db.load_matched_pairs()
    matcher.load_confirmed_pairs(cached_pairs)
    logger.info(f"Loaded {len(cached_pairs)} cached matched pairs.")

    # Start scanners
    await polymarket_scanner.start()
    await kalshi_scanner.start()

    # Start the scan → match → score loop
    scan_task = asyncio.create_task(scan_loop())

    yield

    # Shutdown
    if scan_task:
        scan_task.cancel()
    await polymarket_scanner.stop()
    await kalshi_scanner.stop()
    await executor.close()
    await db.close()


app = FastAPI(title="Polymarket ↔ Kalshi Arbitrage", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve Dashboard Static Files ───────────────────────────────────────

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


@app.get("/")
async def serve_dashboard():
    """Serve the dashboard index.html at the root."""
    return FileResponse(DASHBOARD_DIR / "index.html")


# Mount static assets (CSS, JS) at /static but also at root for simple paths
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


# ── Background Scan Loop ───────────────────────────────────────────────

async def scan_loop():
    """Main loop: scan → match → find opportunities → broadcast."""
    global current_opportunities

    # Wait for initial scan data
    await asyncio.sleep(5)

    while True:
        try:
            pm_events = polymarket_scanner.events
            k_events = kalshi_scanner.events

            if pm_events and k_events:
                # Find new matches
                new_matches = matcher.find_matches(pm_events, k_events)
                for match in new_matches:
                    await db.save_matched_pair(match)

                # Get all confirmed pairs
                all_pairs = await db.load_matched_pairs()

                # Calculate opportunities
                opps = find_opportunities(pm_events, k_events, all_pairs)
                current_opportunities = opps

                # Log top opportunities
                for opp in opps[:5]:
                    await db.log_opportunity({
                        "id": opp.id,
                        "event_a_question": opp.event_a.question[:80],
                        "event_b_question": opp.event_b.question[:80],
                        "buy_yes_platform": opp.buy_yes_platform.value,
                        "buy_yes_price": opp.buy_yes_price,
                        "buy_no_platform": opp.buy_no_platform.value,
                        "buy_no_price": opp.buy_no_price,
                        "total_cost": opp.total_cost,
                        "net_edge": opp.net_edge,
                        "net_edge_percent": opp.net_edge_percent,
                        "match_confidence": opp.match_confidence,
                        "discovered_at": opp.discovered_at.isoformat(),
                    })

                # Auto-queue in auto mode
                if settings.execution_mode == "auto":
                    for opp in opps:
                        size = risk_manager.calculate_position_size(
                            opp.net_edge_percent, opp.max_bet_size
                        )
                        if size > 0:
                            queued = executor.queue_opportunity(opp, size)
                            if queued:
                                await executor.execute_trade(opp.id)

                # Broadcast to WebSocket clients
                await broadcast_state()

            logger.info(
                f"[Loop] PM={len(pm_events)} events, K={len(k_events)} events, "
                f"matches={len(matcher.confirmed_pairs)}, "
                f"opps={len(current_opportunities)}"
            )

        except Exception as e:
            logger.error(f"[Loop] Error: {e}", exc_info=True)

        await asyncio.sleep(settings.scan_interval_seconds)


async def broadcast_state():
    """Send current state to all connected WebSocket clients."""
    if not ws_clients:
        return

    state = build_state_payload()
    message = json.dumps(state, default=str)

    disconnected = []
    for ws in ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        ws_clients.remove(ws)


def build_state_payload() -> dict:
    """Build the full state payload for the dashboard."""
    return {
        "type": "state_update",
        "timestamp": datetime.utcnow().isoformat(),
        "scanner": {
            "polymarket_count": len(polymarket_scanner.events),
            "kalshi_count": len(kalshi_scanner.events),
            "matched_pairs": len(matcher.confirmed_pairs),
        },
        "opportunities": [
            {
                "id": opp.id,
                "event_a_question": opp.event_a.question,
                "event_a_platform": opp.event_a.platform.value,
                "event_a_url": opp.event_a.url,
                "event_b_question": opp.event_b.question,
                "event_b_platform": opp.event_b.platform.value,
                "event_b_url": opp.event_b.url,
                "buy_yes_platform": opp.buy_yes_platform.value,
                "buy_yes_price": opp.buy_yes_price,
                "buy_no_platform": opp.buy_no_platform.value,
                "buy_no_price": opp.buy_no_price,
                "total_cost": opp.total_cost,
                "gross_edge": opp.gross_edge,
                "net_edge": opp.net_edge,
                "net_edge_percent": opp.net_edge_percent,
                "match_confidence": opp.match_confidence,
                "max_bet_size": opp.max_bet_size,
                "discovered_at": opp.discovered_at.isoformat(),
                # Price comparison per side
                "yes_a": opp.yes_a,
                "yes_b": opp.yes_b,
                "no_a": opp.no_a,
                "no_b": opp.no_b,
                "yes_spread": opp.yes_spread,
                "no_spread": opp.no_spread,
            }
            for opp in current_opportunities[:50]
        ],
        "execution": {
            "mode": settings.execution_mode,
            "pending_queue": len(executor.pending_queue),
            "active_trades": len(executor.active_trades),
            "total_positions": len(executor.positions),
            "trade_history_count": len(executor.trade_history),
        },
        "risk": {
            "total_exposure": risk_manager.state.total_exposure,
            "available_exposure": risk_manager.state.available_exposure,
            "open_positions": len(risk_manager.state.open_positions),
            "max_positions": settings.max_concurrent_positions,
            "max_position_size": settings.max_position_size,
            "max_total_exposure": settings.max_total_exposure,
            "killed": risk_manager.state.killed,
        },
        "pending_trades": [
            {
                "opp_id": opp.id,
                "question": opp.event_a.question[:60],
                "net_edge_percent": opp.net_edge_percent,
                "size": size,
            }
            for opp, size in executor.pending_queue
        ],
        "trade_history": [
            {
                "id": t.id,
                "status": t.status.value,
                "total_cost": t.total_cost,
                "expected_profit": t.expected_profit,
                "actual_profit": t.actual_profit,
                "created_at": t.created_at.isoformat(),
            }
            for t in executor.trade_history[-20:]
        ],
    }


# ── WebSocket ───────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    logger.info(f"[WS] Client connected ({len(ws_clients)} total)")

    # Send initial state
    try:
        state = build_state_payload()
        await ws.send_text(json.dumps(state, default=str))
    except Exception:
        pass

    try:
        while True:
            data = await ws.receive_text()
            await handle_ws_message(ws, data)
    except WebSocketDisconnect:
        ws_clients.remove(ws)
        logger.info(f"[WS] Client disconnected ({len(ws_clients)} total)")


async def handle_ws_message(ws: WebSocket, raw: str):
    """Handle commands from the dashboard."""
    try:
        msg = json.loads(raw)
        action = msg.get("action", "")

        if action == "execute_trade":
            opp_id = msg.get("opp_id", "")
            size = float(msg.get("size", settings.max_position_size))

            # Find the opportunity
            opp = next(
                (o for o in current_opportunities if o.id == opp_id), None
            )
            if opp:
                executor.queue_opportunity(opp, size)
                trade = await executor.execute_trade(opp_id)
                await broadcast_state()
            else:
                await ws.send_text(
                    json.dumps({"type": "error", "message": "Opportunity not found"})
                )

        elif action == "dismiss_opportunity":
            opp_id = msg.get("opp_id", "")
            executor.remove_from_queue(opp_id)
            await broadcast_state()

        elif action == "kill_switch":
            enabled = msg.get("enabled", True)
            if enabled:
                risk_manager.engage_kill_switch()
            else:
                risk_manager.disengage_kill_switch()
            await broadcast_state()

        elif action == "refresh":
            await broadcast_state()

    except Exception as e:
        logger.error(f"[WS] Message handling error: {e}")


# ── REST Endpoints ──────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return build_state_payload()


@app.get("/api/opportunities")
async def get_opportunities():
    return {
        "count": len(current_opportunities),
        "opportunities": build_state_payload()["opportunities"],
    }


@app.get("/api/volume")
async def get_volume():
    stats = await db.get_volume_stats()
    return {
        platform: {
            "total_volume_usd": s.total_volume_usd,
            "total_trades": s.total_trades,
            "total_markets_traded": s.total_markets_traded,
            "first_trade_date": s.first_trade_date.isoformat() if s.first_trade_date else None,
            "last_trade_date": s.last_trade_date.isoformat() if s.last_trade_date else None,
        }
        for platform, s in stats.items()
    }


class KillSwitchRequest(BaseModel):
    enabled: bool


@app.post("/api/kill-switch")
async def toggle_kill_switch(req: KillSwitchRequest):
    if req.enabled:
        risk_manager.engage_kill_switch()
    else:
        risk_manager.disengage_kill_switch()
    return {"killed": risk_manager.state.killed}


@app.get("/api/matched-pairs")
async def get_matched_pairs():
    pairs = await db.load_matched_pairs()
    return {
        "count": len(pairs),
        "pairs": [
            {
                "id": p.id,
                "polymarket_id": p.polymarket_id,
                "kalshi_ticker": p.kalshi_ticker,
                "polymarket_question": p.polymarket_question,
                "kalshi_question": p.kalshi_question,
                "match_confidence": p.match_confidence,
                "confirmed_by_user": p.confirmed_by_user,
            }
            for p in pairs
        ],
    }


# ── Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False,  # Set True for dev (requires 2x Ctrl+C to stop)
    )
