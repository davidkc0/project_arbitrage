"""Kalshi order executor for weather bets — places orders via Trade API v2."""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from datetime import datetime

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, utils

from weather_bets.config import (
    EXECUTION_MODE,
    KALSHI_API_KEY_ID,
    KALSHI_BASE_URL,
    KALSHI_PRIVATE_KEY_PATH,
)
from weather_bets.models import BetRecommendation, PlacedBet

logger = logging.getLogger(__name__)


class WeatherExecutor:
    """Places weather bet orders on Kalshi."""

    def __init__(self):
        self._http: httpx.AsyncClient | None = None
        self._private_key = None
        self.placed_bets: list[PlacedBet] = []

    async def initialize(self) -> None:
        """Set up HTTP client and load private key."""
        self._http = httpx.AsyncClient(
            base_url=KALSHI_BASE_URL,
            timeout=15.0,
        )

        # Load private key for request signing
        try:
            from pathlib import Path
            key_path = Path(KALSHI_PRIVATE_KEY_PATH)
            if key_path.exists():
                key_data = key_path.read_bytes()
                self._private_key = serialization.load_pem_private_key(key_data, password=None)
                logger.info("[Executor] Loaded Kalshi private key")
            else:
                logger.warning(f"[Executor] Private key not found at {KALSHI_PRIVATE_KEY_PATH}")
        except Exception as e:
            logger.error(f"[Executor] Failed to load private key: {e}")

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """Sign a Kalshi API request using the private key."""
        if not self._private_key:
            return ""

        message = f"{timestamp_ms}{method}{path}"
        message_bytes = message.encode("utf-8")

        signature = self._private_key.sign(
            message_bytes,
            ec.ECDSA(hashes.SHA256()),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict:
        """Generate auth headers for a Kalshi API request."""
        timestamp_ms = int(time.time() * 1000)
        signature = self._sign_request(method, path, timestamp_ms)
        return {
            "KALSHI-ACCESS-KEY": KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
            "Content-Type": "application/json",
        }

    async def place_bet(self, rec: BetRecommendation) -> PlacedBet:
        """
        Place a bet on Kalshi based on the LLM recommendation.
        
        In dry-run mode, logs the order but doesn't execute.
        In live mode, places the actual order.
        """
        opp = rec.opportunity
        ticker = rec.ticker or opp.bucket.ticker
        price_cents = int(opp.bucket.yes_price * 100)
        
        # Calculate contracts: each contract costs price_cents cents
        # and pays $1 (100 cents) on win
        contracts = max(1, int(rec.bet_size_usd / opp.bucket.yes_price))

        bet = PlacedBet(
            id=str(uuid.uuid4())[:8],
            ticker=ticker,
            city=opp.city,
            date=opp.date,
            bucket_label=opp.bucket.label,
            side=rec.side,
            price=opp.bucket.yes_price,
            quantity=contracts,
            cost_usd=opp.bucket.yes_price * contracts,
            our_probability=opp.our_probability,
            market_probability=opp.market_probability,
            edge_percent=opp.edge_percent,
            llm_reasoning=rec.reasoning,
        )

        if EXECUTION_MODE == "dry":
            logger.info(
                f"[Executor] 🔸 DRY RUN — Would place order:\n"
                f"  Ticker: {ticker}\n"
                f"  Side: {rec.side.upper()}\n"
                f"  Price: ${opp.bucket.yes_price:.2f} ({price_cents}¢)\n"
                f"  Contracts: {contracts}\n"
                f"  Total cost: ${bet.cost_usd:.2f}\n"
                f"  Edge: {opp.edge_percent:+.1f}%\n"
                f"  LLM confidence: {rec.confidence:.0%}\n"
                f"  Reasoning: {rec.reasoning}"
            )
        elif EXECUTION_MODE == "live":
            if not self._http or not self._private_key:
                logger.error("[Executor] Cannot place live order — not authenticated")
                return bet

            try:
                path = "/portfolio/orders"
                order_payload = {
                    "ticker": ticker,
                    "client_order_id": str(uuid.uuid4()),
                    "type": "limit",
                    "action": "buy",
                    "side": rec.side,
                    "count": contracts,
                    "yes_price": price_cents,
                }

                headers = self._auth_headers("POST", f"/trade-api/v2{path}")
                resp = await self._http.post(path, json=order_payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                order_id = data.get("order", {}).get("order_id", "unknown")
                logger.info(
                    f"[Executor] ✅ LIVE ORDER PLACED — order_id={order_id}\n"
                    f"  Ticker: {ticker}  Contracts: {contracts}  Cost: ${bet.cost_usd:.2f}"
                )
                bet.id = order_id

            except Exception as e:
                logger.error(f"[Executor] ❌ Order failed: {e}", exc_info=True)

        self.placed_bets.append(bet)
        return bet

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
