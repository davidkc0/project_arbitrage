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
    BET_SIZE_USD,
    EXECUTION_MODE,
    KALSHI_API_KEY_ID,
    KALSHI_BASE_URL,
    KALSHI_PRIVATE_KEY_PATH,
)
from weather_bets.models import BetRecommendation, PlacedBet

logger = logging.getLogger(__name__)

# Maximum percentage of available balance we'll put at risk
MAX_BALANCE_USAGE_PCT = 0.50  # 50% rule — never bet more than half of available funds


class WeatherExecutor:
    """Places weather bet orders on Kalshi."""

    def __init__(self):
        self._http: httpx.AsyncClient | None = None
        self._private_key = None
        self._key_type: str = "unknown"  # "rsa" or "ec"
        self.placed_bets: list[PlacedBet] = []
        self._cached_balance: float | None = None
        self._balance_fetched_at: float = 0

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
                # Detect key type
                from cryptography.hazmat.primitives.asymmetric import rsa as rsa_mod, ec as ec_mod
                if isinstance(self._private_key, rsa_mod.RSAPrivateKey):
                    self._key_type = "rsa"
                elif isinstance(self._private_key, ec_mod.EllipticCurvePrivateKey):
                    self._key_type = "ec"
                logger.info(f"[Executor] Loaded Kalshi private key (type={self._key_type})")
            else:
                logger.warning(f"[Executor] Private key not found at {KALSHI_PRIVATE_KEY_PATH}")
        except Exception as e:
            logger.error(f"[Executor] Failed to load private key: {e}")

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """Sign a Kalshi API request using the private key (RSA-PSS or EC)."""
        if not self._private_key:
            return ""

        message = f"{timestamp_ms}{method}{path}"
        message_bytes = message.encode("utf-8")

        if self._key_type == "rsa":
            # Kalshi uses RSA-PSS with MGF1(SHA256)
            signature = self._private_key.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
        else:
            signature = self._private_key.sign(
                message_bytes,
                ec.ECDSA(hashes.SHA256()),
            )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict:
        """Generate auth headers for a Kalshi API request.
        
        path should be the full path (e.g., /trade-api/v2/portfolio/balance).
        Query params are stripped before signing per Kalshi docs.
        """
        timestamp_ms = int(time.time() * 1000)
        # Strip query params for signing
        sign_path = path.split("?")[0]
        signature = self._sign_request(method, sign_path, timestamp_ms)
        return {
            "KALSHI-ACCESS-KEY": KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
            "Content-Type": "application/json",
        }

    async def get_balance(self) -> float:
        """Fetch available balance from Kalshi. Caches for 60 seconds."""
        now = time.time()
        if self._cached_balance is not None and (now - self._balance_fetched_at) < 60:
            return self._cached_balance

        if not self._http or not self._private_key:
            logger.warning("[Executor] Cannot fetch balance — not authenticated")
            return 0.0

        try:
            path = "/portfolio/balance"
            full_path = f"/trade-api/v2{path}"
            headers = self._auth_headers("GET", full_path)
            resp = await self._http.get(path, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Kalshi returns balance in cents
            balance_cents = data.get("balance", 0)
            balance_usd = balance_cents / 100.0
            self._cached_balance = balance_usd
            self._balance_fetched_at = now
            logger.info(f"[Executor] 💰 Account balance: ${balance_usd:.2f}")
            return balance_usd
        except Exception as e:
            logger.error(f"[Executor] Failed to fetch balance: {e}")
            return self._cached_balance or 0.0

    async def get_betting_budget(self) -> float:
        """
        Calculate how much we're allowed to bet this scan cycle.
        
        Rule: Never bet more than 50% of available funds total.
        This accounts for existing open positions.
        """
        balance = await self.get_balance()
        max_at_risk = balance * MAX_BALANCE_USAGE_PCT
        
        # Subtract cost of existing open (unsettled) bets placed this session
        from weather_bets.trade_log import load_trades
        open_trades = [t for t in load_trades() if not t.get("settled")]
        open_cost = sum(t.get("cost", 0) for t in open_trades)
        
        remaining_budget = max(0, max_at_risk - open_cost)
        logger.info(
            f"[Executor] Budget: ${balance:.2f} balance × 50% = ${max_at_risk:.2f} max | "
            f"${open_cost:.2f} open | ${remaining_budget:.2f} available to bet"
        )
        return remaining_budget

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
            # ── 50% RULE: check remaining budget ──
            remaining_budget = await self.get_betting_budget()
            if bet.cost_usd > remaining_budget:
                if remaining_budget <= 0:
                    logger.warning(
                        f"[Executor] ⛔ SKIPPING — 50% budget exhausted "
                        f"(would cost ${bet.cost_usd:.2f}, $0 remaining)"
                    )
                    return bet
                # Scale down contracts to fit budget
                old_contracts = contracts
                contracts = max(1, int(remaining_budget / opp.bucket.yes_price))
                bet.quantity = contracts
                bet.cost_usd = opp.bucket.yes_price * contracts
                logger.info(
                    f"[Executor] 📉 Scaled down: {old_contracts} → {contracts} contracts "
                    f"(${bet.cost_usd:.2f}) to fit 50% budget"
                )
            if not self._http or not self._private_key:
                logger.error("[Executor] Cannot place live order — not authenticated")
                return bet

            try:
                path = "/portfolio/orders"
                full_path = f"/trade-api/v2{path}"
                order_payload = {
                    "ticker": ticker,
                    "client_order_id": str(uuid.uuid4()),
                    "type": "limit",
                    "action": "buy",
                    "side": rec.side,
                    "count": contracts,
                    "yes_price": price_cents,
                }

                headers = self._auth_headers("POST", full_path)
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
