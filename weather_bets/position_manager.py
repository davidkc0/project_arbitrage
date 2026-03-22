"""Position manager — handles selling open Kalshi positions and querying the portfolio."""

from __future__ import annotations

import base64
import logging
import time
import uuid
from datetime import datetime, timezone

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding

from weather_bets.config import (
    EXECUTION_MODE,
    KALSHI_API_KEY_ID,
    KALSHI_BASE_URL,
    KALSHI_PRIVATE_KEY_PATH,
)
from weather_bets.trade_log import load_trades, save_trade

logger = logging.getLogger(__name__)


class PositionManager:
    """Manages open Kalshi positions: query and sell."""

    def __init__(self):
        self._http: httpx.AsyncClient | None = None
        self._private_key = None
        self._key_type: str = "unknown"

    async def initialize(self) -> None:
        """Set up HTTP client and load private key. Call once before use."""
        self._http = httpx.AsyncClient(
            base_url=KALSHI_BASE_URL,
            timeout=15.0,
        )
        try:
            from pathlib import Path
            key_path = Path(KALSHI_PRIVATE_KEY_PATH)
            if key_path.exists():
                key_data = key_path.read_bytes()
                self._private_key = serialization.load_pem_private_key(key_data, password=None)
                from cryptography.hazmat.primitives.asymmetric import rsa as rsa_mod
                from cryptography.hazmat.primitives.asymmetric import ec as ec_mod
                if isinstance(self._private_key, rsa_mod.RSAPrivateKey):
                    self._key_type = "rsa"
                elif isinstance(self._private_key, ec_mod.EllipticCurvePrivateKey):
                    self._key_type = "ec"
                logger.info(f"[PositionManager] Loaded private key (type={self._key_type})")
            else:
                logger.warning(f"[PositionManager] Private key not found: {KALSHI_PRIVATE_KEY_PATH}")
        except Exception as e:
            logger.error(f"[PositionManager] Key load error: {e}")

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        if not self._private_key:
            return ""
        message = f"{timestamp_ms}{method}{path}"
        message_bytes = message.encode("utf-8")
        if self._key_type == "rsa":
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
        """Generate Kalshi auth headers. path = full path e.g. /trade-api/v2/portfolio/positions"""
        timestamp_ms = int(time.time() * 1000)
        sign_path = path.split("?")[0]
        signature = self._sign_request(method, sign_path, timestamp_ms)
        return {
            "KALSHI-ACCESS-KEY": KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
            "Content-Type": "application/json",
        }

    # ── Public methods ──────────────────────────────────────────────────

    async def get_open_positions(self) -> list[dict]:
        """
        Fetch open positions from Kalshi portfolio.

        Returns a list of position dicts:
          {ticker, quantity, cost_basis, current_price, unrealized_pnl, ...}
        """
        if not self._http or not self._private_key:
            logger.warning("[PositionManager] Not authenticated — cannot fetch positions")
            return []

        try:
            path = "/portfolio/positions"
            full_path = f"/trade-api/v2{path}"
            headers = self._auth_headers("GET", full_path)
            resp = await self._http.get(path, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            raw_positions = data.get("market_positions", [])
            positions = []
            for p in raw_positions:
                qty = p.get("position", 0)
                if qty == 0:
                    continue
                positions.append({
                    "ticker": p.get("ticker", ""),
                    "quantity": qty,
                    "side": "yes" if qty > 0 else "no",
                    "realized_pnl_cents": p.get("realized_pnl", 0),
                    "resting_orders_count": p.get("resting_orders_count", 0),
                    "total_traded_cents": p.get("total_traded", 0),
                    "market_exposure_cents": p.get("market_exposure", 0),
                    "fees_paid_cents": p.get("fees_paid", 0),
                })

            logger.info(f"[PositionManager] {len(positions)} open positions from Kalshi")
            return positions

        except Exception as e:
            logger.error(f"[PositionManager] get_open_positions error: {e}", exc_info=True)
            return []

    async def sell_position(
        self,
        ticker: str,
        quantity: int | None = None,
        yes_price_cents: int | None = None,
        reason: str = "momentum sell",
    ) -> dict:
        """
        Sell an open position on Kalshi.

        Kalshi sell = POST /portfolio/orders with action="sell".

        Args:
            ticker: Market ticker to sell.
            quantity: Number of contracts to sell. If None, looks up the full
                      position quantity from trade log.
            yes_price_cents: Limit price in cents. If None, uses a market-friendly
                             default (current ask − 2¢ to get filled fast).
            reason: Human-readable reason for the sale (logged).

        Returns:
            dict with order info and status.
        """
        result = {
            "ticker": ticker,
            "status": "unknown",
            "order_id": None,
            "quantity": quantity,
            "reason": reason,
            "time": datetime.now(timezone.utc).isoformat(),
        }

        # Resolve quantity from trade log if not provided
        if quantity is None:
            quantity = self._get_trade_log_quantity(ticker)

        if quantity <= 0:
            logger.warning(f"[PositionManager] sell_position: no quantity for {ticker}")
            result["status"] = "skipped_no_quantity"
            return result

        result["quantity"] = quantity

        if EXECUTION_MODE == "dry":
            logger.info(
                f"[PositionManager] 🔸 DRY RUN — Would sell {quantity}× {ticker} "
                f"(reason: {reason})"
            )
            result["status"] = "dry_run"
            self._log_sell_trade(ticker, quantity, None, reason, dry=True)
            return result

        if not self._http or not self._private_key:
            logger.error("[PositionManager] Not authenticated — cannot sell")
            result["status"] = "error_no_auth"
            return result

        try:
            # Determine sell price: use a slightly below-ask price to get filled
            # If caller didn't specify, we'll use a taker-friendly limit (98¢ on a ≥96 market)
            # In practice Kalshi fills limit sells immediately at market price if within spread
            if yes_price_cents is None:
                # Determine a reasonable sell price from trade log
                # Don't fire-sale at 2¢ — use original cost minus small discount
                trades = load_trades()
                orig_trade = next(
                    (t for t in reversed(trades) 
                     if t.get("ticker") == ticker and not t.get("settled")),
                    None,
                )
                if orig_trade and orig_trade.get("price"):
                    # Sell at original price minus 5¢ for quick fill, min 5¢
                    orig_cents = int(orig_trade["price"] * 100)
                    yes_price_cents = max(5, orig_cents - 5)
                    logger.info(
                        f"[PositionManager] Sell price: {yes_price_cents}¢ "
                        f"(original buy: {orig_cents}¢ minus 5¢ discount)"
                    )
                else:
                    # No trade history — use 10¢ floor, not 2¢
                    yes_price_cents = 10

            path = "/portfolio/orders"
            full_path = f"/trade-api/v2{path}"
            order_payload = {
                "ticker": ticker,
                "client_order_id": str(uuid.uuid4()),
                "type": "limit",
                "action": "sell",
                "side": "yes",          # selling YES contracts
                "count": quantity,
                "yes_price": yes_price_cents,
            }

            headers = self._auth_headers("POST", full_path)
            resp = await self._http.post(path, json=order_payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            order_id = data.get("order", {}).get("order_id", "unknown")
            logger.info(
                f"[PositionManager] ✅ SELL ORDER placed — "
                f"order_id={order_id} ticker={ticker} qty={quantity} "
                f"price={yes_price_cents}¢ reason={reason}"
            )
            result["status"] = "placed"
            result["order_id"] = order_id
            self._log_sell_trade(ticker, quantity, order_id, reason, dry=False)

        except Exception as e:
            logger.error(f"[PositionManager] Sell order failed for {ticker}: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)

        return result

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_trade_log_quantity(self, ticker: str) -> int:
        """Look up quantity of open position in the trade log."""
        trades = load_trades()
        for t in reversed(trades):
            if t.get("ticker") == ticker and not t.get("settled"):
                return t.get("qty", 1)
        return 0

    def _log_sell_trade(
        self,
        ticker: str,
        quantity: int,
        order_id: str | None,
        reason: str,
        dry: bool,
    ) -> None:
        """Append a sell record to the trade log."""
        save_trade({
            "type": "sell",
            "ticker": ticker,
            "qty": quantity,
            "order_id": order_id,
            "reason": reason,
            "mode": "dry" if dry else EXECUTION_MODE,
            "settled": False,
            "won": None,
            "pnl": None,
        })

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
