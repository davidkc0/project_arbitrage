"""Trade executor — places orders on both platforms for arbitrage trades."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx

from src.config import settings
from src.execution.risk import RiskManager
from src.models import (
    ArbitrageOpportunity,
    OutcomeSide,
    Platform,
    Position,
    Trade,
    TradeLeg,
    TradeStatus,
)

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Manages trade execution in semi-automated or automated mode.

    Semi-auto: Opportunities are queued; user confirms via dashboard.
    Auto: Trades execute immediately if they pass risk checks.
    """

    def __init__(self, risk_manager: RiskManager):
        self.risk = risk_manager
        self._pending_queue: list[tuple[ArbitrageOpportunity, float]] = []  # (opp, size)
        self._active_trades: dict[str, Trade] = {}
        self._positions: list[Position] = []
        self._trade_history: list[Trade] = []
        self._polymarket_http: httpx.AsyncClient | None = None
        self._kalshi_http: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Set up HTTP clients for trade execution."""
        self._polymarket_http = httpx.AsyncClient(
            base_url=settings.polymarket_host,
            timeout=15.0,
        )
        self._kalshi_http = httpx.AsyncClient(
            base_url=settings.kalshi_base_url,
            timeout=15.0,
        )

    # ── Queue management (semi-auto mode) ───────────────────────────────

    def queue_opportunity(
        self, opp: ArbitrageOpportunity, position_size: float
    ) -> str | None:
        """
        Queue an opportunity for user confirmation (semi-auto mode).
        Returns the opportunity ID if queued, None if rejected by risk.
        """
        allowed, reason = self.risk.check_trade(position_size)
        if not allowed:
            logger.warning(f"[Executor] Trade rejected: {reason}")
            return None

        self._pending_queue.append((opp, position_size))
        logger.info(
            f"[Executor] Queued opportunity {opp.id[:8]} — "
            f"edge={opp.net_edge_percent:.2f}%, size=${position_size:.2f}"
        )
        return opp.id

    @property
    def pending_queue(self) -> list[tuple[ArbitrageOpportunity, float]]:
        return list(self._pending_queue)

    @property
    def active_trades(self) -> list[Trade]:
        return list(self._active_trades.values())

    @property
    def positions(self) -> list[Position]:
        return list(self._positions)

    @property
    def trade_history(self) -> list[Trade]:
        return list(self._trade_history)

    def remove_from_queue(self, opp_id: str) -> bool:
        """Remove an opportunity from the pending queue."""
        for i, (opp, _) in enumerate(self._pending_queue):
            if opp.id == opp_id:
                self._pending_queue.pop(i)
                return True
        return False

    # ── Execution ───────────────────────────────────────────────────────

    async def execute_trade(self, opp_id: str) -> Trade | None:
        """
        Execute a queued trade (user confirmed in semi-auto mode).
        Returns the Trade object, or None on failure.
        """
        # Find the opportunity in the queue
        target = None
        for i, (opp, size) in enumerate(self._pending_queue):
            if opp.id == opp_id:
                target = (opp, size)
                self._pending_queue.pop(i)
                break

        if not target:
            logger.error(f"[Executor] Opportunity {opp_id} not found in queue.")
            return None

        opp, position_size = target

        # Final risk check
        allowed, reason = self.risk.check_trade(position_size)
        if not allowed:
            logger.warning(f"[Executor] Trade blocked at execution: {reason}")
            return None

        # Calculate number of contracts
        # Each contract pays $1.00 on the winning side
        contracts = position_size / opp.total_cost

        # Build trade legs
        leg_a = TradeLeg(
            platform=opp.buy_yes_platform,
            market_id=opp.event_a.platform_id
            if opp.buy_yes_platform == opp.event_a.platform
            else opp.event_b.platform_id,
            side=OutcomeSide.YES,
            price=opp.buy_yes_price,
            quantity=contracts,
            cost=opp.buy_yes_price * contracts,
        )

        leg_b = TradeLeg(
            platform=opp.buy_no_platform,
            market_id=opp.event_b.platform_id
            if opp.buy_no_platform == opp.event_b.platform
            else opp.event_a.platform_id,
            side=OutcomeSide.NO,
            price=opp.buy_no_price,
            quantity=contracts,
            cost=opp.buy_no_price * contracts,
        )

        trade = Trade(
            opportunity_id=opp.id,
            leg_a=leg_a,
            leg_b=leg_b,
            total_cost=position_size,
            expected_profit=opp.net_edge * contracts,
            status=TradeStatus.CONFIRMED,
        )

        self._active_trades[trade.id] = trade

        # Execute both legs concurrently
        logger.info(
            f"[Executor] Executing trade {trade.id[:8]} — "
            f"YES on {opp.buy_yes_platform.value} @ {opp.buy_yes_price:.4f}, "
            f"NO on {opp.buy_no_platform.value} @ {opp.buy_no_price:.4f}, "
            f"contracts={contracts:.2f}"
        )

        try:
            results = await asyncio.gather(
                self._place_order(leg_a),
                self._place_order(leg_b),
                return_exceptions=True,
            )

            # Check results
            success_a = not isinstance(results[0], Exception)
            success_b = not isinstance(results[1], Exception)

            if success_a and success_b:
                trade.status = TradeStatus.FILLED
                trade.leg_a.status = TradeStatus.FILLED
                trade.leg_a.filled_at = datetime.utcnow()
                trade.leg_b.status = TradeStatus.FILLED
                trade.leg_b.filled_at = datetime.utcnow()

                # Create position
                position = Position(
                    trade_id=trade.id,
                    platform_a=leg_a.platform,
                    platform_a_market_id=leg_a.market_id,
                    platform_a_side=leg_a.side,
                    platform_a_quantity=leg_a.quantity,
                    platform_b=leg_b.platform,
                    platform_b_market_id=leg_b.market_id,
                    platform_b_side=leg_b.side,
                    platform_b_quantity=leg_b.quantity,
                    total_cost=trade.total_cost,
                )
                self._positions.append(position)
                self.risk.update_positions(self._positions)

                logger.info(f"[Executor] ✅ Trade {trade.id[:8]} fully filled.")

            elif success_a or success_b:
                trade.status = TradeStatus.PARTIALLY_FILLED
                if success_a:
                    trade.leg_a.status = TradeStatus.FILLED
                    trade.leg_b.status = TradeStatus.FAILED
                else:
                    trade.leg_a.status = TradeStatus.FAILED
                    trade.leg_b.status = TradeStatus.FILLED
                logger.warning(
                    f"[Executor] ⚠️  Trade {trade.id[:8]} partially filled! "
                    "Manual intervention may be needed."
                )
            else:
                trade.status = TradeStatus.FAILED
                trade.leg_a.status = TradeStatus.FAILED
                trade.leg_b.status = TradeStatus.FAILED
                logger.error(f"[Executor] ❌ Trade {trade.id[:8]} failed on both legs.")

        except Exception as e:
            trade.status = TradeStatus.FAILED
            logger.error(f"[Executor] Trade {trade.id[:8]} error: {e}", exc_info=True)

        # Move to history
        self._trade_history.append(trade)
        self._active_trades.pop(trade.id, None)

        return trade

    async def _place_order(self, leg: TradeLeg) -> bool:
        """
        Place a single order on the appropriate platform.

        NOTE: This is a DRY RUN implementation for safety.
        Real API calls are commented out — uncomment when ready to go live.
        """
        leg.placed_at = datetime.utcnow()

        if leg.platform == Platform.POLYMARKET:
            return await self._place_polymarket_order(leg)
        elif leg.platform == Platform.KALSHI:
            return await self._place_kalshi_order(leg)
        else:
            raise ValueError(f"Unknown platform: {leg.platform}")

    async def _place_polymarket_order(self, leg: TradeLeg) -> bool:
        """
        Place an order on Polymarket via the CLOB API.

        ⚠️ DRY RUN MODE: Logs the order but does not execute.
        To go live, uncomment the API call below.
        """
        logger.info(
            f"[Polymarket] 🔸 DRY RUN — Would place {leg.side.value} order: "
            f"market={leg.market_id}, price={leg.price:.4f}, qty={leg.quantity:.2f}"
        )

        # ── UNCOMMENT TO GO LIVE ────────────────────────────────────────
        # from py_clob_client.client import ClobClient
        # from py_clob_client.clob_types import OrderArgs, OrderType
        #
        # client = ClobClient(
        #     settings.polymarket_host,
        #     key=settings.polymarket_private_key,
        #     chain_id=settings.polymarket_chain_id,
        #     funder=settings.polymarket_funder_address or None,
        # )
        # client.set_api_creds(client.create_or_derive_api_creds())
        #
        # token_index = 0 if leg.side == OutcomeSide.YES else 1
        # token_id = ...  # Get from the MarketEvent
        #
        # order = client.create_and_post_order(OrderArgs(
        #     token_id=token_id,
        #     price=leg.price,
        #     size=leg.quantity,
        #     side="BUY",
        #     order_type=OrderType.GTC,
        # ))
        # leg.platform_order_id = order.get("orderID", "")
        # ────────────────────────────────────────────────────────────────

        leg.filled_quantity = leg.quantity
        leg.filled_avg_price = leg.price
        return True

    async def _place_kalshi_order(self, leg: TradeLeg) -> bool:
        """
        Place an order on Kalshi via the Trade API v2.

        ⚠️ DRY RUN MODE: Logs the order but does not execute.
        To go live, uncomment the API call below.
        """
        logger.info(
            f"[Kalshi] 🔸 DRY RUN — Would place {leg.side.value} order: "
            f"market={leg.market_id}, price={leg.price:.4f}, qty={leg.quantity:.2f}"
        )

        # ── UNCOMMENT TO GO LIVE ────────────────────────────────────────
        # import uuid
        #
        # resp = await self._kalshi_http.post(
        #     "/portfolio/orders",
        #     json={
        #         "ticker": leg.market_id,
        #         "client_order_id": str(uuid.uuid4()),
        #         "type": "limit",
        #         "action": "buy",
        #         "side": "yes" if leg.side == OutcomeSide.YES else "no",
        #         "count": int(leg.quantity),
        #         "yes_price": int(leg.price * 100),  # Kalshi uses cents
        #     },
        # )
        # resp.raise_for_status()
        # data = resp.json()
        # leg.platform_order_id = data.get("order", {}).get("order_id", "")
        # ────────────────────────────────────────────────────────────────

        leg.filled_quantity = leg.quantity
        leg.filled_avg_price = leg.price
        return True

    async def close(self) -> None:
        """Clean up HTTP clients."""
        if self._polymarket_http:
            await self._polymarket_http.aclose()
        if self._kalshi_http:
            await self._kalshi_http.aclose()
