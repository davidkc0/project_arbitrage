"""Risk management — position limits, exposure checks, kill switch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import settings
from src.models import Position, TradeStatus

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Current portfolio risk state."""
    open_positions: list[Position] = field(default_factory=list)
    total_exposure: float = 0.0
    pending_trades: int = 0
    killed: bool = False  # Kill switch engaged

    @property
    def available_exposure(self) -> float:
        return max(0.0, settings.max_total_exposure - self.total_exposure)

    @property
    def can_open_new(self) -> bool:
        if self.killed:
            return False
        if len(self.open_positions) >= settings.max_concurrent_positions:
            return False
        if self.available_exposure <= 0:
            return False
        return True


class RiskManager:
    """Enforces risk limits and manages the kill switch."""

    def __init__(self):
        self._state = RiskState()

    @property
    def state(self) -> RiskState:
        return self._state

    def update_positions(self, positions: list[Position]) -> None:
        """Refresh the risk state from current positions."""
        open_positions = [p for p in positions if p.closed_at is None]
        self._state.open_positions = open_positions
        self._state.total_exposure = sum(p.total_cost for p in open_positions)

    def check_trade(self, trade_cost: float) -> tuple[bool, str]:
        """
        Check if a proposed trade passes risk limits.
        Returns (allowed, reason).
        """
        if self._state.killed:
            return False, "Kill switch is engaged. No new trades allowed."

        if not self._state.can_open_new:
            return False, (
                f"Position limit reached ({len(self._state.open_positions)}/"
                f"{settings.max_concurrent_positions}) or no available exposure."
            )

        if trade_cost > settings.max_position_size:
            return False, (
                f"Trade cost ${trade_cost:.2f} exceeds max position size "
                f"${settings.max_position_size:.2f}."
            )

        if trade_cost > self._state.available_exposure:
            return False, (
                f"Trade cost ${trade_cost:.2f} exceeds available exposure "
                f"${self._state.available_exposure:.2f}."
            )

        return True, "OK"

    def calculate_position_size(self, edge_pct: float, max_bet_from_book: float) -> float:
        """
        Calculate the optimal position size for a given opportunity.

        Uses a conservative approach:
        - Never exceed max_position_size per leg
        - Never exceed available exposure
        - Never exceed order book depth
        - Scale down for lower-confidence edges
        """
        base_size = settings.max_position_size

        # Scale by edge (higher edge → larger position)
        # 1% edge → 50% of max, 3% edge → 100% of max
        edge_scale = min(1.0, edge_pct / 3.0)
        scaled = base_size * edge_scale

        # Cap by available exposure
        capped = min(scaled, self._state.available_exposure)

        # Cap by order book depth (use 80% of depth to avoid slippage)
        if max_bet_from_book > 0:
            capped = min(capped, max_bet_from_book * 0.8)

        # Minimum viable trade ($5)
        if capped < 5.0:
            return 0.0

        return round(capped, 2)

    def engage_kill_switch(self) -> None:
        """Stop all trading immediately."""
        self._state.killed = True
        logger.warning("[Risk] ⚠️  KILL SWITCH ENGAGED — all trading halted.")

    def disengage_kill_switch(self) -> None:
        """Resume trading."""
        self._state.killed = False
        logger.info("[Risk] Kill switch disengaged — trading resumed.")
