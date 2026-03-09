"""Abstract base for all platform scanners."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from src.models import MarketEvent

logger = logging.getLogger(__name__)


class BaseScanner(ABC):
    """Base class that all platform scanners must implement."""

    def __init__(self, name: str):
        self.name = name
        self._events: dict[str, MarketEvent] = {}  # platform_id → MarketEvent
        self._running = False
        self._task: asyncio.Task | None = None

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def events(self) -> list[MarketEvent]:
        """Return all currently tracked market events."""
        return list(self._events.values())

    def get_event(self, platform_id: str) -> MarketEvent | None:
        return self._events.get(platform_id)

    async def start(self) -> None:
        """Start continuous scanning in the background."""
        if self._running:
            return
        self._running = True
        logger.info(f"[{self.name}] Scanner starting...")
        await self.initialize()
        self._task = asyncio.create_task(self._scan_loop())

    async def stop(self) -> None:
        """Stop the scanner."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"[{self.name}] Scanner stopped.")

    # ── Abstract methods ────────────────────────────────────────────────

    @abstractmethod
    async def initialize(self) -> None:
        """One-time setup (authenticate, etc.)."""
        ...

    @abstractmethod
    async def fetch_markets(self) -> list[MarketEvent]:
        """Fetch and return normalized market events from the platform."""
        ...

    # ── Internal ────────────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        """Continuously fetch markets at the configured interval."""
        from src.config import settings

        while self._running:
            try:
                markets = await self.fetch_markets()
                for market in markets:
                    self._events[market.platform_id] = market
                logger.info(
                    f"[{self.name}] Scanned {len(markets)} markets "
                    f"(total tracked: {len(self._events)})"
                )
            except Exception as e:
                logger.error(f"[{self.name}] Scan error: {e}", exc_info=True)

            await asyncio.sleep(settings.scan_interval_seconds)
