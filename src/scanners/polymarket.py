"""Polymarket scanner — fetches markets from the CLOB API."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx

from src.config import settings
from src.models import MarketEvent, OutcomePrice, OutcomeSide, Platform
from src.scanners.base import BaseScanner

logger = logging.getLogger(__name__)

# Polymarket Gamma API for market discovery
GAMMA_API_URL = "https://gamma-api.polymarket.com"

# Concurrent order book fetch settings
BOOK_BATCH_SIZE = 10  # Fetch 10 books at a time
MIN_VOLUME_FOR_BOOK = 500  # Skip book fetch for very low volume markets


class PolymarketScanner(BaseScanner):
    """Scan Polymarket for active binary markets with pricing data."""

    def __init__(self):
        super().__init__("Polymarket")
        self._http: httpx.AsyncClient | None = None
        self._clob_http: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        self._http = httpx.AsyncClient(timeout=30.0)
        self._clob_http = httpx.AsyncClient(
            base_url=settings.polymarket_host,
            timeout=15.0,
        )
        logger.info("[Polymarket] Initialized HTTP clients.")

    async def _fetch_book(self, token_id: str) -> dict | None:
        """Fetch order book for a single token. Returns None on failure."""
        try:
            resp = await self._clob_http.get(
                "/book",
                params={"token_id": token_id},
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    async def _fetch_books_batch(self, token_ids: list[str]) -> dict[str, dict]:
        """Fetch order books for multiple tokens concurrently."""
        tasks = [self._fetch_book(tid) for tid in token_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        books = {}
        for tid, result in zip(token_ids, results):
            if isinstance(result, dict):
                books[tid] = result
        return books

    async def fetch_markets(self) -> list[MarketEvent]:
        """
        1. Use the Gamma API to discover active binary markets.
        2. Enrich with live pricing from the CLOB API (concurrent batches).
        """
        if not self._http or not self._clob_http:
            raise RuntimeError("Scanner not initialized. Call start() first.")

        events: list[MarketEvent] = []

        try:
            # ── Step 1: Discover markets via Gamma API (paginated) ──────
            raw_markets = []
            pages = 20  # 20 pages × 100 = up to 2000 markets
            for page in range(pages):
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": 100,
                    "offset": page * 100,
                    "order": "volume24hr",
                    "ascending": "false",
                }
                resp = await self._http.get(f"{GAMMA_API_URL}/markets", params=params)
                resp.raise_for_status()
                page_markets = resp.json()
                if not page_markets:
                    break
                raw_markets.extend(page_markets)

            # Pre-parse markets and collect token IDs for batch book fetch
            parsed_markets = []
            token_to_market_idx = {}  # token_id → index in parsed_markets

            for market in raw_markets:
                # Only process binary markets (YES/NO)
                outcomes_raw = market.get("outcomes", "")
                if isinstance(outcomes_raw, str):
                    outcomes_raw = outcomes_raw.strip("[]").replace('"', "").split(",")
                outcomes_raw = [o.strip() for o in outcomes_raw]

                if len(outcomes_raw) != 2:
                    continue

                # Parse token IDs
                clob_token_ids = market.get("clobTokenIds", "")
                if isinstance(clob_token_ids, str):
                    clob_token_ids = clob_token_ids.strip("[]").replace('"', "").split(",")
                    clob_token_ids = [t.strip() for t in clob_token_ids if t.strip()]

                # Parse initial prices
                outcome_prices_raw = market.get("outcomePrices", "")
                if isinstance(outcome_prices_raw, str):
                    outcome_prices_raw = outcome_prices_raw.strip("[]").replace('"', "").split(",")
                    outcome_prices_raw = [p.strip() for p in outcome_prices_raw]

                yes_price = 0.0
                no_price = 0.0
                if len(outcome_prices_raw) >= 2:
                    try:
                        yes_price = float(outcome_prices_raw[0])
                        no_price = float(outcome_prices_raw[1])
                    except (ValueError, IndexError):
                        pass

                idx = len(parsed_markets)
                parsed_markets.append({
                    "market": market,
                    "clob_token_ids": clob_token_ids,
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "bid_depth_yes": 0.0,
                    "ask_depth_yes": 0.0,
                })

                # Only fetch books for markets with enough volume
                volume = float(market.get("volume", 0) or 0)
                if clob_token_ids and volume >= MIN_VOLUME_FOR_BOOK:
                    token_to_market_idx[clob_token_ids[0]] = idx

            # ── Step 2: Batch fetch order books concurrently ────────────
            token_ids = list(token_to_market_idx.keys())
            all_books: dict[str, dict] = {}

            for i in range(0, len(token_ids), BOOK_BATCH_SIZE):
                batch = token_ids[i : i + BOOK_BATCH_SIZE]
                batch_books = await self._fetch_books_batch(batch)
                all_books.update(batch_books)

            # Apply book data back to parsed markets
            for tid, book in all_books.items():
                idx = token_to_market_idx.get(tid)
                if idx is None:
                    continue
                pm = parsed_markets[idx]
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                if bids:
                    pm["yes_price"] = float(bids[0].get("price", pm["yes_price"]))
                    pm["bid_depth_yes"] = sum(
                        float(b.get("size", 0)) * float(b.get("price", 0))
                        for b in bids[:5]
                    )
                if asks:
                    pm["ask_depth_yes"] = sum(
                        float(a.get("size", 0)) * float(a.get("price", 0))
                        for a in asks[:5]
                    )

            # ── Step 3: Build MarketEvent objects ───────────────────────
            for pm in parsed_markets:
                market = pm["market"]
                outcomes = [
                    OutcomePrice(
                        side=OutcomeSide.YES,
                        best_bid=pm["yes_price"],
                        best_ask=pm["yes_price"],
                        last_price=pm["yes_price"],
                        bid_depth=pm["bid_depth_yes"],
                        ask_depth=pm["ask_depth_yes"],
                    ),
                    OutcomePrice(
                        side=OutcomeSide.NO,
                        best_bid=pm["no_price"],
                        best_ask=pm["no_price"],
                        last_price=pm["no_price"],
                    ),
                ]

                end_date = None
                end_str = market.get("endDate") or market.get("end_date_iso")
                if end_str:
                    try:
                        end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass

                condition_id = market.get("conditionId", market.get("condition_id", ""))
                market_slug = market.get("slug", "")
                url = f"https://polymarket.com/event/{market_slug}" if market_slug else ""

                event = MarketEvent(
                    platform=Platform.POLYMARKET,
                    platform_id=str(market.get("id", "")),
                    question=market.get("question", ""),
                    category=market.get("category", market.get("groupItemTitle", "")),
                    outcomes=outcomes,
                    end_date=end_date,
                    is_active=True,
                    url=url,
                    condition_id=condition_id,
                    token_ids=pm["clob_token_ids"] if pm["clob_token_ids"] else [],
                )
                events.append(event)

        except httpx.HTTPStatusError as e:
            logger.error(f"[Polymarket] HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[Polymarket] Fetch error: {e}", exc_info=True)

        return events

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()
        if self._clob_http:
            await self._clob_http.aclose()
        await super().stop()
