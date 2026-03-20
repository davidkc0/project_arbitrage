"""Kalshi scanner — fetches binary markets via the Events API."""

from __future__ import annotations

import base64
import logging
import time
from datetime import datetime
from pathlib import Path

import httpx

from src.config import settings
from src.models import MarketEvent, OutcomePrice, OutcomeSide, Platform
from src.scanners.base import BaseScanner

logger = logging.getLogger(__name__)


class KalshiScanner(BaseScanner):
    """Scan Kalshi for active binary event markets using the Events API."""

    def __init__(self):
        super().__init__("Kalshi")
        self._http: httpx.AsyncClient | None = None
        self._auth_token: str = ""
        self._auth_expiry: float = 0.0

    async def initialize(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.kalshi_base_url,
            timeout=30.0,
        )
        await self._authenticate()
        logger.info("[Kalshi] Initialized and authenticated.")

    async def _authenticate(self) -> None:
        """
        Authenticate with Kalshi using API key + private key RSA signing.
        Tokens expire every 30 minutes, so we track expiry.
        """
        if not settings.kalshi_api_key_id or not settings.kalshi_private_key_path:
            logger.warning("[Kalshi] No API credentials configured. Running in read-only mode.")
            return

        try:
            key_path = Path(settings.kalshi_private_key_path)
            if not key_path.exists():
                logger.error(f"[Kalshi] Private key not found: {key_path}")
                return

            timestamp = str(int(time.time() * 1000))
            private_key_pem = key_path.read_text()

            try:
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import padding

                private_key = serialization.load_pem_private_key(
                    private_key_pem.encode(), password=None
                )

                msg = f"POST/trade-api/v2/login{timestamp}"
                signature = private_key.sign(
                    msg.encode(),
                    padding.PKCS1v15(),
                    hashes.SHA256(),
                )
                sig_b64 = base64.b64encode(signature).decode()

            except ImportError:
                logger.warning(
                    "[Kalshi] 'cryptography' package not installed. "
                    "Install with: pip install cryptography"
                )
                return

            resp = await self._http.post(
                "/login",
                json={
                    "id": settings.kalshi_api_key_id,
                    "signature": sig_b64,
                    "timestamp": timestamp,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._auth_token = data.get("token", "")
            self._auth_expiry = time.time() + 25 * 60  # Refresh 5 min early

            if self._http:
                self._http.headers["Authorization"] = f"Bearer {self._auth_token}"

            logger.info("[Kalshi] Authentication successful.")

        except Exception as e:
            logger.error(f"[Kalshi] Authentication failed: {e}", exc_info=True)

    async def _ensure_auth(self) -> None:
        """Re-authenticate if token is near expiry."""
        if time.time() > self._auth_expiry:
            await self._authenticate()

    async def fetch_markets(self) -> list[MarketEvent]:
        """
        Fetch active Kalshi markets via the /events endpoint.
        This returns clean binary markets (not multi-leg parlays).
        """
        if not self._http:
            raise RuntimeError("Scanner not initialized. Call start() first.")

        await self._ensure_auth()
        events: list[MarketEvent] = []
        seen_tickers: set[str] = set()

        try:
            cursor: str | None = None
            total_events_fetched = 0
            max_events = 1000  # Fetch up to 1000 events

            while total_events_fetched < max_events:
                params: dict = {
                    "limit": 100,
                    "status": "open",
                    "with_nested_markets": "true",
                }
                if cursor:
                    params["cursor"] = cursor

                resp = await self._http.get("/events", params=params)
                resp.raise_for_status()
                data = resp.json()

                event_list = data.get("events", [])
                if not event_list:
                    break

                for event_data in event_list:
                    event_title = event_data.get("title", "")
                    event_category = event_data.get("category", "")
                    event_ticker = event_data.get("event_ticker", "")

                    sub_markets = event_data.get("markets", [])
                    if not sub_markets:
                        continue

                    for market in sub_markets:
                        ticker = market.get("ticker", "")

                        # Skip multi-leg combo/parlay markets
                        if "KXMVECROSSCATEGORY" in ticker or "KXMVE" in ticker:
                            continue

                        # Skip duplicates
                        if ticker in seen_tickers:
                            continue
                        seen_tickers.add(ticker)

                        # Use event title as the question (cleaner than market title)
                        # For multi-market events (e.g. Fed rate brackets),
                        # append the market-specific detail
                        market_title = market.get("title", "")
                        if len(sub_markets) == 1:
                            question = event_title or market_title
                        else:
                            # Multi-outcome event: use market title if distinct
                            question = market_title or event_title

                        if not question:
                            continue

                        # Pricing — Kalshi now uses _dollars fields (already in dollars, e.g., 0.09 = 9¢)
                        # Fall back to old cents-based fields if _dollars aren't present
                        yes_bid = market.get("yes_bid_dollars")
                        if yes_bid is None:
                            yes_bid = (market.get("yes_bid") or 0) / 100.0
                        else:
                            yes_bid = float(yes_bid)

                        yes_ask = market.get("yes_ask_dollars")
                        if yes_ask is None:
                            yes_ask = (market.get("yes_ask") or 0) / 100.0
                        else:
                            yes_ask = float(yes_ask)

                        no_bid = market.get("no_bid_dollars")
                        if no_bid is None:
                            no_bid = (market.get("no_bid") or 0) / 100.0
                        else:
                            no_bid = float(no_bid)

                        no_ask = market.get("no_ask_dollars")
                        if no_ask is None:
                            no_ask = (market.get("no_ask") or 0) / 100.0
                        else:
                            no_ask = float(no_ask)

                        # Skip illiquid markets with no orderbook
                        if yes_ask <= 0 and no_ask <= 0:
                            continue

                        last_price = market.get("last_price_dollars")
                        if last_price is None:
                            last_price = (market.get("last_price") or 0) / 100.0
                        else:
                            last_price = float(last_price)

                        volume = float(market.get("volume", 0))

                        outcomes = [
                            OutcomePrice(
                                side=OutcomeSide.YES,
                                best_bid=yes_bid,
                                best_ask=yes_ask,
                                last_price=last_price,
                                volume=volume,
                            ),
                            OutcomePrice(
                                side=OutcomeSide.NO,
                                best_bid=no_bid,
                                best_ask=no_ask,
                                last_price=1.0 - last_price if last_price else 0.0,
                                volume=volume,
                            ),
                        ]

                        # End date
                        end_date = None
                        close_time = market.get("close_time")
                        if close_time:
                            try:
                                end_date = datetime.fromisoformat(
                                    close_time.replace("Z", "+00:00")
                                )
                            except (ValueError, TypeError):
                                pass

                        url = f"https://kalshi.com/markets/{ticker}" if ticker else ""

                        me = MarketEvent(
                            platform=Platform.KALSHI,
                            platform_id=ticker,
                            question=question,
                            category=event_category or "",
                            outcomes=outcomes,
                            end_date=end_date,
                            is_active=True,
                            url=url,
                            ticker=ticker,
                        )
                        events.append(me)

                total_events_fetched += len(event_list)
                cursor = data.get("cursor")
                if not cursor:
                    break

            # ── Fetch sports game series that don't appear in default listing ──
            game_series_tickers = [
                "KXNCAAMBGAME",   # NCAA Men's Basketball games
                "KXNBAGAME",      # NBA games
                "KXMLBGAME",      # MLB games
                "KXNHLGAME",      # NHL games
                "KXSOCCERGAME",   # Soccer matches
                "KXNCAAWBGAME",   # NCAA Women's Basketball
                "KXNCAAFBGAME",   # NCAA Football games
                "KXNFLGAME",      # NFL games
            ]
            game_market_count = 0
            for series_ticker in game_series_tickers:
                try:
                    series_cursor: str | None = None
                    for _ in range(5):  # Up to 500 game events per series
                        params: dict = {
                            "limit": 100,
                            "status": "open",
                            "with_nested_markets": "true",
                            "series_ticker": series_ticker,
                        }
                        if series_cursor:
                            params["cursor"] = series_cursor

                        resp = await self._http.get(
                            "/events", params=params,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        event_list = data.get("events", [])
                        if not event_list:
                            break

                        for event in event_list:
                            for market in event.get("markets", []):
                                ticker = market.get("ticker", "")
                                if ticker in seen_tickers:
                                    continue
                                # Skip multi-outcome parlays
                                if "KXMVE" in ticker:
                                    continue
                                seen_tickers.add(ticker)

                                question = market.get("title") or event.get("title", "")
                                if not question:
                                    continue

                                # Pricing
                                yes_bid = market.get("yes_bid_dollars")
                                if yes_bid is None:
                                    yes_bid = (market.get("yes_bid") or 0) / 100.0
                                else:
                                    yes_bid = float(yes_bid)

                                yes_ask = market.get("yes_ask_dollars")
                                if yes_ask is None:
                                    yes_ask = (market.get("yes_ask") or 0) / 100.0
                                else:
                                    yes_ask = float(yes_ask)

                                no_bid = market.get("no_bid_dollars")
                                if no_bid is None:
                                    no_bid = (market.get("no_bid") or 0) / 100.0
                                else:
                                    no_bid = float(no_bid)

                                no_ask = market.get("no_ask_dollars")
                                if no_ask is None:
                                    no_ask = (market.get("no_ask") or 0) / 100.0
                                else:
                                    no_ask = float(no_ask)

                                if yes_ask <= 0 and no_ask <= 0:
                                    continue

                                last_price = market.get("last_price_dollars")
                                if last_price is None:
                                    last_price = (market.get("last_price") or 0) / 100.0
                                else:
                                    last_price = float(last_price)

                                volume = float(market.get("volume", 0))

                                end_date_str = market.get("close_time") or market.get("expiration_time", "")
                                end_date = None
                                if end_date_str:
                                    try:
                                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                                    except (ValueError, TypeError):
                                        pass

                                outcomes = [
                                    OutcomePrice(side=OutcomeSide.YES, best_bid=yes_bid, best_ask=yes_ask, last_price=last_price),
                                    OutcomePrice(side=OutcomeSide.NO, best_bid=no_bid, best_ask=no_ask, last_price=1.0 - last_price if last_price else 0.0),
                                ]

                                url = f"https://kalshi.com/markets/{series_ticker.lower()}/{ticker.lower()}"
                                me = MarketEvent(
                                    platform=Platform.KALSHI,
                                    platform_id=ticker,
                                    question=question,
                                    outcomes=outcomes,
                                    volume=volume,
                                    end_date=end_date,
                                    url=url,
                                    ticker=ticker,
                                )
                                events.append(me)
                                game_market_count += 1

                        series_cursor = data.get("cursor")
                        if not series_cursor:
                            break
                except httpx.HTTPStatusError as e:
                    logger.debug(f"[Kalshi] Game series {series_ticker}: HTTP {e.response.status_code}")
                except Exception as e:
                    logger.debug(f"[Kalshi] Game series {series_ticker}: {e}")

            if game_market_count > 0:
                logger.info(f"[Kalshi] Fetched {game_market_count} additional game markets from sports series")

        except httpx.HTTPStatusError as e:
            logger.error(f"[Kalshi] HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[Kalshi] Fetch error: {e}", exc_info=True)

        return events

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()
        await super().stop()
