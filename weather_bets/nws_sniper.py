"""NWS Peak Temperature Sniper — detects NWS high temp changes and trades instantly.

Multi-city support: Runs independent sniper instances for each active Kalshi city.
Each city has its own NWS station, timezone, and season-aware peak temp window.

Strategy: Poll NWS observations every 60 seconds during each city's peak temp hours.
When the max °C reading increases and changes the winning Kalshi bucket, immediately
buy YES on the new winning bucket before the market fully reprices.

Evidence: On Mar 23, NWS went from 30°C→31°C at ~4:15 PM CT for Austin.
86-87° was at $0.93. It took the market 20-30 minutes to reprice to 88-89°.
This bot detects the change within 60 seconds and trades ahead.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta

import httpx

from weather_bets.rounding_map import get_kalshi_bucket, get_bucket_label
from weather_bets.kalshi_weather import fetch_weather_markets
from weather_bets.trade_log import save_trade
from weather_bets import config
from weather_bets import bucket_cache

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

POLL_INTERVAL_SECONDS = 60  # How often to poll NWS per city
MAX_BUY_PRICE = 0.50        # Don't buy if YES price is already above this
MIN_CONTRACTS = 1
MAX_CONTRACTS = 10

# Kalshi API
KALSHI_BASE_URL = config.KALSHI_BASE_URL
KALSHI_API_KEY_ID = config.KALSHI_API_KEY_ID
KALSHI_PRIVATE_KEY_PATH = config.KALSHI_PRIVATE_KEY_PATH

# ── City Sniper Configs ────────────────────────────────────────────────
# Each city: NWS station, UTC offset, and season-aware peak hour windows.
# Peak hours shift by season: summer peaks later (due to longer days),
# winter peaks earlier.

CITY_SNIPER_CONFIGS = {
    "AUS": {
        "station": "KAUS",
        "utc_offset": -5,  # CDT
        # month -> (start_hour_local, end_hour_local) for peak temp window
        "peak_hours": {
            1: (13, 16), 2: (13, 17), 3: (14, 18), 4: (14, 18),
            5: (14, 19), 6: (15, 19), 7: (15, 19), 8: (15, 19),
            9: (14, 18), 10: (14, 18), 11: (13, 17), 12: (13, 16),
        },
    },
    "LAX": {
        "station": "KLAX",
        "utc_offset": -7,  # PDT
        "peak_hours": {
            1: (12, 16), 2: (12, 16), 3: (13, 17), 4: (13, 17),
            5: (13, 18), 6: (13, 18), 7: (13, 18), 8: (13, 18),
            9: (13, 17), 10: (13, 17), 11: (12, 16), 12: (12, 16),
        },
    },
    "CHI": {
        "station": "KORD",
        "utc_offset": -5,  # CDT
        "peak_hours": {
            1: (12, 15), 2: (12, 16), 3: (13, 17), 4: (14, 18),
            5: (14, 18), 6: (14, 19), 7: (14, 19), 8: (14, 18),
            9: (13, 17), 10: (13, 17), 11: (12, 16), 12: (12, 15),
        },
    },
    "MIA": {
        "station": "KMIA",
        "utc_offset": -4,  # EDT
        "peak_hours": {
            1: (13, 17), 2: (13, 17), 3: (14, 17), 4: (14, 17),
            5: (14, 18), 6: (14, 18), 7: (14, 18), 8: (14, 18),
            9: (14, 17), 10: (14, 17), 11: (13, 17), 12: (13, 17),
        },
    },
    "DEN": {
        "station": "KDEN",
        "utc_offset": -6,  # MDT
        "peak_hours": {
            1: (12, 16), 2: (12, 16), 3: (13, 17), 4: (13, 17),
            5: (14, 18), 6: (14, 18), 7: (14, 18), 8: (14, 18),
            9: (13, 17), 10: (13, 17), 11: (12, 16), 12: (12, 16),
        },
    },
    "HOU": {
        "station": "KIAH",
        "utc_offset": -5,  # CDT
        "peak_hours": {
            1: (13, 17), 2: (13, 17), 3: (14, 18), 4: (14, 18),
            5: (14, 19), 6: (15, 19), 7: (15, 19), 8: (15, 19),
            9: (14, 18), 10: (14, 18), 11: (13, 17), 12: (13, 17),
        },
    },
    "DAL": {
        "station": "KDFW",
        "utc_offset": -5,  # CDT
        "peak_hours": {
            1: (13, 17), 2: (13, 17), 3: (14, 18), 4: (14, 18),
            5: (14, 19), 6: (15, 19), 7: (15, 19), 8: (15, 19),
            9: (14, 18), 10: (14, 18), 11: (13, 17), 12: (13, 17),
        },
    },
    "LAS": {
        "station": "KLAS",
        "utc_offset": -7,  # PDT
        "peak_hours": {
            1: (12, 16), 2: (12, 16), 3: (13, 17), 4: (13, 17),
            5: (14, 18), 6: (14, 18), 7: (14, 18), 8: (14, 18),
            9: (13, 17), 10: (13, 17), 11: (12, 16), 12: (12, 16),
        },
    },
    "SAT": {
        "station": "KSAT",
        "utc_offset": -5,  # CDT
        "peak_hours": {
            1: (13, 17), 2: (13, 17), 3: (14, 18), 4: (14, 18),
            5: (14, 19), 6: (15, 19), 7: (15, 19), 8: (15, 19),
            9: (14, 18), 10: (14, 18), 11: (13, 17), 12: (13, 17),
        },
    },
    "PHX": {
        "station": "KPHX",
        "utc_offset": -7,  # MST (Arizona doesn't observe DST)
        "peak_hours": {
            1: (12, 16), 2: (13, 16), 3: (13, 17), 4: (14, 17),
            5: (14, 18), 6: (14, 18), 7: (14, 18), 8: (14, 18),
            9: (14, 17), 10: (13, 17), 11: (12, 16), 12: (12, 16),
        },
    },
}


def celsius_to_settlement_f(celsius: int) -> int:
    """Convert integer Celsius (from NWS) to settlement Fahrenheit.

    NWS reports integer °C, converts to °F, rounds to nearest.
    This is the value Kalshi settles on.
    """
    return round(celsius * 9 / 5 + 32)


def settlement_f_to_bucket(temp_f: int) -> tuple[int, int]:
    """Get the Kalshi bucket for a given settlement °F."""
    return get_kalshi_bucket(temp_f)


class CitySniper:
    """Sniper instance for a single city. Tracks NWS readings and trades on bucket shifts."""

    def __init__(self, city_code: str, execution_mode: str = "dry"):
        self.city_code = city_code
        self.city_config = config.CITIES.get(city_code)
        self.sniper_config = CITY_SNIPER_CONFIGS.get(city_code, {})
        self.execution_mode = execution_mode

        self._station = self.sniper_config.get("station", "")
        self._utc_offset = self.sniper_config.get("utc_offset", -5)
        self._peak_hours = self.sniper_config.get("peak_hours", {})

        # Daily tracking state
        self._today: str = ""
        self._max_celsius: int | None = None
        self._current_bucket: str | None = None  # Now stores label string e.g. "86-87"
        self._trades_today: list[dict] = []
        self._last_poll_time: float = 0
        self._bucket_traded: set[str] = set()  # Set of bucket labels already traded

        # Real Kalshi bucket cache — fetched once per day
        self._kalshi_buckets: list = []  # List of TemperatureBucket objects

    def _get_local_time(self) -> datetime:
        """Get current local time for this city."""
        return datetime.now(timezone.utc) + timedelta(hours=self._utc_offset)

    def _is_peak_hour(self) -> bool:
        """Check if current local time is within this city's peak temp window."""
        local = self._get_local_time()
        month = local.month
        hour = local.hour
        peak = self._peak_hours.get(month, (13, 18))
        return peak[0] <= hour < peak[1]

    def _reset_daily(self, today: str):
        """Reset state for a new day."""
        self._today = today
        self._max_celsius = None
        self._current_bucket = None
        self._trades_today = []
        self._bucket_traded = set()
        self._kalshi_buckets = []  # Force re-fetch
        logger.info(f"[Sniper/{self.city_code}] New day: {today}")

    async def _load_kalshi_buckets(self, today: str):
        """Fetch and cache today's actual Kalshi bucket boundaries for this city."""
        if self._kalshi_buckets:
            return  # Already loaded for today
        try:
            if self.city_config:
                self._kalshi_buckets = await bucket_cache.get_or_fetch(
                    self.city_code, self.city_config, today
                )
                if self._kalshi_buckets:
                    labels = [b.label for b in sorted(
                        self._kalshi_buckets,
                        key=lambda b: b.low_bound if b.low_bound is not None else -999
                    )]
                    logger.info(
                        f"[Sniper/{self.city_code}] Loaded {len(self._kalshi_buckets)} "
                        f"Kalshi buckets: {', '.join(labels)}"
                    )
        except Exception as e:
            logger.warning(f"[Sniper/{self.city_code}] Failed to load buckets: {e}")

    def _find_bucket_for_temp(self, temp_f: int) -> dict | None:
        """Find which real Kalshi bucket contains this temperature.

        Returns dict with 'label', 'low', 'high', 'ticker', 'yes_price', 'no_price'
        or None if no buckets loaded or temp doesn't fit.
        """
        for b in self._kalshi_buckets:
            # Edge bucket: ≤X
            if b.low_bound is None and b.high_bound is not None:
                if temp_f <= b.high_bound:
                    return {
                        "label": b.label, "low": None, "high": b.high_bound,
                        "ticker": b.ticker, "yes_price": b.yes_price, "no_price": b.no_price,
                    }
            # Edge bucket: ≥X
            elif b.high_bound is None and b.low_bound is not None:
                if temp_f >= b.low_bound:
                    return {
                        "label": b.label, "low": b.low_bound, "high": None,
                        "ticker": b.ticker, "yes_price": b.yes_price, "no_price": b.no_price,
                    }
            # Normal bucket: X-Y
            elif b.low_bound is not None and b.high_bound is not None:
                if b.low_bound <= temp_f <= b.high_bound:
                    return {
                        "label": b.label, "low": b.low_bound, "high": b.high_bound,
                        "ticker": b.ticker, "yes_price": b.yes_price, "no_price": b.no_price,
                    }
        return None

    async def _poll_nws(self) -> int | None:
        """Fetch the latest NWS observation and return temperature in integer °C."""
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                headers={"User-Agent": "(weather-bets-sniper)"},
            ) as http:
                resp = await http.get(
                    f"https://api.weather.gov/stations/{self._station}/observations/latest"
                )
                resp.raise_for_status()
                data = resp.json()

            temp_c_raw = data.get("properties", {}).get("temperature", {}).get("value")
            if temp_c_raw is None:
                return None
            return round(temp_c_raw)
        except Exception as e:
            logger.warning(f"[Sniper/{self.city_code}] NWS poll failed: {e}")
            return None

    # _find_bucket_market removed — was dead code, use _find_bucket_for_temp instead

    async def _execute_buy(
        self, ticker: str, yes_price: float, label: str,
        http_client, auth_fn,
    ):
        """Place a YES buy order on Kalshi."""
        price_cents = int(yes_price * 100)
        contracts = min(MAX_CONTRACTS, max(MIN_CONTRACTS, int(5.0 / yes_price)))

        trade_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "nws_sniper",
            "city": self.city_code,
            "ticker": ticker,
            "label": label,
            "side": "yes",
            "action": "buy",
            "price": yes_price,
            "contracts": contracts,
            "cost": round(yes_price * contracts, 2),
            "nws_max_celsius": self._max_celsius,
            "settlement_f": celsius_to_settlement_f(self._max_celsius) if self._max_celsius else None,
        }

        if self.execution_mode == "dry":
            logger.info(
                f"[Sniper/{self.city_code}] 🔸 DRY RUN — Would buy:\n"
                f"  Ticker: {ticker} | Bucket: {label}\n"
                f"  Price: ${yes_price:.2f} | Contracts: {contracts}\n"
                f"  Cost: ${yes_price * contracts:.2f}"
            )
            trade_record["mode"] = "dry"

        elif self.execution_mode == "live":
            if not http_client or not auth_fn:
                logger.error(f"[Sniper/{self.city_code}] Cannot place live order — not authenticated")
                trade_record["error"] = "not_authenticated"
                self._trades_today.append(trade_record)
                return

            try:
                path = "/portfolio/orders"
                full_path = f"/trade-api/v2{path}"
                order_payload = {
                    "ticker": ticker,
                    "client_order_id": str(uuid.uuid4()),
                    "type": "limit",
                    "action": "buy",
                    "side": "yes",
                    "count": contracts,
                    "yes_price": price_cents,
                }

                headers = auth_fn("POST", full_path)
                resp = await http_client.post(path, json=order_payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                order_id = data.get("order", {}).get("order_id", "unknown")
                logger.info(
                    f"[Sniper/{self.city_code}] ✅ LIVE ORDER — {order_id}\n"
                    f"  {ticker} {contracts}x @ ${yes_price:.2f} = ${yes_price * contracts:.2f}"
                )
                trade_record["order_id"] = order_id
                trade_record["mode"] = "live"
                save_trade(trade_record, dry_run=False)

            except Exception as e:
                logger.error(f"[Sniper/{self.city_code}] ❌ Order failed: {e}", exc_info=True)
                trade_record["error"] = str(e)

        self._trades_today.append(trade_record)

    async def poll_and_act(self, http_client=None, auth_fn=None):
        """Single poll cycle for this city."""
        local = self._get_local_time()
        today = local.strftime("%Y-%m-%d")

        # Reset on new day
        if today != self._today:
            self._reset_daily(today)

        # Only active during peak hours
        if not self._is_peak_hour():
            return

        # Rate limit
        now_ts = time.time()
        if now_ts - self._last_poll_time < POLL_INTERVAL_SECONDS:
            return
        self._last_poll_time = now_ts

        # Load real Kalshi buckets (once per day)
        await self._load_kalshi_buckets(today)
        if not self._kalshi_buckets:
            logger.warning(f"[Sniper/{self.city_code}] No Kalshi buckets loaded, skipping")
            return

        # Poll NWS
        current_c = await self._poll_nws()
        if current_c is None:
            return

        current_f = celsius_to_settlement_f(current_c)

        # Find which REAL Kalshi bucket this temperature falls into
        bucket_info = self._find_bucket_for_temp(current_f)
        if not bucket_info:
            logger.warning(
                f"[Sniper/{self.city_code}] {current_f}°F doesn't fit any Kalshi bucket"
            )
            return

        current_label = bucket_info["label"]

        # First reading
        if self._max_celsius is None:
            self._max_celsius = current_c
            self._current_bucket = current_label
            logger.info(
                f"[Sniper/{self.city_code}] Initial: {current_c}°C = {current_f}°F "
                f"→ {current_label}"
            )
            return

        # No increase
        if current_c <= self._max_celsius:
            return

        # ── NEW MAX ──
        old_c = self._max_celsius
        old_f = celsius_to_settlement_f(old_c)
        old_label = self._current_bucket

        self._max_celsius = current_c
        self._current_bucket = current_label

        logger.info(
            f"[Sniper/{self.city_code}] 🔥 NEW MAX: {old_c}°C ({old_f}°F) "
            f"→ {current_c}°C ({current_f}°F)"
        )

        # Same bucket?
        if current_label == old_label:
            logger.info(f"[Sniper/{self.city_code}] Same bucket {current_label} — no trade")
            return

        # ── BUCKET SHIFT ──
        logger.info(
            f"[Sniper/{self.city_code}] 🚨 BUCKET SHIFT: {old_label} → {current_label}! "
            f"({old_c}°C→{current_c}°C)"
        )

        if current_label in self._bucket_traded:
            logger.info(f"[Sniper/{self.city_code}] Already traded {current_label}")
            return

        # ── MARKET-AWARE FILTER ──
        # Re-fetch fresh prices to get current favorite
        fresh_buckets = []
        try:
            if self.city_config:
                fresh_buckets = await bucket_cache.refresh_prices(
                    self.city_code, self.city_config, today
                )
        except Exception as e:
            logger.warning(f"[Sniper/{self.city_code}] Failed to fetch fresh prices: {e}")

        if fresh_buckets:
            # Update cached buckets with fresh prices
            self._kalshi_buckets = fresh_buckets

            # Find market favorite (highest YES price)
            favorite = max(fresh_buckets, key=lambda b: b.yes_price)
            fav_low = favorite.low_bound if favorite.low_bound is not None else -999
            fav_label = favorite.label

            # Our new bucket's lower bound
            new_low = bucket_info["low"] if bucket_info["low"] is not None else -999

            if new_low < fav_low:
                logger.info(
                    f"[Sniper/{self.city_code}] ⏭️ SKIP — temp still climbing. "
                    f"New bucket {current_label} is BELOW favorite {fav_label} "
                    f"(${favorite.yes_price:.2f}). Not a peak shift."
                )
                return

            if new_low == fav_low:
                logger.info(
                    f"[Sniper/{self.city_code}] ⏭️ SKIP — temp reached favorite "
                    f"{fav_label} (${favorite.yes_price:.2f}). Market already knows."
                )
                return

            logger.info(
                f"[Sniper/{self.city_code}] ✅ New bucket {current_label} is ABOVE "
                f"favorite {fav_label} (${favorite.yes_price:.2f}) — market is WRONG!"
            )

        # Re-lookup fresh price for our target bucket
        market = self._find_bucket_for_temp(current_f)
        if not market:
            logger.error(f"[Sniper/{self.city_code}] No market for {current_label}")
            return

        # Price check — don't buy if market already caught up
        if market["yes_price"] > MAX_BUY_PRICE:
            logger.info(
                f"[Sniper/{self.city_code}] {current_label} at ${market['yes_price']:.2f} "
                f"> ${MAX_BUY_PRICE:.2f} — market caught up"
            )
            return

        # ── TRADE ──
        logger.info(
            f"[Sniper/{self.city_code}] 💰 BUY {market['label']} YES @ ${market['yes_price']:.2f}"
        )
        await self._execute_buy(
            market["ticker"], market["yes_price"], market["label"],
            http_client, auth_fn,
        )
        self._bucket_traded.add(current_label)

    def get_status(self) -> dict:
        """Current state for API/dashboard."""
        return {
            "city": self.city_code,
            "active": self._is_peak_hour(),
            "today": self._today,
            "max_celsius": self._max_celsius,
            "max_fahrenheit": celsius_to_settlement_f(self._max_celsius) if self._max_celsius else None,
            "current_bucket": self._current_bucket,
            "trades_today": len(self._trades_today),
            "trade_details": self._trades_today,
            "peak_window": self._peak_hours.get(
                self._get_local_time().month, (0, 0)
            ),
        }


class MultiCitySniper:
    """Manages sniper instances for all active Kalshi cities."""

    def __init__(self, execution_mode: str = "dry"):
        self.execution_mode = execution_mode
        self.snipers: dict[str, CitySniper] = {}

        # Kalshi auth (shared across all snipers)
        self._http: httpx.AsyncClient | None = None
        self._private_key = None
        self._key_type = "rsa"

        # Create sniper for each configured city
        for city_code in CITY_SNIPER_CONFIGS:
            if city_code in config.CITIES:
                self.snipers[city_code] = CitySniper(city_code, execution_mode)

    async def initialize(self):
        """Load Kalshi credentials for live trading."""
        if self.execution_mode != "live":
            return

        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        from pathlib import Path

        self._http = httpx.AsyncClient(
            base_url=KALSHI_BASE_URL,
            timeout=15.0,
        )

        try:
            key_path = Path(KALSHI_PRIVATE_KEY_PATH)
            if key_path.exists():
                key_data = key_path.read_bytes()
                self._private_key = load_pem_private_key(key_data, password=None)
                from cryptography.hazmat.primitives.asymmetric import rsa as rsa_mod
                if isinstance(self._private_key, rsa_mod.RSAPrivateKey):
                    self._key_type = "rsa"
                logger.info("[Sniper] Loaded Kalshi private key")
        except Exception as e:
            logger.error(f"[Sniper] Failed to load key: {e}")

    def _auth_headers(self, method: str, path: str) -> dict:
        """Generate Kalshi auth headers."""
        import base64
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp_ms = int(time.time() * 1000)
        sign_path = path.split("?")[0]
        message = f"{timestamp_ms}{method}{sign_path}".encode("utf-8")

        if self._private_key and self._key_type == "rsa":
            signature = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
            sig_b64 = base64.b64encode(signature).decode("utf-8")
        else:
            sig_b64 = ""

        return {
            "KALSHI-ACCESS-KEY": KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
            "Content-Type": "application/json",
        }

    async def poll_all(self):
        """Poll all cities sequentially (staggered to avoid Kalshi 429 rate limits)."""
        for code, sniper in self.snipers.items():
            if sniper._is_peak_hour() or sniper._today == "":
                try:
                    await sniper.poll_and_act(
                        http_client=self._http,
                        auth_fn=self._auth_headers if self._private_key else None,
                    )
                except Exception as e:
                    logger.error(f"[Sniper/{code}] Poll error: {e}")
                # Stagger Kalshi API calls to avoid 429
                await asyncio.sleep(2)

    def get_status(self) -> dict:
        """Get status for all cities."""
        return {
            "execution_mode": self.execution_mode,
            "cities": {
                code: sniper.get_status()
                for code, sniper in self.snipers.items()
            },
        }


# ── Run loop ───────────────────────────────────────────────────────────

async def sniper_loop(execution_mode: str = "dry"):
    """Run the multi-city sniper as an independent async loop."""
    multi = MultiCitySniper(execution_mode)
    await multi.initialize()

    city_list = ", ".join(multi.snipers.keys())
    logger.info(f"[Sniper] Starting multi-city sniper (mode={execution_mode})")
    logger.info(f"[Sniper] Cities: {city_list}")
    logger.info(f"[Sniper] Poll interval: {POLL_INTERVAL_SECONDS}s per city")
    logger.info(f"[Sniper] Max buy price: ${MAX_BUY_PRICE:.2f}")

    # Log peak windows for today
    for code, sniper in multi.snipers.items():
        local = sniper._get_local_time()
        peak = sniper._peak_hours.get(local.month, (0, 0))
        tz_name = {-4: "ET", -5: "CT", -6: "MT", -7: "PT"}.get(sniper._utc_offset, "?")
        logger.info(f"[Sniper]   {code}: {peak[0]}:00-{peak[1]}:00 {tz_name}")

    while True:
        try:
            await multi.poll_all()
        except Exception as e:
            logger.error(f"[Sniper] Loop error: {e}", exc_info=True)
        await asyncio.sleep(5)  # Inner loop — rate limiting is per-city inside


# ── CLI entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    print(f"Starting Multi-City NWS Sniper in {mode} mode...")
    asyncio.run(sniper_loop(mode))
