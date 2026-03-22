"""Synoptic Data API poller — pulls KAUS temperature data and archives it.

Polls every 60 seconds during trading hours, archives all readings to CSV
for post-trial analysis. Falls back to NWS 5-min feed if Synoptic is down.

Data is archived to weather_bets/data/synoptic_archive/ with daily CSV files.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SYNOPTIC_TOKEN = os.getenv("SYNOPTIC_API_TOKEN", "")
SYNOPTIC_BASE = "https://api.synopticdata.com/v2"
STATION_ID = "KAUS"
ARCHIVE_DIR = Path(__file__).parent / "data" / "synoptic_archive"


class SynopticPoller:
    """Polls Synoptic Data API for KAUS temperature readings.

    Archives every reading to daily CSV files for post-trial analysis.
    """

    def __init__(self, token: str = SYNOPTIC_TOKEN, station: str = STATION_ID):
        self.token = token
        self.station = station
        self.archive_dir = ARCHIVE_DIR
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self.readings: list[dict] = []  # Today's readings
        self.latest: dict | None = None  # Most recent reading
        self.morning_low: int | None = None
        self.day_max: int | None = None

    def _archive_path(self, date_str: str) -> Path:
        """Get archive CSV path for a given date."""
        return self.archive_dir / f"kaus_{date_str}.csv"

    def _ensure_archive_header(self, path: Path):
        """Create CSV with header if it doesn't exist."""
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc", "timestamp_cdt", "temp_f", "temp_c",
                    "station", "source",
                ])

    async def poll_once(self) -> dict | None:
        """Fetch the latest temperature reading from Synoptic.

        Returns dict with temp_f, temp_c, timestamp, or None on failure.
        """
        if not self.token:
            logger.warning("[Synoptic] No API token configured")
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                resp = await http.get(
                    f"{SYNOPTIC_BASE}/stations/latest",
                    params={
                        "token": self.token,
                        "stid": self.station,
                        "vars": "air_temp",
                        "units": "english",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            stations = data.get("STATION", [])
            if not stations:
                logger.warning("[Synoptic] No station data returned")
                return None

            obs = stations[0].get("OBSERVATIONS", {})
            temp_entry = obs.get("air_temp_value_1", {})
            temp_f = temp_entry.get("value")
            ts_str = temp_entry.get("date_time", "")

            if temp_f is None:
                logger.warning("[Synoptic] No temperature value in response")
                return None

            # Parse UTC timestamp
            ts_utc = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            # Convert to CDT (UTC - 5)
            from datetime import timedelta
            ts_cdt = ts_utc - timedelta(hours=5)

            # Convert to Celsius for rounding analysis
            temp_c = round((temp_f - 32) * 5 / 9)

            reading = {
                "temp_f": round(temp_f, 1),
                "temp_f_int": round(temp_f),
                "temp_c": temp_c,
                "timestamp_utc": ts_utc.isoformat(),
                "timestamp_cdt": ts_cdt.strftime("%Y-%m-%d %H:%M"),
                "date": ts_cdt.strftime("%Y-%m-%d"),
                "hour": ts_cdt.hour,
                "minute": ts_cdt.minute,
                "station": self.station,
                "source": "synoptic",
            }

            # Update state
            self.latest = reading
            self.readings.append(reading)

            # Track day max / morning low
            t = reading["temp_f_int"]
            if self.day_max is None or t > self.day_max:
                self.day_max = t
            if reading["hour"] <= 10:
                if self.morning_low is None or t < self.morning_low:
                    self.morning_low = t

            # Archive to CSV
            self._archive_reading(reading)

            logger.info(
                f"[Synoptic] {reading['timestamp_cdt']} — "
                f"{reading['temp_f']}°F ({temp_c}°C) | "
                f"day max: {self.day_max}°F"
            )

            return reading

        except httpx.HTTPError as e:
            logger.error(f"[Synoptic] HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"[Synoptic] Error polling: {e}", exc_info=True)
            return None

    def _archive_reading(self, reading: dict):
        """Append a reading to the daily archive CSV."""
        path = self._archive_path(reading["date"])
        self._ensure_archive_header(path)

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                reading["timestamp_utc"],
                reading["timestamp_cdt"],
                reading["temp_f"],
                reading["temp_c"],
                reading["station"],
                reading["source"],
            ])

    def reset_daily(self):
        """Reset state for a new day."""
        self.readings = []
        self.latest = None
        self.morning_low = None
        self.day_max = None

    def get_trajectory(self) -> dict:
        """Compute trajectory features from today's readings so far."""
        if not self.readings:
            return {}

        temps = [r["temp_f_int"] for r in self.readings]
        hours = [r["hour"] + r["minute"] / 60 for r in self.readings]

        current = temps[-1]
        current_hour = hours[-1]

        # Rate of climb (°F per hour)
        rates = {}
        for lookback_h in [1, 2, 3]:
            cutoff_hour = current_hour - lookback_h
            past_temps = [
                t for t, h in zip(temps, hours) if h <= cutoff_hour
            ]
            if past_temps:
                rate = (current - past_temps[-1]) / lookback_h
                rates[f"rise_rate_{lookback_h}h"] = round(rate, 1)

        # 8 AM and 10 AM reference temps
        temp_at_8 = None
        temp_at_10 = None
        for r in self.readings:
            if r["hour"] == 8 and (temp_at_8 is None):
                temp_at_8 = r["temp_f_int"]
            if r["hour"] == 10 and (temp_at_10 is None):
                temp_at_10 = r["temp_f_int"]

        return {
            "current_temp": current,
            "current_hour": round(current_hour, 1),
            "morning_low": self.morning_low,
            "day_max": self.day_max,
            "total_rise": current - self.morning_low if self.morning_low else 0,
            "n_readings": len(self.readings),
            "temp_at_8am": temp_at_8,
            "temp_at_10am": temp_at_10,
            **rates,
        }

    def get_archive_dates(self) -> list[str]:
        """List all dates that have archived data."""
        files = sorted(self.archive_dir.glob("kaus_*.csv"))
        return [f.stem.replace("kaus_", "") for f in files]

    def load_archive(self, date_str: str) -> list[dict]:
        """Load archived readings for a specific date."""
        path = self._archive_path(date_str)
        if not path.exists():
            return []

        readings = []
        with open(path) as f:
            for row in csv.DictReader(f):
                readings.append(row)
        return readings


# ── Sync polling function (for testing) ──────────────────────────────

def poll_once_sync() -> dict | None:
    """Synchronous version of poll_once for quick testing."""
    import asyncio
    poller = SynopticPoller()
    return asyncio.run(poller.poll_once())


if __name__ == "__main__":
    """Quick test: fetch the latest reading."""
    logging.basicConfig(level=logging.INFO)
    result = poll_once_sync()
    if result:
        print(f"\n✅ Got reading: {result['temp_f']}°F at {result['timestamp_cdt']}")
        print(f"   Celsius: {result['temp_c']}°C")
        date_val = result["date"]
        print(f"   Archived to: {ARCHIVE_DIR / f'kaus_{date_val}.csv'}")
    else:
        print("❌ Failed to get reading")
