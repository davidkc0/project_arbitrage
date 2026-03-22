"""IEM Historical Data Fetcher — pulls hourly METAR data from Iowa Environmental Mesonet.

Data source: Iowa State University IEM ASOS archive (free, no API key)
Station: KAUS (Austin-Bergstrom International Airport)
Fields: temperature (°F), wind speed (knots), sky cover (CLR/FEW/SCT/BKN/OVC)

Used to build the intraday temperature prediction model.
"""

from __future__ import annotations

import csv
import io
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DATA_DIR = Path(__file__).parent / "data"
HISTORY_FILE = DATA_DIR / "kaus_hourly_history.csv"
DAILY_SUMMARY_FILE = DATA_DIR / "kaus_daily_summary.csv"

# Austin-Bergstrom coordinates
KAUS_LAT = 30.1975
KAUS_LON = -97.6664

# CSV headers
HOURLY_HEADERS = [
    "date", "hour", "temp_f", "wind_kt", "sky_cover", "daylight_hours",
]
DAILY_HEADERS = [
    "date", "daily_high", "daily_low", "peak_hour", "sky_dominant",
    "wind_avg_kt", "daylight_hours", "month", "day_of_year",
]


def compute_daylight_hours(dt: date, lat: float = KAUS_LAT) -> float:
    """
    Calculate hours of daylight for a given date and latitude.

    Uses the standard solar declination + hour angle formula.
    Returns hours of daylight (e.g., 10.5 in December, 14.0 in June for Austin).
    """
    day_of_year = dt.timetuple().tm_yday

    # Solar declination angle (radians)
    declination = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
    declination_rad = math.radians(declination)
    lat_rad = math.radians(lat)

    # Hour angle at sunrise/sunset
    cos_hour_angle = -math.tan(lat_rad) * math.tan(declination_rad)

    # Clamp for polar cases (shouldn't happen at Austin's latitude)
    cos_hour_angle = max(-1.0, min(1.0, cos_hour_angle))

    hour_angle = math.acos(cos_hour_angle)

    # Daylight hours = 2 * hour_angle / 15 (degrees per hour)
    daylight = 2 * math.degrees(hour_angle) / 15

    return round(daylight, 2)


async def fetch_iem_range(
    station: str,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Fetch hourly METAR observations from IEM for a date range.

    Returns list of {date, hour, temp_f, wind_kt, sky_cover} dicts.
    """
    params = {
        "station": station,
        "data": ["tmpf", "sknt", "skyc1", "skyc2"],
        "year1": start_date.year,
        "month1": start_date.month,
        "day1": start_date.day,
        "year2": end_date.year,
        "month2": end_date.month,
        "day2": end_date.day,
        "tz": "America/Chicago",
        "format": "comma",
        "latlon": "no",
        "elev": "no",
        "missing": "null",
        "trace": "null",
        "direct": "no",
        "report_type": "3",  # All report types
    }

    # IEM uses repeated params for 'data' field
    param_str = "&".join(
        f"{k}={v}" if k != "data" else "&".join(f"data={d}" for d in v)
        for k, v in params.items()
    )
    url = f"{IEM_BASE}?{param_str}"

    async with httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.get(url)
        resp.raise_for_status()
        text = resp.text

    rows = []
    reader = csv.DictReader(
        (line for line in text.splitlines() if not line.startswith("#")),
    )
    for row in reader:
        try:
            valid = row.get("valid", "")
            temp_str = row.get("tmpf", "null")
            wind_str = row.get("sknt", "null")

            if temp_str == "null" or not valid:
                continue

            dt = datetime.strptime(valid, "%Y-%m-%d %H:%M")
            temp_f = round(float(temp_str))
            wind_kt = float(wind_str) if wind_str != "null" else 0.0

            # Combine sky cover layers into dominant condition
            sky1 = row.get("skyc1", "null")
            sky2 = row.get("skyc2", "null")
            sky = _dominant_sky(sky1, sky2)

            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "hour": dt.hour,
                "temp_f": temp_f,
                "wind_kt": round(wind_kt, 1),
                "sky_cover": sky,
            })
        except (ValueError, KeyError) as e:
            continue

    logger.info(f"[IEM] Fetched {len(rows)} hourly observations for {station} "
                f"({start_date} → {end_date})")
    return rows


def _dominant_sky(sky1: str, sky2: str) -> str:
    """Pick the most significant sky cover from two layers."""
    order = {"OVC": 5, "BKN": 4, "SCT": 3, "FEW": 2, "CLR": 1, "null": 0, "": 0}
    s1 = order.get(sky1, 0)
    s2 = order.get(sky2, 0)
    if s1 >= s2:
        return sky1 if sky1 not in ("null", "") else "CLR"
    return sky2 if sky2 not in ("null", "") else "CLR"


def build_daily_summary(hourly_rows: list[dict]) -> list[dict]:
    """
    From hourly observations, compute one summary row per day:
    - daily_high, daily_low, peak_hour
    - dominant sky condition (mode of daytime readings 8 AM - 7 PM)
    - average daytime wind speed
    - daylight hours (computed from date)
    """
    from collections import Counter, defaultdict

    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in hourly_rows:
        by_date[row["date"]].append(row)

    summaries = []
    for date_str in sorted(by_date.keys()):
        readings = by_date[date_str]
        if not readings:
            continue

        temps = [r["temp_f"] for r in readings]
        daily_high = max(temps)
        daily_low = min(temps)

        # Find peak hour (hour with highest temp)
        peak_reading = max(readings, key=lambda r: r["temp_f"])
        peak_hour = peak_reading["hour"]

        # Daytime readings only (8 AM - 7 PM) for sky and wind
        daytime = [r for r in readings if 8 <= r["hour"] <= 19]
        if daytime:
            sky_counts = Counter(r["sky_cover"] for r in daytime)
            sky_dominant = sky_counts.most_common(1)[0][0]
            wind_avg = round(sum(r["wind_kt"] for r in daytime) / len(daytime), 1)
        else:
            sky_dominant = "CLR"
            wind_avg = 0.0

        dt = date.fromisoformat(date_str)
        daylight = compute_daylight_hours(dt)

        summaries.append({
            "date": date_str,
            "daily_high": daily_high,
            "daily_low": daily_low,
            "peak_hour": peak_hour,
            "sky_dominant": sky_dominant,
            "wind_avg_kt": wind_avg,
            "daylight_hours": daylight,
            "month": dt.month,
            "day_of_year": dt.timetuple().tm_yday,
        })

    return summaries


def save_hourly_csv(rows: list[dict], path: Path = HISTORY_FILE) -> None:
    """Save hourly observations to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add daylight hours to each row
    daylight_cache: dict[str, float] = {}
    for row in rows:
        d = row["date"]
        if d not in daylight_cache:
            daylight_cache[d] = compute_daylight_hours(date.fromisoformat(d))
        row["daylight_hours"] = daylight_cache[d]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HOURLY_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"[IEM] Saved {len(rows)} hourly rows to {path}")


def save_daily_csv(summaries: list[dict], path: Path = DAILY_SUMMARY_FILE) -> None:
    """Save daily summaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DAILY_HEADERS)
        writer.writeheader()
        writer.writerows(summaries)

    logger.info(f"[IEM] Saved {len(summaries)} daily summaries to {path}")


def load_daily_summary(path: Path = DAILY_SUMMARY_FILE) -> list[dict]:
    """Load daily summaries from CSV."""
    if not path.exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["daily_high"] = int(float(row["daily_high"]))
            row["daily_low"] = int(float(row["daily_low"]))
            row["peak_hour"] = int(row["peak_hour"])
            row["wind_avg_kt"] = float(row["wind_avg_kt"])
            row["daylight_hours"] = float(row["daylight_hours"])
            row["month"] = int(row["month"])
            row["day_of_year"] = int(row["day_of_year"])
            rows.append(row)
    return rows


async def fetch_and_build_history(
    station: str = "AUS",
    years_back: int = 5,
) -> tuple[list[dict], list[dict]]:
    """
    Full pipeline: fetch IEM data, compute daily summaries, save both CSVs.

    Fetches in 6-month chunks to avoid IEM timeouts.
    Returns (hourly_rows, daily_summaries).
    """
    end = date.today()
    start = date(end.year - years_back, 1, 1)

    logger.info(f"[IEM] Fetching {years_back} years of data: {start} → {end}")

    all_hourly: list[dict] = []

    # Fetch in 6-month chunks to avoid timeout
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=180), end)
        logger.info(f"[IEM] Fetching chunk: {chunk_start} → {chunk_end}")

        try:
            chunk = await fetch_iem_range(station, chunk_start, chunk_end)
            all_hourly.extend(chunk)
        except Exception as e:
            logger.error(f"[IEM] Failed chunk {chunk_start} → {chunk_end}: {e}")

        chunk_start = chunk_end + timedelta(days=1)

    # Deduplicate by (date, hour)
    seen = set()
    unique = []
    for row in all_hourly:
        key = (row["date"], row["hour"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    all_hourly = unique

    logger.info(f"[IEM] Total unique hourly observations: {len(all_hourly)}")

    # Build daily summaries
    daily = build_daily_summary(all_hourly)

    # Save both
    save_hourly_csv(all_hourly)
    save_daily_csv(daily)

    return all_hourly, daily
