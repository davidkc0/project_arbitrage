"""IEM Daily Summary Message (DSM) poller.

Polls the Iowa Environmental Mesonet for the ASOS Daily Summary,
which is the same data source that feeds the NWS Daily Climate Report
used by Kalshi for settlement.

API: https://mesonet.agron.iastate.edu/api/1/daily.json
No auth required. Free. Returns JSON.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import httpx

logger = logging.getLogger(__name__)

IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/api/1/daily.json"


class IEMPoller:
    """Polls IEM for the ASOS Daily Summary Message (DSM) for KAUS."""

    def __init__(self, station: str = "AUS", network: str = "TX_ASOS"):
        self.station = station
        self.network = network
        self.last_max: int | None = None
        self.last_min: int | None = None
        self.last_poll_time: str | None = None
        self.is_estimated: bool = False

    async def poll_once(self, date_str: str | None = None) -> dict | None:
        """Fetch today's DSM from IEM.

        Returns dict with max_tmpf, min_tmpf, and metadata, or None on error.
        """
        if date_str is None:
            # Use CDT (UTC - 5)
            now_cdt = datetime.now(timezone.utc) - timedelta(hours=5)
            date_str = now_cdt.strftime("%Y-%m-%d")

        try:
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(IEM_DAILY_URL, params={
                    "station": self.station,
                    "network": self.network,
                    "date": date_str,
                })
                resp.raise_for_status()
                data = resp.json()

                rows = data.get("data", [])
                if not rows:
                    logger.warning("[IEM] No DSM data returned for %s", date_str)
                    return None

                row = rows[0]
                max_tmpf = row.get("max_tmpf")
                min_tmpf = row.get("min_tmpf")
                is_est = row.get("tmpf_est", False)

                if max_tmpf is not None:
                    self.last_max = int(round(max_tmpf))
                    self.last_min = int(round(min_tmpf)) if min_tmpf else None
                    self.is_estimated = is_est
                    self.last_poll_time = datetime.now(timezone.utc).isoformat()

                    logger.info(
                        "[IEM] DSM %s — high=%d°F low=%s°F%s",
                        date_str,
                        self.last_max,
                        self.last_min if self.last_min else "?",
                        " (estimated)" if is_est else "",
                    )

                    return {
                        "date": date_str,
                        "max_tmpf": max_tmpf,
                        "max_tmpf_int": self.last_max,
                        "min_tmpf": min_tmpf,
                        "is_estimated": is_est,
                        "station": self.station,
                        "source": "IEM_DSM",
                    }
                else:
                    logger.warning("[IEM] DSM returned null max_tmpf for %s", date_str)
                    return None

        except Exception as e:
            logger.warning("[IEM] Poll error: %s", e)
            return None

    def get_settlement_high(self) -> int | None:
        """Return the current DSM high — this is likely what Kalshi will settle on."""
        return self.last_max
