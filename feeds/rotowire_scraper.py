"""Rotowire WNBA lineups/injuries (Stage 6, Sec 5B) — projected source.

Strategy order: official structured data -> hidden JSON -> static HTML -> (manual
Playwright fallback). Rotowire's WNBA pages are JS-heavy; we try requests+BS4 and
classify viability honestly rather than build brittle DOM scraping. If the page
is not reliably parseable, `probe()` returns viable=False and the scheduler skips it.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from .base import BaseClient
from .storage import now_iso
from src.utils import get_logger

log = get_logger(__name__)

# candidate URLs (the league path has shifted historically — probe several)
CANDIDATE_URLS = [
    "https://www.rotowire.com/basketball/injury-report.php?league=WNBA",
    "https://www.rotowire.com/wnba/injury-report.php",
    "https://www.rotowire.com/basketball/wnba/injuries.php",
]


class RotowireScraper(BaseClient):
    name = "rotowire"

    def __init__(self):
        super().__init__(min_interval=120.0, timeout=25.0)   # slow + polite

    def probe(self) -> dict:
        """Find a working URL and judge whether structured rows are parseable."""
        for url in CANDIDATE_URLS:
            res = self.get(url, as_json=False)
            if not res.ok:
                continue
            soup = BeautifulSoup(res.data, "lxml")
            tables = soup.find_all("table")
            # Rotowire often renders via JS; check for embedded JSON too
            has_json = bool(re.search(r'(injuries|lineups)\s*[:=]\s*[\[{]', res.data))
            if tables or has_json:
                self.viable = True
                return {"viable": True, "url": url, "tables": len(tables),
                        "embedded_json": has_json}
        self.viable = False
        return {"viable": False,
                "recommendation": "JS-rendered / path-shifted; use ESPN + X for news, "
                                  "or add a Playwright fallback if Rotowire becomes essential."}

    def fetch_injuries(self) -> list[dict]:
        info = self.probe()
        if not info.get("viable"):
            log.info("rotowire not viable for static scraping: %s", info.get("recommendation"))
            return []
        res = self.get(info["url"], as_json=False)
        soup = BeautifulSoup(res.data, "lxml")
        rows = []
        for tr in soup.select("table tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if len(cells) >= 2 and cells[0]:
                rows.append({"source": "rotowire", "captured_at": now_iso(),
                             "player_name": cells[0], "raw_text": " | ".join(cells),
                             "projected_vs_confirmed": "projected", "metadata_json": {}})
        return rows
