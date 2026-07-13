"""PrizePicks DFS pick'em lines (public JSON endpoint).

PrizePicks is a DFS pick'em app (more/less at fixed payout), NOT a sportsbook, so
it is absent from The Odds API. Its public projections endpoint returns current
WNBA lines (no auth). We normalize them so the engine can compare PrizePicks
lines to the SHARP sportsbook anchor — the standard DFS edge (PrizePicks lines
are often soft/slow). odds_type: standard | goblin (lower line) | demon (higher).

Underdog and Sleeper require auth/device tokens (see prizepicks notes); a paid
aggregator (OddsJam / OddsBlaze) is the one-stop for ALL DFS apps.
"""

from __future__ import annotations

import pandas as pd

from .base import BaseClient
from .normalizers import norm_market

WNBA_LEAGUE_ID = 3
URL = "https://api.prizepicks.com/projections"

# PrizePicks stat_type -> our market (core singles only; combos noted, not modeled)
STAT_MAP = {
    "Points": "points", "Rebounds": "rebounds", "Assists": "assists",
}
COMBO_STATS = {"Pts+Rebs+Asts", "Pts+Rebs", "Pts+Asts", "Rebs+Asts"}


class PrizePicksClient(BaseClient):
    name = "prizepicks"

    def __init__(self):
        super().__init__(min_interval=5.0, timeout=25.0)

    def _fetch_json(self, league_id: int) -> dict | None:
        """PrizePicks is Cloudflare-protected -> httpx/requests get 403.
        curl_cffi with Chrome TLS impersonation passes (verified)."""
        try:
            from curl_cffi import requests as creq
            r = creq.get(URL, params={"league_id": league_id, "per_page": 250},
                         impersonate="chrome", timeout=25)
            self.viable = r.status_code == 200
            return r.json() if r.status_code == 200 else None
        except Exception as exc:  # noqa: BLE001
            self.viable = False
            return None

    def fetch_wnba(self, include_alt=False, league_id: int = WNBA_LEAGUE_ID) -> pd.DataFrame:
        """Current WNBA projections -> tidy rows (player, market, line, odds_type)."""
        d = self._fetch_json(league_id)
        if not d:
            return pd.DataFrame()
        players = {i["id"]: i["attributes"].get("name")
                   for i in d.get("included", []) if i.get("type") == "new_player"}
        teams = {i["id"]: i["attributes"].get("team")
                 for i in d.get("included", []) if i.get("type") == "new_player"}
        from .prop_taxonomy import SIMULATABLE_COMBOS, normalize_label
        rows = []
        for p in d.get("data", []):
            a = p.get("attributes", {})
            stat = a.get("stat_type")
            canon = normalize_label(stat)          # canonical name from taxonomy
            pid = (p.get("relationships", {}).get("new_player", {})
                   .get("data", {}) or {}).get("id")
            rows.append({
                "book": "prizepicks", "player": players.get(pid),
                "team": teams.get(pid), "stat_type": stat,
                "market": canon, "is_combo": canon in SIMULATABLE_COMBOS,
                "line": a.get("line_score"), "odds_type": a.get("odds_type", "standard"),
                "start_time": a.get("start_time"),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df[df["player"].notna() & df["line"].notna()]
        if not include_alt:
            df = df[df["odds_type"] == "standard"]   # standard lines for fair comparison
        return df.reset_index(drop=True)


def fetch_prizepicks_wnba() -> pd.DataFrame:
    # include all line types (standard / goblin / demon) so the board shows every
    # available PrizePicks line; odds_type is surfaced so you can tell them apart.
    return PrizePicksClient().fetch_wnba(include_alt=True)
