"""Source-agnostic normalization (Stage 6, Sec 6).

Standardizes player names, team abbreviations, market types and sides across X,
Rotowire, ESPN, The Odds API and SGP outputs. Name matching: exact -> alias ->
fuzzy (rapidfuzz), with unresolved matches logged for review.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
from rapidfuzz import process, fuzz

from src.database import get_engine
from src.utils import get_logger

log = get_logger(__name__)

MARKET_ALIASES = {
    "points": "player_points", "pts": "player_points", "player_points": "player_points",
    "rebounds": "player_rebounds", "reb": "player_rebounds", "player_rebounds": "player_rebounds",
    "assists": "player_assists", "ast": "player_assists", "player_assists": "player_assists",
}
SIDE_ALIASES = {"over": "over", "o": "over", "under": "under", "u": "under",
                "yes": "yes", "no": "no"}
TEAM_ALIASES = {  # full name -> our abbreviation
    "atlanta dream": "ATL", "chicago sky": "CHI", "connecticut sun": "CON",
    "dallas wings": "DAL", "golden state valkyries": "GS", "indiana fever": "IND",
    "las vegas aces": "LV", "los angeles sparks": "LA", "minnesota lynx": "MIN",
    "new york liberty": "NY", "phoenix mercury": "PHX", "portland fire": "POR",
    "seattle storm": "SEA", "toronto tempo": "TOR", "washington mystics": "WSH",
}
PLAYER_ALIASES: dict[str, str] = {}   # extend as unresolved names are reviewed


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_market(m: str) -> str | None:
    return MARKET_ALIASES.get(str(m).lower().strip())


def norm_side(s: str) -> str | None:
    return SIDE_ALIASES.get(str(s).lower().strip())


def norm_team(t: str) -> str:
    return TEAM_ALIASES.get(str(t).lower().strip(), str(t).upper()[:3])


class PlayerResolver:
    """Resolve free-text names to canonical BDL player_id via exact/alias/fuzzy."""

    def __init__(self, threshold=88):
        roster = pd.read_sql(
            "SELECT DISTINCT player_id, player_name FROM player_game_stats", get_engine())
        roster["key"] = roster["player_name"].map(norm_name)
        self.by_key = roster.drop_duplicates("key").set_index("key")
        self.keys = list(self.by_key.index)
        self.threshold = threshold
        self.unresolved: list[str] = []

    def resolve(self, name: str) -> dict | None:
        key = PLAYER_ALIASES.get(norm_name(name), norm_name(name))
        if key in self.by_key.index:
            r = self.by_key.loc[key]
            return {"player_id": r["player_id"], "player_name": r["player_name"], "match": "exact"}
        hit = process.extractOne(key, self.keys, scorer=fuzz.WRatio,
                                 score_cutoff=self.threshold)
        if hit:
            r = self.by_key.loc[hit[0]]
            return {"player_id": r["player_id"], "player_name": r["player_name"],
                    "match": f"fuzzy:{hit[1]:.0f}"}
        self.unresolved.append(name)
        log.debug("unresolved player name: %s", name)
        return None
