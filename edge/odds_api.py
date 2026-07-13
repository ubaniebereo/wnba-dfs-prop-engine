"""The Odds API client — real multi-book WNBA game odds + player props.

Free tier is 500 requests/month; the client logs `x-requests-remaining` after
every call so quota is visible. Game odds cost 1 request per market; player
props cost 1 per market per event. The events list is free.

Set the key in the ODDS_API_KEY environment variable (or .env).
"""

from __future__ import annotations

import os
import re
import unicodedata

import pandas as pd
import requests

from src.utils import get_logger

log = get_logger(__name__)
BASE = "https://api.the-odds-api.com/v4/sports/basketball_wnba"

# Odds API full team name -> our ESPN abbreviation (for game-odds matching).
TEAM_ABBR = {
    "Atlanta Dream": "ATL", "Chicago Sky": "CHI", "Connecticut Sun": "CON",
    "Dallas Wings": "DAL", "Golden State Valkyries": "GS", "Indiana Fever": "IND",
    "Las Vegas Aces": "LV", "Los Angeles Sparks": "LA", "Minnesota Lynx": "MIN",
    "New York Liberty": "NY", "Phoenix Mercury": "PHX", "Portland Fire": "POR",
    "Seattle Storm": "SEA", "Toronto Tempo": "TOR", "Washington Mystics": "WSH",
}
PROP_MARKETS = "player_points,player_rebounds,player_assists"


def _key() -> str:
    k = os.environ.get("ODDS_API_KEY", "")
    if not k:
        raise RuntimeError("ODDS_API_KEY not set (put it in .env).")
    return k


def _get(path: str, params: dict):
    params = dict(params)
    params["apiKey"] = _key()
    r = requests.get(f"{BASE}{path}", params=params, timeout=30)
    rem = r.headers.get("x-requests-remaining")
    if rem is not None:
        log.info("Odds API: %s requests remaining (last cost %s)",
                 rem, r.headers.get("x-requests-last"))
    r.raise_for_status()
    return r.json()


def norm_name(s: str) -> str:
    """Normalize a player name for matching across feeds."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Game odds (multi-book)
# ---------------------------------------------------------------------------
def fetch_game_odds(regions="us", markets="h2h,spreads,totals") -> pd.DataFrame:
    data = _get("/odds/", {"regions": regions, "markets": markets,
                           "oddsFormat": "american"})
    rows = []
    for g in data:
        home, away = g["home_team"], g["away_team"]
        gid, gdate = g["id"], g["commence_time"][:10]
        for bk in g.get("bookmakers", []):
            for m in bk.get("markets", []):
                mk = m["key"]
                for o in m.get("outcomes", []):
                    if mk == "h2h":
                        market, side = "moneyline", ("home" if o["name"] == home else "away")
                        line = None
                    elif mk == "spreads":
                        market, side = "spread", ("home" if o["name"] == home else "away")
                        line = o.get("point")
                    elif mk == "totals":
                        market, side = "total", o["name"].lower()  # over/under
                        line = o.get("point")
                    else:
                        continue
                    rows.append({"game_id": gid, "date": gdate,
                                 "home_team": TEAM_ABBR.get(home, home),
                                 "away_team": TEAM_ABBR.get(away, away),
                                 "book": bk["key"], "market_type": market,
                                 "side": side, "line_value": line,
                                 "odds": float(o["price"])})
    return pd.DataFrame(rows)


def best_line_per_side(odds: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-book rows to the BEST available price per side (line shop)."""
    if odds.empty:
        return odds
    from .odds_espn import american_to_decimal, american_to_implied
    idx = odds.groupby(["game_id", "market_type", "side"])["odds"].idxmax()
    best = odds.loc[idx].copy()
    best["decimal"] = best["odds"].apply(american_to_decimal)
    best["implied_raw"] = best["odds"].apply(american_to_implied)
    best["line_move"] = 0.0
    return best.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Player props
# ---------------------------------------------------------------------------
def fetch_player_props(regions="us", markets=PROP_MARKETS) -> pd.DataFrame:
    events = _get("/events/", {})  # free
    rows = []
    for e in events:
        eid, gdate = e["id"], e["commence_time"][:10]
        try:
            d = _get(f"/events/{eid}/odds/", {"regions": regions, "markets": markets,
                                              "oddsFormat": "american"})
        except requests.RequestException as exc:
            log.warning("props fetch failed for %s: %s", eid, exc)
            continue
        for bk in d.get("bookmakers", []):
            for m in bk.get("markets", []):
                stat = m["key"].replace("player_", "")  # points/rebounds/assists
                for o in m.get("outcomes", []):
                    rows.append({"event_id": eid, "date": gdate, "book": bk["key"],
                                 "market": stat, "player": o.get("description"),
                                 "side": o["name"].lower(), "line": o.get("point"),
                                 "odds": float(o["price"])})
    return pd.DataFrame(rows)
