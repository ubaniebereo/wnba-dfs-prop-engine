"""Fetch REAL sportsbook odds (DraftKings via ESPN) for upcoming games.

ESPN's scoreboard embeds a book's moneyline / spread / total with open & close
prices. We parse these into tidy market rows and convert American odds to
implied probabilities, with two-way de-vigging so model edges are measured
against the book's *fair* (vig-removed) probability.

No odds are fabricated. If a game has no posted line, it is simply skipped.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd

from src.api_espn import fetch_scoreboard
from src.utils import get_logger, parse_espn_date

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Odds math
# ---------------------------------------------------------------------------
def american_to_implied(odds: float) -> float:
    """American odds -> implied probability (includes vig)."""
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else (-o) / (-o + 100.0)


def american_to_decimal(odds: float) -> float:
    o = float(odds)
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / (-o))


def devig_two_way(imp_a: float, imp_b: float) -> tuple[float, float]:
    """Normalize a two-way market so the fair probabilities sum to 1."""
    s = imp_a + imp_b
    if s <= 0:
        return float("nan"), float("nan")
    return imp_a / s, imp_b / s


def _num(text) -> float | None:
    if text is None:
        return None
    m = re.search(r"-?\d+(\.\d+)?", str(text))
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _price(node: dict, which: str) -> float | None:
    """Pull a 'close' (current) odds value, falling back to 'open'."""
    if not node:
        return None
    for state in ("close", "open"):
        v = (node.get(state) or {}).get("odds")
        if v is not None:
            return _num(v)
    return None


def _line(node: dict) -> float | None:
    for state in ("close", "open"):
        v = (node.get(state) or {}).get("line")
        if v is not None:
            return _num(v)
    return None


def parse_event_odds(event: dict) -> list[dict]:
    """Return tidy market rows for one scoreboard event (or [] if no odds)."""
    comp = (event.get("competitions") or [{}])[0]
    odds_list = comp.get("odds") or []
    if not odds_list:
        return []
    o = odds_list[0]
    book = (o.get("provider") or {}).get("name", "book")
    gid = str(event.get("id"))
    gdate = parse_espn_date(event.get("date"))
    home = away = None
    for c in comp.get("competitors", []):
        nm = (c.get("team") or {}).get("abbreviation")
        if c.get("homeAway") == "home":
            home = nm
        else:
            away = nm

    rows: list[dict] = []

    def add(market, side, line, price, open_price):
        if price is None:
            return
        rows.append({"game_id": gid, "date": gdate, "home_team": home,
                     "away_team": away, "book": book, "market_type": market,
                     "side": side, "line_value": line, "odds": price,
                     "open_odds": open_price})

    ml = o.get("moneyline") or {}
    add("moneyline", "home", None, _price(ml.get("home"), "c"),
        _num((ml.get("home", {}).get("open") or {}).get("odds")))
    add("moneyline", "away", None, _price(ml.get("away"), "c"),
        _num((ml.get("away", {}).get("open") or {}).get("odds")))

    ps = o.get("pointSpread") or {}
    add("spread", "home", _line(ps.get("home")), _price(ps.get("home"), "c"),
        _num((ps.get("home", {}).get("open") or {}).get("odds")))
    add("spread", "away", _line(ps.get("away")), _price(ps.get("away"), "c"),
        _num((ps.get("away", {}).get("open") or {}).get("odds")))

    tot = o.get("total") or {}
    tline = _line(tot.get("over"))
    add("total", "over", tline, _price(tot.get("over"), "c"),
        _num((tot.get("over", {}).get("open") or {}).get("odds")))
    add("total", "under", _line(tot.get("under")), _price(tot.get("under"), "c"),
        _num((tot.get("under", {}).get("open") or {}).get("odds")))
    return rows


def fetch_upcoming_odds(days_ahead: int = 5) -> pd.DataFrame:
    """All posted markets for games over the next N days."""
    rows: list[dict] = []
    today = date.today()
    for i in range(days_ahead + 1):
        payload = fetch_scoreboard(today + timedelta(days=i))
        for ev in (payload or {}).get("events", []) or []:
            rows.extend(parse_event_odds(ev))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["implied_raw"] = df["odds"].apply(american_to_implied)
        df["decimal"] = df["odds"].apply(american_to_decimal)
        if "open_odds" in df:
            df["line_move"] = df.apply(
                lambda r: (american_to_implied(r["odds"]) -
                           american_to_implied(r["open_odds"]))
                if pd.notnull(r["open_odds"]) else 0.0, axis=1)
    return df
