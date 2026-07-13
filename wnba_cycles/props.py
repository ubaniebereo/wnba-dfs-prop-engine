"""Map flagged players to sportsbook player props and rank 'under' candidates.

For each red-flagged player who has a game on/near a predicted dip date, pull the
player-props market and surface the points (and PRA) line as an under candidate.
The bet edge is heuristic: how far the player's predicted-dip average sits BELOW
the posted line. Bigger gap below the line = stronger statistical 'under' lean.
"""

from __future__ import annotations

from dataclasses import dataclass

from .client import BDLClient, BDLError
from .analysis import CycleResult


@dataclass
class UnderCandidate:
    player: str
    game_id: int
    game_date: str
    market: str
    line: float
    dip_avg: float        # player's avg on predicted-dip games
    season_avg: float     # player's avg on normal games
    edge_vs_line: float   # season/dip avg minus line (negative => leans under)
    red_flag_score: float
    book: str = ""


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _extract_player_lines(props_rows: list[dict], player_name: str) -> list[dict]:
    """Best-effort parse of the props payload for one player's markets.

    The props schema varies; we match on player name appearing in the row and
    pull (market_type, line/value, book) triplets defensively.
    """
    out = []
    target = player_name.lower()
    for row in props_rows:
        blob = str(row).lower()
        if target.split()[-1] not in blob:   # match on last name at least
            continue
        market = (row.get("market") or row.get("market_type")
                  or row.get("type") or row.get("category") or "points")
        line = _num(row.get("line") or row.get("value")
                    or row.get("over_under") or row.get("handicap"))
        book = (row.get("book") or row.get("sportsbook")
                or row.get("bookmaker") or "")
        if line is not None:
            out.append({"market": str(market), "line": line, "book": str(book)})
    return out


def scan_unders(client: BDLClient, flagged: list[CycleResult],
                upcoming_games: list[dict], *, dip_window_days: int = 4
                ) -> list[UnderCandidate]:
    """flagged: red-flagged CycleResults. upcoming_games: list from the API."""
    # index upcoming games by participating team is non-trivial without roster
    # joins, so we scan every upcoming game's props and match players by name.
    flagged_by_name = {f.name.lower(): f for f in flagged if f.red_flag_score > 0}
    candidates: list[UnderCandidate] = []
    for g in upcoming_games:
        gid = g.get("id")
        gdate = (g.get("date") or "")[:10]
        try:
            props = client.player_props(gid)
        except BDLError:
            continue
        for name_lc, fr in flagged_by_name.items():
            # only bet the dip: game date must fall in a predicted dip window
            if fr.next_dip_dates and not any(
                abs((_d(gdate) - _d(dd)).days) <= dip_window_days
                for dd in fr.next_dip_dates if dd
            ):
                continue
            for line in _extract_player_lines(props, fr.name):
                market = line["market"]
                # compare the relevant predicted-dip average to the line
                if "point" in market.lower() or market.lower() in ("pts", "points"):
                    dip_avg, season_avg = fr.drop_mean_pts, fr.base_mean_pts
                else:
                    dip_avg, season_avg = fr.drop_mean_gmsc, fr.base_mean_gmsc
                edge = dip_avg - line["line"]
                candidates.append(UnderCandidate(
                    player=fr.name, game_id=gid, game_date=gdate,
                    market=market, line=line["line"], dip_avg=round(dip_avg, 1),
                    season_avg=round(season_avg, 1), edge_vs_line=round(edge, 1),
                    red_flag_score=round(fr.red_flag_score, 2), book=line["book"],
                ))
    # strongest under lean first (most negative edge), tie-break red-flag score
    candidates.sort(key=lambda c: (c.edge_vs_line, -c.red_flag_score))
    return candidates


def _d(s: str):
    from datetime import datetime
    return datetime.strptime(s[:10], "%Y-%m-%d")
