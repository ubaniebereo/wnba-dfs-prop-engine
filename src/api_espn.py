"""ESPN public/unofficial WNBA endpoints + JSON -> flat-dataframe normalization.

Two endpoints are used:
  * scoreboard?dates=YYYYMMDD  -> games on a date (event ids, teams, scores, status)
  * summary?event=<id>         -> per-game player and team box scores

Nothing here writes to the database; functions return pandas DataFrames so the
ingestion layer stays in control of persistence.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from . import config
from .utils import (get_logger, http_get_json, parse_espn_date,
                    split_made_attempted, to_float, to_int, yyyymmdd)

log = get_logger(__name__)

# ESPN box-score columns arrive as a positional list aligned to these labels.
_STAT_LABELS = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO",
                "STL", "BLK", "OREB", "DREB", "PF", "+/-"]


# ---------------------------------------------------------------------------
# Raw fetchers
# ---------------------------------------------------------------------------
def fetch_scoreboard(day: date) -> dict | None:
    """Raw scoreboard JSON for a single date."""
    return http_get_json(config.ESPN_SCOREBOARD, {"dates": yyyymmdd(day)})


def fetch_summary(event_id: str) -> dict | None:
    """Raw summary (box score) JSON for one event."""
    return http_get_json(config.ESPN_SUMMARY, {"event": event_id})


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------
def scoreboard_to_games(payload: dict | None) -> pd.DataFrame:
    """Flatten a scoreboard payload into one row per game.

    Columns: game_id, game_date, season, status, completed,
             home_team_id, home_team, home_score,
             away_team_id, away_team, away_score
    """
    rows: list[dict] = []
    for event in (payload or {}).get("events", []) or []:
        comps = event.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        status = (comp.get("status") or {}).get("type", {}) or {}
        home = away = {}
        for c in comp.get("competitors", []):
            if c.get("homeAway") == "home":
                home = c
            elif c.get("homeAway") == "away":
                away = c
        if not home or not away:
            continue
        rows.append({
            "game_id": str(event.get("id")),
            "game_date": parse_espn_date(event.get("date")),
            "season": int((event.get("season") or {}).get("year") or 0) or None,
            "status": status.get("name"),
            "completed": bool(status.get("completed")),
            "home_team_id": str((home.get("team") or {}).get("id")),
            "home_team": (home.get("team") or {}).get("abbreviation"),
            "home_score": to_int(home.get("score"), None),
            "away_team_id": str((away.get("team") or {}).get("id")),
            "away_team": (away.get("team") or {}).get("abbreviation"),
            "away_score": to_int(away.get("score"), None),
        })
    return pd.DataFrame(rows)


def _athlete_rows(team_block: dict, opp_abbr: str, opp_id: str,
                  game_id: str, game_date: str, is_home: bool) -> list[dict]:
    team = team_block.get("team") or {}
    stat_groups = team_block.get("statistics") or []
    if not stat_groups:
        return []
    group = stat_groups[0]
    labels = group.get("labels") or _STAT_LABELS
    rows = []
    for ath in group.get("athletes", []) or []:
        athlete = ath.get("athlete") or {}
        if ath.get("didNotPlay") or not ath.get("stats"):
            continue
        values = dict(zip(labels, ath.get("stats", [])))
        fgm, fga = split_made_attempted(values.get("FG"))
        tpm, tpa = split_made_attempted(values.get("3PT"))
        ftm, fta = split_made_attempted(values.get("FT"))
        rows.append({
            "stat_id": f"{game_id}_{athlete.get('id')}",
            "game_id": game_id,
            "game_date": game_date,
            "player_id": str(athlete.get("id")),
            "player_name": athlete.get("displayName"),
            "team_id": str(team.get("id")),
            "team": team.get("abbreviation"),
            "opponent_id": str(opp_id),
            "opponent": opp_abbr,
            "is_home": int(is_home),
            "starter": int(bool(ath.get("starter"))),
            "minutes": to_int(values.get("MIN"), 0),
            "points": to_int(values.get("PTS"), 0),
            "fgm": fgm, "fga": fga,
            "tpm": tpm, "tpa": tpa,
            "ftm": ftm, "fta": fta,
            "rebounds": to_int(values.get("REB"), 0),
            "oreb": to_int(values.get("OREB"), 0),
            "dreb": to_int(values.get("DREB"), 0),
            "assists": to_int(values.get("AST"), 0),
            "turnovers": to_int(values.get("TO"), 0),
            "steals": to_int(values.get("STL"), 0),
            "blocks": to_int(values.get("BLK"), 0),
            "fouls": to_int(values.get("PF"), 0),
            "plus_minus": to_float(values.get("+/-"), 0.0),
        })
    return rows


def summary_to_player_stats(payload: dict | None, game_id: str,
                            game_date: str) -> pd.DataFrame:
    """Flatten a summary payload into one row per player-game.

    Unique key = stat_id (game_id + '_' + player_id).
    """
    boxscore = (payload or {}).get("boxscore") or {}
    team_blocks = boxscore.get("players") or []
    if len(team_blocks) != 2:
        return pd.DataFrame()
    # determine which block is home using the header competition
    home_abbr = _home_abbr_from_header(payload)
    rows: list[dict] = []
    abbrs = [(tb.get("team") or {}).get("abbreviation") for tb in team_blocks]
    ids = [(tb.get("team") or {}).get("id") for tb in team_blocks]
    for i, block in enumerate(team_blocks):
        opp_i = 1 - i
        is_home = abbrs[i] == home_abbr if home_abbr else (i == 0)
        rows.extend(_athlete_rows(block, abbrs[opp_i], ids[opp_i],
                                  game_id, game_date, is_home))
    return pd.DataFrame(rows)


def summary_to_team_stats(payload: dict | None, game_id: str,
                          game_date: str) -> pd.DataFrame:
    """One row per team-game with points scored / allowed (pace proxy inputs)."""
    boxscore = (payload or {}).get("boxscore") or {}
    team_blocks = boxscore.get("teams") or []
    if len(team_blocks) != 2:
        return pd.DataFrame()
    # pull team point totals from the player block totals when available
    players = boxscore.get("players") or []
    points = {}
    for tb in players:
        abbr = (tb.get("team") or {}).get("abbreviation")
        groups = tb.get("statistics") or []
        if groups:
            labels = groups[0].get("labels") or _STAT_LABELS
            totals = dict(zip(labels, groups[0].get("totals", [])))
            points[abbr] = to_int(totals.get("PTS"), None)
    home_abbr = _home_abbr_from_header(payload)
    rows = []
    abbrs = [(tb.get("team") or {}).get("abbreviation") for tb in team_blocks]
    ids = [(tb.get("team") or {}).get("id") for tb in team_blocks]
    for i, tb in enumerate(team_blocks):
        opp_i = 1 - i
        pf = points.get(abbrs[i])
        pa = points.get(abbrs[opp_i])
        rows.append({
            "team_game_id": f"{game_id}_{ids[i]}",
            "game_id": game_id,
            "game_date": game_date,
            "team_id": str(ids[i]),
            "team": abbrs[i],
            "opponent_id": str(ids[opp_i]),
            "opponent": abbrs[opp_i],
            "is_home": int(abbrs[i] == home_abbr) if home_abbr else int(i == 0),
            "points_for": pf,
            "points_against": pa,
        })
    return pd.DataFrame(rows)


def _home_abbr_from_header(payload: dict | None) -> str | None:
    """Find the home team's abbreviation from the summary header block."""
    header = (payload or {}).get("header") or {}
    comps = header.get("competitions") or []
    if not comps:
        return None
    for c in comps[0].get("competitors", []):
        if c.get("homeAway") == "home":
            return (c.get("team") or {}).get("abbreviation")
    return None
