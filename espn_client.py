"""ESPN public WNBA client (Stage 3, Section 1).

Provides confirmed starters / actives, which the box-score feeds give per game.
  scoreboard?dates=YYYYMMDD     -> events + competitors (live schedule)
  summary?event=ID              -> boxscore.players[].statistics[0].athletes[]
                                   each with athlete{id,displayName}, starter,
                                   didNotPlay  (=> is_starter / is_active)
"""

from __future__ import annotations

from datetime import date

from src.utils import get_logger, http_get_json

log = get_logger(__name__)
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"


def get_scoreboard(d: date | str) -> dict | None:
    ds = d if isinstance(d, str) else d.strftime("%Y%m%d")
    return http_get_json(f"{SITE}/scoreboard", {"dates": ds})


def get_event_summary(event_id: str) -> dict | None:
    return http_get_json(f"{SITE}/summary", {"event": event_id})


def get_teams() -> dict | None:
    return http_get_json(f"{SITE}/teams")


def _home_abbr(summary: dict) -> str | None:
    for c in (((summary or {}).get("header") or {}).get("competitions") or [{}])[0].get(
            "competitors", []):
        if c.get("homeAway") == "home":
            return (c.get("team") or {}).get("abbreviation")
    return None


def get_game_roster(event_id: str) -> list[dict]:
    """Parsed roster for one event: player, team, is_starter, is_active."""
    s = get_event_summary(event_id)
    if not s:
        return []
    home = _home_abbr(s)
    rows = []
    for block in (s.get("boxscore") or {}).get("players", []) or []:
        team = (block.get("team") or {}).get("abbreviation")
        groups = block.get("statistics") or []
        if not groups:
            continue
        for a in groups[0].get("athletes", []) or []:
            ath = a.get("athlete") or {}
            rows.append({
                "game_id": str(event_id), "player_espn_id": str(ath.get("id")),
                "player_name": ath.get("displayName"), "team": team,
                "is_home": int(team == home) if home else None,
                "is_starter": int(bool(a.get("starter"))),
                "is_active": int(not a.get("didNotPlay", False)),
            })
    return rows
