"""ESPN confirmation layer (Stage 6, Sec 5C) -> lineup_events (confirmed)."""

from __future__ import annotations

import espn_client as _espn
from .normalizers import PlayerResolver
from .storage import now_iso
from src.utils import get_logger

log = get_logger(__name__)


def confirmed_lineups(resolver: PlayerResolver | None = None,
                      date: str | None = None) -> list[dict]:
    """Confirmed starters/actives for the current (or given) slate."""
    resolver = resolver or PlayerResolver()
    from datetime import date as _d
    ds = (date or _d.today().strftime("%Y%m%d")).replace("-", "")
    sb = _espn.get_scoreboard(ds) or {}
    rows = []
    for ev in sb.get("events", []) or []:
        for r in _espn.get_game_roster(ev["id"]):
            res = resolver.resolve(r["player_name"]) if r.get("player_name") else None
            rows.append({"source": "espn", "captured_at": now_iso(),
                         "game_id": r["game_id"],
                         "player_id": res["player_id"] if res else None,
                         "player_name": r["player_name"], "team_id": r.get("team"),
                         "is_starter": r.get("is_starter"), "is_active": r.get("is_active"),
                         "projected_vs_confirmed": "confirmed", "metadata_json": {}})
    return rows


def capture_lineups() -> int:
    from .storage import init_storage, insert
    init_storage()
    n = insert("lineup_events", confirmed_lineups())
    log.info("captured %d confirmed lineup rows", n)
    return n
