"""Module 2 — injuries / inactives from the ESPN injuries feed (real, free).

Ingests current injury statuses so the edge engine never prices a player who is
OUT, and so expected minutes can be nudged when rotation players are sidelined.
Endpoint: .../basketball/wnba/injuries (status + athlete + position).
"""

from __future__ import annotations

import pandas as pd

from src.utils import get_logger, http_get_json
from src.database import get_engine
from sqlalchemy import text

log = get_logger(__name__)
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries"
OUT_STATUSES = {"out", "injured reserve", "suspension"}


def fetch_injuries() -> pd.DataFrame:
    payload = http_get_json(ESPN_INJURIES) or {}
    rows = []
    for team in payload.get("injuries", []) or []:
        for inj in team.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            status = (inj.get("status") or "").strip()
            rows.append({
                "player_id": str(ath.get("id")),
                "player_name": ath.get("displayName"),
                "position": (ath.get("position") or {}).get("abbreviation"),
                "status": status,
                "is_out": int(status.lower() in OUT_STATUSES),
                "date": (inj.get("date") or "")[:10],
                "detail": inj.get("shortComment"),
            })
    return pd.DataFrame(rows)


def store_injuries(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    eng = get_engine()
    with eng.begin() as c:
        c.execute(text("CREATE TABLE IF NOT EXISTS injuries ("
                       "player_id TEXT, player_name TEXT, position TEXT, status TEXT, "
                       "is_out INTEGER, date TEXT, detail TEXT)"))
        c.execute(text("DELETE FROM injuries"))
    df.to_sql("injuries", eng, if_exists="append", index=False)
    return len(df)


def out_player_ids() -> set[str]:
    """Player IDs currently OUT — exclude these from prop pricing."""
    df = fetch_injuries()
    return set(df.loc[df["is_out"] == 1, "player_id"]) if not df.empty else set()


def evaluate_injuries() -> dict:
    df = fetch_injuries()
    return {"players_listed": len(df), "out": int(df["is_out"].sum()) if not df.empty else 0,
            "day_to_day": int((df["status"].str.lower() == "day-to-day").sum())
            if not df.empty else 0}
