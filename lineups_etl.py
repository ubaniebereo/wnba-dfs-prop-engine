"""Confirmed lineups ETL (Stage 3, Section 1).

Replaces the top-5-minutes starter PROXY with REAL ESPN confirmed starters.
Two ingestion paths:
  * from_espn_backup()  — maps real starters already captured in the Stage-1 ESPN
    DB (2025-26) onto the BDL-id player_game_stats by (date, normalized name).
  * from_espn_api(dates)— fetches confirmed rosters live/historically via ESPN.

Writes a `game_rosters` table and overwrites player_game_stats.starter with the
real value where available (leaving the proxy only where ESPN has no coverage).
"""

from __future__ import annotations

import re
import sqlite3

import pandas as pd
from sqlalchemy import text

from src.config import DB_PATH
from src.database import get_engine
from src.utils import get_logger

log = get_logger(__name__)
BACKUP = DB_PATH.parent / "wnba_espn_backup.sqlite"


def _norm(name: str) -> str:
    return re.sub(r"[^a-z ]", " ", str(name).lower()).strip()


def from_espn_backup() -> dict:
    """Map real starters from the ESPN backup DB onto the current BDL DB."""
    if not BACKUP.exists():
        return {"updated": 0, "note": "no ESPN backup DB"}
    esp = pd.read_sql("SELECT game_date, team, player_name, starter, "
                      "CASE WHEN minutes>0 THEN 1 ELSE 0 END is_active "
                      "FROM player_game_stats", sqlite3.connect(BACKUP))
    esp["key"] = esp["game_date"] + "|" + esp["player_name"].map(_norm)
    real = esp.drop_duplicates("key").set_index("key")

    eng = get_engine()
    bdl = pd.read_sql("SELECT stat_id, game_id, game_date, team, player_id, "
                      "player_name FROM player_game_stats", eng)
    bdl["key"] = bdl["game_date"] + "|" + bdl["player_name"].map(_norm)
    bdl = bdl.merge(real[["starter", "is_active"]], left_on="key", right_index=True,
                    how="left")
    matched = bdl.dropna(subset=["starter"]).copy()
    matched["starter"] = matched["starter"].astype(int)

    # update starter in player_game_stats
    with eng.begin() as c:
        for r in matched.itertuples():
            c.execute(text("UPDATE player_game_stats SET starter=:s WHERE stat_id=:id"),
                      {"s": int(r.starter), "id": r.stat_id})
    # build game_rosters
    rosters = matched[["game_id", "player_id", "team", "starter", "is_active"]].rename(
        columns={"starter": "is_starter"})
    rosters["roster_id"] = rosters["game_id"] + "_" + rosters["player_id"]
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS game_rosters"))
    rosters.to_sql("game_rosters", eng, if_exists="replace", index=False)
    return {"updated": len(matched), "games_covered": matched["game_id"].nunique()}


def from_espn_api(dates: list[str]) -> dict:
    """Fetch confirmed rosters for the given dates via ESPN (live/historical)."""
    import espn_client as ec
    rows = []
    for d in dates:
        sb = ec.get_scoreboard(d.replace("-", ""))
        for ev in (sb or {}).get("events", []):
            rows.extend(ec.get_game_roster(ev["id"]))
    df = pd.DataFrame(rows)
    if not df.empty:
        eng = get_engine()
        df["roster_id"] = df["game_id"] + "_" + df["player_espn_id"]
        df.to_sql("game_rosters_espn", eng, if_exists="replace", index=False)
    return {"rows": len(df)}


def evaluate_lineups() -> dict:
    eng = get_engine()
    df = pd.read_sql("SELECT game_id, starter FROM player_game_stats", eng)
    per_game = df.groupby("game_id")["starter"].sum()
    return {"rows": len(df),
            "median_starters_per_game": float(per_game.median()),
            "share_games_with_10_starters": round(float((per_game == 10).mean()), 2)}
