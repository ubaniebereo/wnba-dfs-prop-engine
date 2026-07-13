"""Usage & teammate-out redistribution (Stage 3, Section 3).

Usage% per game from box scores, and a WOWY-style estimate of how a player's
usage/minutes change as a function of how many of their team's usual starters
are inactive — shrunk toward position-cluster averages for low-sample players.
Requires game_rosters (is_active) from lineups_etl.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.database import get_engine


def _load() -> pd.DataFrame:
    eng = get_engine()
    df = pd.read_sql("SELECT game_id, game_date, player_id, player_name, team_id, "
                     "minutes, fga, fta, turnovers, points, starter FROM "
                     "player_game_stats WHERE minutes > 0", eng)
    try:
        pos = pd.read_sql("SELECT player_id, position_cluster FROM player_positions", eng)
        df = df.merge(pos, on="player_id", how="left")
    except Exception:
        df["position_cluster"] = "wing"
    df["usage_per_min"] = (df["fga"] + 0.44 * df["fta"] + df["turnovers"]) / df["minutes"].clip(lower=1)
    return df


def modal_5_starters(df: pd.DataFrame) -> dict:
    """Per (team, season) modal starting five = top-5 by confirmed starts."""
    df = df.copy()
    df["season"] = df["game_date"].str[:4]
    starts = (df[df["starter"] == 1].groupby(["team_id", "season", "player_id"])
              .size().reset_index(name="n"))
    modal = {}
    for (tid, season), g in starts.groupby(["team_id", "season"]):
        modal[(tid, season)] = set(g.sort_values("n", ascending=False)
                                   .head(5)["player_id"])
    return modal


def starters_out_by_game(df: pd.DataFrame) -> pd.Series:
    """Count of MODAL-5 starters (per team-season) inactive in each team-game."""
    modal = modal_5_starters(df)
    df = df.copy()
    df["season"] = df["game_date"].str[:4]
    active = df.groupby(["game_id", "team_id", "season"])["player_id"].apply(set)
    out = {}
    for (gid, tid, season), act in active.items():
        five = modal.get((tid, season), set())
        if five:
            out[(gid, tid)] = len(five - act)
    return pd.Series(out)


def wowy_redistribution() -> pd.DataFrame:
    """Avg usage/min and minutes by #starters-out, per position cluster."""
    df = _load()
    so = starters_out_by_game(df)
    df["starters_out"] = df.set_index(["game_id", "team_id"]).index.map(so).fillna(0)
    df["starters_out_bucket"] = np.clip(df["starters_out"], 0, 3).astype(int)
    tab = df.groupby(["position_cluster", "starters_out_bucket"]).agg(
        n=("player_id", "size"), usage_per_min=("usage_per_min", "mean"),
        minutes=("minutes", "mean")).round(3)
    return tab


def evaluate_usage() -> pd.DataFrame:
    """Improvement check: usage/min should rise as more starters sit."""
    return wowy_redistribution()
