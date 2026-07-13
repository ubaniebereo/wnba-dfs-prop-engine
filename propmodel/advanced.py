"""Module 3 — advanced team context from EXISTING box scores (no new feed).

Derives possessions, pace, offensive/defensive rating, and opponent
defense-vs-position from the player_game_stats we already ingest. Position is
not stored, so we infer a coarse position group from each player's per-minute
rebound/assist profile (documented proxy; real positions via ESPN roster would
refine it).

Everything here returns RAW per-game rows; leakage-free rolling is applied in
featureset.py via shift(1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def position_group(player_df: pd.DataFrame) -> pd.Series:
    """player_id -> {'big','wing','guard'} from season per-minute reb/ast profile."""
    g = player_df.groupby("player_id").agg(
        mn=("minutes", "sum"), reb=("rebounds", "sum"),
        ast=("assists", "sum"), oreb=("oreb", "sum")).query("mn > 0")
    reb_pm = g["reb"] / g["mn"]
    ast_pm = g["ast"] / g["mn"]
    grp = pd.Series("wing", index=g.index)
    grp[(reb_pm >= 0.22)] = "big"           # rebound-heavy frontcourt
    grp[(ast_pm >= 0.13) & (reb_pm < 0.22)] = "guard"  # playmaking guards
    return grp.rename("position_group")


def team_box(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player rows to one team-game row with possessions & ratings."""
    agg = player_df.groupby(["game_id", "game_date", "team_id", "team",
                             "opponent_id"]).agg(
        pts=("points", "sum"), fga=("fga", "sum"), fta=("fta", "sum"),
        oreb=("oreb", "sum"), tov=("turnovers", "sum")).reset_index()
    # standard possession estimate
    agg["poss"] = (agg["fga"] - agg["oreb"] + agg["tov"] + 0.44 * agg["fta"]).clip(lower=1)
    agg["off_rtg"] = 100 * agg["pts"] / agg["poss"]
    # join opponent possessions/points for defensive rating + pace
    opp = agg[["game_id", "team_id", "poss", "pts"]].rename(
        columns={"team_id": "opponent_id", "poss": "opp_poss", "pts": "opp_pts"})
    agg = agg.merge(opp, on=["game_id", "opponent_id"], how="left")
    agg["def_rtg"] = 100 * agg["opp_pts"] / agg["opp_poss"].clip(lower=1)
    agg["pace"] = (agg["poss"] + agg["opp_poss"]) / 2.0   # ~possessions per game
    return agg


def defense_vs_position(player_df: pd.DataFrame, pos: pd.Series) -> pd.DataFrame:
    """Per team-game: stat allowed to each opponent position group.

    Returns rows keyed by (game_id, team_id) = the DEFENDING team, with columns
    def_pts_vs_<grp>, def_reb_vs_<grp>, def_ast_vs_<grp> = production the team's
    opponents at that position group put up in this game.
    """
    df = player_df.copy()
    df["position_group"] = df["player_id"].map(pos).fillna("wing")
    # offense by (game, offense team, group)
    off = df.groupby(["game_id", "team_id", "position_group"]).agg(
        pts=("points", "sum"), reb=("rebounds", "sum"),
        ast=("assists", "sum")).reset_index()
    # the DEFENDING team is the opponent of the offense team in that game
    opp_map = df[["game_id", "team_id", "opponent_id", "game_date"]].drop_duplicates()
    off = off.merge(opp_map, on=["game_id", "team_id"], how="left")
    off = off.rename(columns={"opponent_id": "def_team_id"})
    wide = off.pivot_table(index=["game_id", "def_team_id", "game_date"],
                           columns="position_group",
                           values=["pts", "reb", "ast"], aggfunc="sum").fillna(0)
    wide.columns = [f"def_{s}_vs_{g}" for s, g in wide.columns]
    return wide.reset_index().rename(columns={"def_team_id": "team_id"})
