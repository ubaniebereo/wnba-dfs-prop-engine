"""Aggregate the REAL ingested box scores into game-level modeling features.

All rolling team features are leakage-free (prior games only, via shift(1)).
Elo is computed sequentially so each game uses pre-game ratings. The model frame
expresses every feature as a home-minus-away difference plus the Elo gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from src import config
from src.database import get_engine

ROLL_FORM = 5
ROLL_RATE = 10


def _load():
    eng = get_engine()
    with eng.connect() as c:
        games = pd.read_sql(text(
            "SELECT game_id, game_date, home_team, away_team, home_score, away_score "
            "FROM games WHERE completed=1 AND home_score IS NOT NULL "
            "ORDER BY game_date"), c)
        tgs = pd.read_sql(text(
            "SELECT game_id, game_date, team_id, team, opponent, is_home, "
            "points_for, points_against FROM team_game_stats"), c)
    return games, tgs


def _team_timeline(tgs: pd.DataFrame) -> pd.DataFrame:
    """Pre-game rolling form / offense / defense / rest for each team-game."""
    t = tgs.dropna(subset=["points_for", "points_against"]).copy()
    t = t.sort_values(["team", "game_date"])
    t["win"] = (t["points_for"] > t["points_against"]).astype(int)
    g = t.groupby("team", group_keys=False)
    t["form5"] = g["win"].apply(lambda s: s.shift(1).rolling(ROLL_FORM, min_periods=1).mean())
    t["off10"] = g["points_for"].apply(lambda s: s.shift(1).rolling(ROLL_RATE, min_periods=1).mean())
    t["def10"] = g["points_against"].apply(lambda s: s.shift(1).rolling(ROLL_RATE, min_periods=1).mean())
    t["net10"] = t["off10"] - t["def10"]
    prev_date = g["game_date"].shift(1)
    t["rest"] = (pd.to_datetime(t["game_date"]) - pd.to_datetime(prev_date)).dt.days.clip(upper=7)
    t["b2b"] = (t["rest"] == 1).astype(int)
    t["history"] = g.cumcount()
    return t


def _elo(games: pd.DataFrame, k=20.0, home_adv=60.0, base=1500.0) -> pd.DataFrame:
    """Sequential Elo; returns pre-game home/away ratings per game_id."""
    rating: dict[str, float] = {}
    rows = []
    for _, gm in games.iterrows():
        h, a = gm["home_team"], gm["away_team"]
        rh, ra = rating.get(h, base), rating.get(a, base)
        eh = 1.0 / (1.0 + 10 ** (-((rh + home_adv) - ra) / 400))
        home_win = int(gm["home_score"] > gm["away_score"])
        rows.append({"game_id": gm["game_id"], "home_elo_pre": rh, "away_elo_pre": ra,
                     "elo_exp_home": eh})
        rating[h] = rh + k * (home_win - eh)
        rating[a] = ra + k * ((1 - home_win) - (1 - eh))
    return pd.DataFrame(rows)


def build_model_frame(min_history=5) -> pd.DataFrame:
    games, tgs = _load()
    tl = _team_timeline(tgs)
    elo = _elo(games)

    cols = ["game_id", "team", "form5", "off10", "def10", "net10", "rest", "b2b", "history"]
    home = tl[cols].add_prefix("h_").rename(columns={"h_game_id": "game_id", "h_team": "home_team"})
    away = tl[cols].add_prefix("a_").rename(columns={"a_game_id": "game_id", "a_team": "away_team"})

    df = games.merge(home, on=["game_id", "home_team"], how="left") \
              .merge(away, on=["game_id", "away_team"], how="left") \
              .merge(elo, on="game_id", how="left")
    df = df[(df["h_history"] >= min_history) & (df["a_history"] >= min_history)].copy()

    # home-minus-away differential features + elo gap
    df["d_form"] = df["h_form5"] - df["a_form5"]
    df["d_off"] = df["h_off10"] - df["a_off10"]
    df["d_def"] = df["h_def10"] - df["a_def10"]
    df["d_net"] = df["h_net10"] - df["a_net10"]
    df["d_rest"] = df["h_rest"] - df["a_rest"]
    df["d_b2b"] = df["h_b2b"] - df["a_b2b"]
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["margin"] = df["home_score"] - df["away_score"]
    return df.sort_values("game_date").reset_index(drop=True)


FEATURES = ["d_form", "d_off", "d_def", "d_net", "d_rest", "d_b2b", "elo_diff"]
