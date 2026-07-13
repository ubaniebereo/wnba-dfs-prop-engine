"""Module 1 — multi-season player baselines + position clusters.

Uses the deepened BallDontLie history to compute stable per-player baselines and
per-minute rates, anchored by position cluster. These anchor the rate models and
cut star regression-to-mean (a star's multi-season rate is a far better prior
than a one-season RandomForest mean).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from src.database import get_engine

MIN_GAMES = 10


@dataclass
class Baseline:
    player_id: str
    player_name: str
    position_cluster: str
    games: int
    baseline_min: float
    baseline_pts: float
    baseline_reb: float
    baseline_ast: float
    pts_per_min: float
    reb_per_min: float
    ast_per_min: float
    ts_pct: float


def _load() -> pd.DataFrame:
    eng = get_engine()
    df = pd.read_sql("SELECT * FROM player_game_stats WHERE minutes > 0", eng)
    try:
        pos = pd.read_sql("SELECT player_id, position_cluster FROM player_positions", eng)
        df = df.merge(pos, on="player_id", how="left")
    except Exception:
        df["position_cluster"] = None
    return df


def compute_baselines() -> pd.DataFrame:
    df = _load()
    g = df.groupby(["player_id", "player_name"])
    out = g.agg(
        position_cluster=("position_cluster", lambda s: s.dropna().iloc[0] if s.notna().any() else "wing"),
        games=("points", "size"),
        baseline_min=("minutes", "mean"), baseline_pts=("points", "mean"),
        baseline_reb=("rebounds", "mean"), baseline_ast=("assists", "mean"),
        tot_min=("minutes", "sum"), tot_pts=("points", "sum"),
        tot_reb=("rebounds", "sum"), tot_ast=("assists", "sum"),
        fga=("fga", "sum"), fta=("fta", "sum")).reset_index()
    out["pts_per_min"] = out["tot_pts"] / out["tot_min"].clip(lower=1)
    out["reb_per_min"] = out["tot_reb"] / out["tot_min"].clip(lower=1)
    out["ast_per_min"] = out["tot_ast"] / out["tot_min"].clip(lower=1)
    out["ts_pct"] = out["tot_pts"] / (2 * (out["fga"] + 0.44 * out["fta"]).clip(lower=1))
    out = out[out["games"] >= MIN_GAMES].copy()
    eng = get_engine()
    out.drop(columns=["tot_min", "tot_pts", "tot_reb", "tot_ast", "fga", "fta"]).to_sql(
        "player_baselines", eng, if_exists="replace", index=False)
    return out


def get_player_baseline(player_id: str) -> Baseline | None:
    df = pd.read_sql("SELECT * FROM player_baselines WHERE player_id = ?",
                     get_engine(), params=(str(player_id),))
    if df.empty:
        return None
    r = df.iloc[0]
    return Baseline(**{k: r[k] for k in Baseline.__dataclass_fields__})


def evaluate_baselines() -> pd.DataFrame:
    """Improvement check: per-cluster rates should match basketball intuition."""
    df = compute_baselines()
    tab = df.groupby("position_cluster").agg(
        players=("player_id", "size"),
        pts_per_min=("pts_per_min", "mean"), reb_per_min=("reb_per_min", "mean"),
        ast_per_min=("ast_per_min", "mean"), ts_pct=("ts_pct", "mean")).round(3)
    return tab


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("Per-cluster baselines (bigs->REB, guards->AST expected):")
    print(evaluate_baselines().to_string())
