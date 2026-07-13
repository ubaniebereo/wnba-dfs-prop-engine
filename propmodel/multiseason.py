"""Module 1 — multi-season backfill via the EXISTING ESPN ingest (free).

We already have a working ESPN date-range ingester (src.ingest). wehoop is R and
sportsdataverse coverage for WNBA is partial, so the lowest-friction path to more
history is to walk earlier seasons through the same ingester into the same schema.
Run this, then rebuild features; baselines stabilize as seasons are added.
"""

from __future__ import annotations

import pandas as pd

from src.database import get_engine
from src.ingest import ingest_dates
from src.utils import get_logger
from datetime import date

log = get_logger(__name__)


def backfill(seasons=(2022, 2023, 2024), start_mmdd=(5, 1), end_mmdd=(10, 15)) -> dict:
    eng = get_engine()
    counts = {}
    for yr in seasons:
        c = ingest_dates(eng, date(yr, *start_mmdd), date(yr, *end_mmdd))
        counts[yr] = c
        log.info("Backfilled %d: %s", yr, c)
    return counts


def evaluate_baseline_stability() -> pd.DataFrame:
    """Per-player season scoring SD across seasons — should fall as data grows."""
    eng = get_engine()
    df = pd.read_sql("SELECT player_id, game_date, points FROM player_game_stats "
                     "WHERE minutes > 0", eng)
    df["season"] = df["game_date"].str[:4].astype(int)
    by = df.groupby(["player_id", "season"])["points"].mean().reset_index()
    spread = by.groupby("player_id")["points"].std().dropna()
    n_seasons = df.groupby("player_id")["season"].nunique()
    return pd.DataFrame({
        "players_total": [df["player_id"].nunique()],
        "players_multi_season": [int((n_seasons >= 2).sum())],
        "mean_cross_season_pts_sd": [round(float(spread.mean()), 2) if len(spread) else None],
    })
