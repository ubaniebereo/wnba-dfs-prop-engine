"""Assemble leakage-free per-player-game features for the minutes x rate model.

Key ideas that fight the diagnosed problems:
  * per-MINUTE rates (not raw totals) + a projected-minutes model => star
    production is preserved instead of regressed to the population mean.
  * empirical-Bayes POOLING of each player's per-minute rate toward their
    position-group league mean stabilizes low-sample players (and 2026 rookies).
  * advanced opponent context (def rating, pace, defense-vs-position) gives the
    model real matchup signal so "missing context" stops creating fake edges.

All rolling/expanding stats use shift(1): only games strictly before the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import advanced

STATS = ["points", "rebounds", "assists"]
# extended count props (DB column names); modeled per-family but no defense-vs-position
EXTENDED_STATS = ["tpm", "steals", "blocks", "turnovers"]
POOL_K = 120.0          # empirical-Bayes pseudo-minutes (shrinkage strength)
CTX_W = 10              # rolling window for opponent context


_NUMERIC = ["minutes", "points", "rebounds", "assists", "oreb", "dreb", "fgm",
            "fga", "tpm", "tpa", "ftm", "fta", "turnovers", "steals", "blocks",
            "fouls", "plus_minus", "starter", "is_home"]


def _load(engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM player_game_stats WHERE minutes > 0", engine)
    for col in _NUMERIC:                  # BDL can return null stats -> coerce to 0
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df[df["minutes"] > 0]
    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def _real_positions(engine):
    """Real BallDontLie position clusters if ingested, else None (use proxy)."""
    if engine is None:
        from src.database import get_engine
        engine = get_engine()
    try:
        p = pd.read_sql("SELECT player_id, position_cluster FROM player_positions", engine)
        return p.set_index("player_id")["position_cluster"] if not p.empty else None
    except Exception:
        return None


def build(engine=None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build features from SQLite (engine) or an in-memory frame (df).

    Passing df lets the live edge engine append synthetic 'future' rows (NaN
    stats) so upcoming-game features are computed with the SAME code as training.
    """
    pdf = df.copy() if df is not None else _load(engine)
    pos = _real_positions(engine)
    if pos is None:                       # fall back to stat-profile proxy
        pos = advanced.position_group(pdf)
    pdf["position_group"] = pdf["player_id"].map(pos).fillna("wing")
    pdf["season"] = pdf["game_date"].str[:4].astype(int)
    g = pdf.groupby("player_id", group_keys=False)

    # ---- minutes features ----
    for w in (3, 5):
        pdf[f"last_{w}_min"] = g["minutes"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    pdf["season_avg_min"] = g["minutes"].apply(lambda s: s.shift(1).expanding().mean())
    pdf["starter_rate"] = g["starter"].apply(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    pdf["history"] = g.cumcount()
    prev = g["game_date"].shift(1)
    pdf["days_rest"] = (pd.to_datetime(pdf["game_date"]) -
                        pd.to_datetime(prev)).dt.days.clip(upper=7)
    pdf["home_flag"] = pdf["is_home"].astype(int)
    pdf["is_starter"] = pdf["starter"].fillna(0).astype(int)   # real (confirmed) where available

    # rotation size: team's rolling count of players with >10 minutes (pre-game)
    rot = (pdf.assign(plays10=(pdf["minutes"] > 10).astype(int))
           .groupby(["game_id", "team_id"])["plays10"].transform("sum"))
    pdf["_rot_game"] = rot
    tr = pdf.groupby(["team_id", "game_date"])["_rot_game"].first().reset_index()
    tr = tr.sort_values(["team_id", "game_date"])
    tr["rotation_size"] = tr.groupby("team_id")["_rot_game"].apply(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()).values
    pdf = pdf.merge(tr[["team_id", "game_date", "rotation_size"]],
                    on=["team_id", "game_date"], how="left")

    # ---- per-minute rate features + empirical-Bayes pooled baseline ----
    prior_rate = {}   # position-group league prior (mean per-minute rate)
    for stat in STATS:
        rate = pdf[stat] / pdf["minutes"].clip(lower=1)
        prior_rate[stat] = pdf.groupby("position_group").apply(
            lambda d, s=stat: d[s].sum() / d["minutes"].clip(lower=1).sum()).to_dict()
        # leakage-free cumulative (prior games only)
        cum_stat = g[stat].apply(lambda s: s.shift(1).expanding().sum())
        cum_min = g["minutes"].apply(lambda s: s.shift(1).expanding().sum())
        prior_vec = pdf["position_group"].map(prior_rate[stat])
        pdf[f"pool_rate_{stat}"] = (cum_stat + POOL_K * prior_vec) / (cum_min + POOL_K)
        for w in (3, 5):
            num = g[stat].apply(lambda s: s.shift(1).rolling(w, min_periods=1).sum())
            den = g["minutes"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).sum())
            pdf[f"last_{w}_rate_{stat}"] = num / den.clip(lower=1)
        # raw rolling totals (for the 'old direct RF' baseline comparison)
        pdf[f"last_5_{stat}_raw"] = g[stat].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        pdf[f"season_avg_{stat}_raw"] = g[stat].apply(lambda s: s.shift(1).expanding().mean())

    # ---- extended-stat rate features (threes/steals/blocks/turnovers) ----
    for stat in EXTENDED_STATS:
        if stat not in pdf.columns:
            continue
        pdf[stat] = pd.to_numeric(pdf[stat], errors="coerce").fillna(0)
        prior = pdf.groupby("position_group").apply(
            lambda d, s=stat: d[s].sum() / d["minutes"].clip(lower=1).sum()).to_dict()
        cum_stat = g[stat].apply(lambda s: s.shift(1).expanding().sum())
        cum_min = g["minutes"].apply(lambda s: s.shift(1).expanding().sum())
        prior_vec = pdf["position_group"].map(prior)
        pdf[f"pool_rate_{stat}"] = (cum_stat + POOL_K * prior_vec) / (cum_min + POOL_K)
        for w in (3, 5):
            num = g[stat].apply(lambda s: s.shift(1).rolling(w, min_periods=1).sum())
            den = g["minutes"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).sum())
            pdf[f"last_{w}_rate_{stat}"] = num / den.clip(lower=1)

    # ---- opponent advanced context (leakage-free rolling) ----
    tb = advanced.team_box(pdf)
    tb = tb.sort_values(["team_id", "game_date"])
    tg = tb.groupby("team_id", group_keys=False)
    tb["roll_def_rtg"] = tg["def_rtg"].apply(lambda s: s.shift(1).rolling(CTX_W, min_periods=1).mean())
    tb["roll_off_rtg"] = tg["off_rtg"].apply(lambda s: s.shift(1).rolling(CTX_W, min_periods=1).mean())
    tb["roll_pace"] = tg["pace"].apply(lambda s: s.shift(1).rolling(CTX_W, min_periods=1).mean())
    ctx = tb[["game_id", "team_id", "roll_def_rtg", "roll_off_rtg", "roll_pace"]].rename(
        columns={"team_id": "opponent_id", "roll_def_rtg": "opp_def_rtg",
                 "roll_off_rtg": "opp_off_rtg", "roll_pace": "opp_pace"})
    pdf = pdf.merge(ctx, on=["game_id", "opponent_id"], how="left")

    # ---- opponent defense-vs-position (leakage-free rolling) ----
    dvp = advanced.defense_vs_position(pdf, pos).sort_values(["team_id", "game_date"])
    dg = dvp.groupby("team_id", group_keys=False)
    for grp in ("big", "wing", "guard"):
        for stat_abbr in ("pts", "reb", "ast"):
            col = f"def_{stat_abbr}_vs_{grp}"
            if col in dvp.columns:
                dvp[f"roll_{col}"] = dg[col].apply(
                    lambda s: s.shift(1).rolling(CTX_W, min_periods=1).mean())
    roll_cols = [c for c in dvp.columns if c.startswith("roll_def_")]
    dvp_m = dvp[["game_id", "team_id"] + roll_cols].rename(columns={"team_id": "opponent_id"})
    pdf = pdf.merge(dvp_m, on=["game_id", "opponent_id"], how="left")
    # collapse to the player's own position group: opp_def_<stat>_vs_pos
    for stat_abbr in ("pts", "reb", "ast"):
        pdf[f"opp_def_{stat_abbr}_vs_pos"] = np.nan
        for grp in ("big", "wing", "guard"):
            col = f"roll_def_{stat_abbr}_vs_{grp}"
            if col in pdf.columns:
                mask = pdf["position_group"] == grp
                pdf.loc[mask, f"opp_def_{stat_abbr}_vs_pos"] = pdf.loc[mask, col]
    return pdf


MIN_FEATURES = ["last_3_min", "last_5_min", "season_avg_min", "starter_rate",
                "is_starter", "rotation_size", "days_rest", "home_flag", "opp_pace"]


def rate_features(stat: str) -> list[str]:
    base = [f"pool_rate_{stat}", f"last_3_rate_{stat}", f"last_5_rate_{stat}",
            "opp_def_rtg", "opp_pace", "home_flag", "starter_rate"]
    sa = {"points": "pts", "rebounds": "reb", "assists": "ast"}.get(stat)
    if sa:                                  # core stats add defense-vs-position
        base.insert(5, f"opp_def_{sa}_vs_pos")
    return base
