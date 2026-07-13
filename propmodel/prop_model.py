"""Unified prop distribution + params (Stage 4, Section 1).

PropDistribution selects NB (rebounds/assists) or Normal (points, variance via
law of total variance) and exposes prob_over / prob_under. build_params() fits
the NB dispersion (MLE) and points variance from held-out residuals so the live
edge engine uses calibrated distributional probabilities, not naive Normal SD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import distributions, models
from .featureset import STATS


def build_params(train: pd.DataFrame, proj_train: pd.DataFrame,
                 var_min: float) -> dict:
    """nb_r per count stat (MLE) + points variance (law of total variance)."""
    m = train.merge(proj_train[["game_id", "player_id"] + [f"E_{s}" for s in STATS]],
                    on=["game_id", "player_id"], how="inner")
    nb_r, var_stat = {}, {}
    for stat in STATS:
        mu = m[f"E_{stat}"].values
        actual = m[stat].values
        if stat in distributions.COUNT_STATS:
            nb_r[stat] = distributions.fit_nb_dispersion(actual, mu)
        # rate residual variance -> LTV total variance for the Normal (points)
        rate_res = (m[stat] / m["minutes"].clip(lower=1)) - (mu / m["minutes"].clip(lower=1))
        var_rate = float(np.var(rate_res))
        e_min = float(m["minutes"].mean())
        e_rate = float((mu / m["minutes"].clip(lower=1)).mean())
        ltv = distributions.total_variance(e_min, var_min, e_rate, var_rate)
        # use the larger of empirical residual var and LTV (honest upper bound)
        var_stat[stat] = max(float(np.var(actual - mu)), ltv)
    return {"nb_r": nb_r, "var_stat": var_stat}


class PropDistribution:
    """Per-(player,game,stat) distribution wrapper."""

    def __init__(self, params: dict):
        self.params = params

    def prob_over(self, stat: str, e_stat: float, line: float) -> float:
        return distributions.prob_over(stat, e_stat, line, self.params)

    def prob_under(self, stat: str, e_stat: float, line: float) -> float:
        return 1.0 - self.prob_over(stat, e_stat, line)


def evaluate_live_edge_distribution(edge_df: pd.DataFrame) -> dict:
    if edge_df is None or edge_df.empty:
        return {"n": 0}
    e = edge_df["edge"]
    return {"n": len(edge_df),
            "median_edge": round(float(e.median()), 3),
            "share_0_5pct": round(float(((e >= 0) & (e <= 0.05)).mean()), 2),
            "share_gt5pct": round(float((e > 0.05).mean()), 2),
            "share_gt7pct": round(float((e > 0.07).mean()), 2),
            "over_share": round(float((edge_df["side"] == "over").mean()), 2)}
