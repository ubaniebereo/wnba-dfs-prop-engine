"""Module 5 — probability calibration + market-as-prior blending.

Two corrections layered on the raw model:
  * isotonic calibration: maps raw P(over) -> realized frequency, removing the
    overconfidence seen in the backtest (0.80 predicted hitting ~0.73).
  * market blend: shrink the projected mean toward the de-vigged market line
    (the sharp line is a strong prior), so our noisy disagreement -- especially
    leftover star misprojection -- doesn't manufacture large edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import models
from .featureset import STATS


def build_calibration_pairs(rows: pd.DataFrame, proj: pd.DataFrame,
                            params: dict) -> pd.DataFrame:
    pairs = []
    m = rows.merge(proj[["game_id", "player_id"] + [f"E_{s}" for s in STATS]],
                   on=["game_id", "player_id"])
    for stat in STATS:
        for _, r in m.iterrows():
            mu = r[f"E_{stat}"]
            for k in (np.floor(mu) - 0.5, np.floor(mu) + 1.5, np.floor(mu) + 3.5):
                if k <= 0:
                    continue
                pairs.append({"stat": stat,
                              "p": models.prob_over(stat, mu, k, params),
                              "hit": int(r[stat] > k)})
    return pd.DataFrame(pairs)


def fit_isotonic(pairs: pd.DataFrame) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(pairs["p"].values, pairs["hit"].values)
    return iso


def calibrate(iso: IsotonicRegression, p: float | np.ndarray):
    return float(np.clip(iso.predict(np.atleast_1d(p))[0], 1e-4, 1 - 1e-4))


def market_blend(e_model: float, line: float, w_model: float = 0.5) -> float:
    """Shrink projection toward the market line (line ~ market's mean estimate)."""
    return w_model * e_model + (1 - w_model) * line


def brier(pairs: pd.DataFrame, iso: IsotonicRegression | None = None) -> float:
    p = pairs["p"].values if iso is None else iso.predict(pairs["p"].values)
    return float(np.mean((p - pairs["hit"].values) ** 2))
