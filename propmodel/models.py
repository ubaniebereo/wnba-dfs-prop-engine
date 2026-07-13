"""Module 4 — minutes x per-minute-rate prop model with count distributions.

E[stat] = E[minutes] * E[rate per minute].  The rate model is fit weighted by
minutes (so garbage-time noise doesn't dominate). Distribution by stat:
  * points          -> Normal(mu, sd)           (continuous-ish, symmetric)
  * rebounds/assists-> Negative Binomial(mu, r)  (over-dispersed counts)
P(over a X.5 line) is then exact from the chosen distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm
from sklearn.ensemble import GradientBoostingRegressor

from .featureset import MIN_FEATURES, STATS, rate_features

COUNT_STATS = {"rebounds", "assists"}


def _gbm():
    # HistGradientBoosting is histogram-based + multithreaded -> ~5-10x faster
    # to train than GradientBoosting, with comparable accuracy.
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=200, max_depth=3,
                                         learning_rate=0.05, random_state=42)


def train_minutes(train: pd.DataFrame):
    X = train[MIN_FEATURES].fillna(train[MIN_FEATURES].median())
    m = _gbm().fit(X, train["minutes"])
    sd = float(np.std(train["minutes"] - m.predict(X)))
    return {"model": m, "sd": sd, "medians": train[MIN_FEATURES].median()}


def train_rate(train: pd.DataFrame, stat: str):
    cols = rate_features(stat)
    d = train[train["minutes"] >= 8]            # stable rate sample
    X = d[cols].fillna(d[cols].median())
    y = (d[stat] / d["minutes"].clip(lower=1))
    m = _gbm().fit(X, y, sample_weight=d["minutes"])   # weight by minutes
    return {"model": m, "cols": cols, "medians": d[cols].median()}


def fit_dispersion(actual: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments Negative Binomial dispersion r (Var = mu + mu^2/r)."""
    resid_var = float(np.var(actual - mu))
    mean_mu = float(np.mean(mu))
    excess = resid_var - mean_mu
    if excess <= 0.1:
        return 50.0                              # ~Poisson if not over-dispersed
    return float(np.clip(mean_mu ** 2 / excess, 1.0, 200.0))


def project(rows: pd.DataFrame, minutes_m, rate_ms: dict) -> pd.DataFrame:
    out = rows[["game_id", "game_date", "player_id", "player_name",
                "team", "opponent_id", "position_group"]].copy()
    Xm = rows[MIN_FEATURES].fillna(minutes_m["medians"])
    e_min = np.clip(minutes_m["model"].predict(Xm), 0, 40)
    out["E_min"] = e_min.round(1)
    for stat, rm in rate_ms.items():
        Xr = rows[rm["cols"]].fillna(rm["medians"])
        e_rate = np.clip(rm["model"].predict(Xr), 0, None)
        out[f"E_{stat}"] = (e_min * e_rate).round(2)
    return out


def prob_over(stat: str, mu: float, line: float, params: dict) -> float:
    """P(stat > line). Lines are X.5, so P(over) = P(count >= ceil(line))."""
    if stat in COUNT_STATS:
        r = params["r"][stat]
        mu = max(mu, 1e-6)
        p = r / (r + mu)
        k = int(np.ceil(line))                   # X.5 -> ceil
        return float(1 - nbinom.cdf(k - 1, r, p))
    sd = params["sd"][stat]
    return float(1 - norm.cdf(line, loc=mu, scale=max(sd, 1e-6)))


def fit_distribution_params(test: pd.DataFrame, proj: pd.DataFrame) -> dict:
    """Estimate per-stat dispersion (NB r) / SD from held-out residuals."""
    r, sd = {}, {}
    m = test.merge(proj[["game_id", "player_id"] + [f"E_{s}" for s in STATS]],
                   on=["game_id", "player_id"], how="inner")
    for stat in STATS:
        mu = m[f"E_{stat}"].values
        actual = m[stat].values
        if stat in COUNT_STATS:
            r[stat] = fit_dispersion(actual, mu)
        sd[stat] = float(np.std(actual - mu))
    return {"r": r, "sd": sd}
