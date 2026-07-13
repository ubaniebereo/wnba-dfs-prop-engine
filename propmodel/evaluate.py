"""Backtest + diagnostics proving the upgrade helps (all leakage-free, as-of).

Compares the new minutes x rate model against (a) the old direct RandomForest and
(b) a recency baseline, on a chronological hold-out, and reports the diagnostics
the spec asks for: projection error, STAR under-projection bias, count-distribution
fit, and probability calibration (reliability + Brier).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import models
from .featureset import STATS, build


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def _split(feat, test_fraction=0.25, min_hist=5):
    d = feat[feat["history"] >= min_hist].dropna(subset=["minutes"]).copy()
    dates = np.sort(d["game_date"].unique())
    cut = dates[int(len(dates) * (1 - test_fraction))]
    return d[d["game_date"] < cut], d[d["game_date"] >= cut], cut


def _old_rf(train, test, stat):
    """Old approach: RF predicting the stat directly from raw rolling totals."""
    cols = [f"last_5_{stat}_raw", f"season_avg_{stat}_raw", "last_5_min",
            "season_avg_min", "home_flag", "days_rest"]
    Xtr = train[cols].fillna(train[cols].median())
    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                               min_samples_leaf=5, n_jobs=-1, random_state=0).fit(
        Xtr, train[stat])
    return rf.predict(test[cols].fillna(train[cols].median()))


def _calibration(pairs: pd.DataFrame, bins=8):
    pairs = pairs.copy()
    pairs["bin"] = pd.cut(pairs["p"], np.linspace(0, 1, bins + 1))
    tab = pairs.groupby("bin", observed=True).agg(
        n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean")).round(3)
    brier = float(np.mean((pairs["p"] - pairs["hit"]) ** 2))
    return tab, brier


def run(engine) -> dict:
    feat = build(engine)
    train, test, cut = _split(feat)
    out = {"n_train": len(train), "n_test": len(test), "cut_date": str(cut)[:10]}

    # --- fit models on train; estimate distribution params on TRAIN residuals ---
    mm = models.train_minutes(train)
    rms = {s: models.train_rate(train, s) for s in STATS}
    proj_tr = models.project(train, mm, rms)
    params = models.fit_distribution_params(train, proj_tr)
    proj_te = models.project(test, mm, rms)

    # --- minutes accuracy ---
    mt = test.merge(proj_te[["game_id", "player_id", "E_min"]], on=["game_id", "player_id"])
    out["minutes"] = {"MAE": round(mean_absolute_error(mt["minutes"], mt["E_min"]), 2),
                      "RMSE": round(_rmse(mt["minutes"], mt["E_min"]), 2)}

    # --- per-stat error: new vs old-RF vs recency, + star bias ---
    rows, calib_pairs = [], []
    star_ids = (train.groupby("player_id")["points"].mean()
                .sort_values(ascending=False).head(20).index)
    for stat in STATS:
        m = test.merge(proj_te[["game_id", "player_id", f"E_{stat}"]],
                       on=["game_id", "player_id"])
        new = m[f"E_{stat}"].values
        actual = m[stat].values
        old = _old_rf(train, test, stat)
        recency = test[f"last_5_{stat}_raw"].fillna(test[f"season_avg_{stat}_raw"]).values
        star_mask = m["player_id"].isin(star_ids).values
        rows.append({
            "stat": stat,
            "MAE_new": round(mean_absolute_error(actual, new), 2),
            "MAE_oldRF": round(mean_absolute_error(actual, old), 2),
            "MAE_recency": round(mean_absolute_error(actual, recency), 2),
            "star_bias_new": round(float(np.mean(new[star_mask] - actual[star_mask])), 2),
            "star_bias_oldRF": round(float(np.mean(old[star_mask] - actual[star_mask])), 2),
        })
        # calibration pairs: P(over k) vs realized, for k near each row's projection
        for _, r in m.iterrows():
            mu = r[f"E_{stat}"]
            for k in (np.floor(mu) - 0.5, np.floor(mu) + 1.5, np.floor(mu) + 3.5):
                if k <= 0:
                    continue
                p = models.prob_over(stat, mu, k, params)
                calib_pairs.append({"p": p, "hit": int(r[stat] > k)})
    out["per_stat"] = pd.DataFrame(rows)

    # --- count distribution fit (dispersion) ---
    disp = []
    for stat in ("rebounds", "assists"):
        m = test.merge(proj_te[["game_id", "player_id", f"E_{stat}"]],
                       on=["game_id", "player_id"])
        mu, actual = m[f"E_{stat}"].values, m[stat].values
        disp.append({"stat": stat, "mean_mu": round(float(np.mean(mu)), 2),
                     "resid_var": round(float(np.var(actual - mu)), 2),
                     "NB_r": round(params["r"][stat], 1),
                     "note": "Var>mean => over-dispersed (NB appropriate)"
                     if np.var(actual - mu) > np.mean(mu) else "~Poisson"})
    out["dispersion"] = pd.DataFrame(disp)

    # --- probability calibration ---
    tab, brier = _calibration(pd.DataFrame(calib_pairs))
    out["calibration_table"] = tab
    out["brier"] = round(brier, 4)
    out["params"] = params
    out["models"] = {"minutes": mm, "rates": rms}
    return out
