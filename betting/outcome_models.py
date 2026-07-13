"""Cycle-agnostic game-outcome models on real data: win prob, margin, vs Elo.

Chronological split. Win probability via regularized logistic regression and
gradient boosting; margin via Ridge and gradient boosting. Everything is
benchmarked against a pure-Elo baseline so we can see whether the richer feature
set actually beats a simple rating system on this (small) sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             mean_absolute_error, mean_squared_error, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from .team_features import FEATURES, build_model_frame


def _chrono_split(df, test_fraction=0.25):
    dates = np.sort(df["game_date"].unique())
    cut = dates[int(len(dates) * (1 - test_fraction))]
    return df[df["game_date"] < cut], df[df["game_date"] >= cut], cut


def _calibration(y, p, bins=5) -> pd.DataFrame:
    b = pd.cut(p, np.linspace(0, 1, bins + 1))
    return pd.DataFrame({"bin": b, "y": y, "p": p}).groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3)


def run() -> dict:
    df = build_model_frame()
    train, test, cut = _chrono_split(df)
    Xtr, Xte = train[FEATURES].fillna(0), test[FEATURES].fillna(0)
    ytr, yte = train["home_win"], test["home_win"]
    mtr, mte = train["margin"], test["margin"]

    scaler = StandardScaler().fit(Xtr)
    results = {"n_train": len(train), "n_test": len(test), "cut_date": str(cut)[:10]}

    # ---- win probability ----
    logit = LogisticRegression(max_iter=1000, C=0.5).fit(scaler.transform(Xtr), ytr)
    gbc = GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                     learning_rate=0.05).fit(Xtr, ytr)
    # Elo baseline: map elo gap straight to a probability
    elo_p = 1.0 / (1.0 + 10 ** (-(test["elo_diff"]) / 400))

    win_rows = []
    for name, p in [("elo_baseline", elo_p.values),
                    ("logistic", logit.predict_proba(scaler.transform(Xte))[:, 1]),
                    ("grad_boost", gbc.predict_proba(Xte)[:, 1])]:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        win_rows.append({
            "model": name,
            "accuracy": round(accuracy_score(yte, (p > 0.5).astype(int)), 3),
            "log_loss": round(log_loss(yte, p), 3),
            "brier": round(brier_score_loss(yte, p), 3),
            "roc_auc": round(roc_auc_score(yte, p), 3)})
    results["win_models"] = pd.DataFrame(win_rows)
    results["calibration_logistic"] = _calibration(
        yte.values, logit.predict_proba(scaler.transform(Xte))[:, 1])

    # ---- margin (point differential) ----
    ridge = Ridge(alpha=1.0).fit(scaler.transform(Xtr), mtr)
    gbr = GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                    learning_rate=0.05).fit(Xtr, mtr)
    margin_rows = []
    for name, pred in [("elo_baseline", (test["elo_diff"] / 25).values),
                       ("ridge", ridge.predict(scaler.transform(Xte))),
                       ("grad_boost", gbr.predict(Xte))]:
        margin_rows.append({
            "model": name,
            "MAE": round(mean_absolute_error(mte, pred), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(mte, pred))), 2)})
    results["margin_models"] = pd.DataFrame(margin_rows)
    return results
