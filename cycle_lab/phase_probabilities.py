"""Phase classifier + bootstrap simulation for cycle-sensitive players.

Two honest comparisons:
  * performance-only model (deltas + context)  -> how much phase is recoverable
    from box-score deviations ALONE (expected: barely above base rate).
  * +symptoms model                            -> adds subjective symptom burden.

Outputs accuracy, multiclass Brier, a calibration table, per-game phase
probabilities for an example player, and a bootstrap of per-phase mean deltas
(to visualize how small and overlapping the effects are).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PHASES = ["menstruation", "follicular", "luteal"]
PERF_FEATURES = ["delta_TS", "delta_eFG", "delta_REB", "delta_PIR", "delta_pts",
                 "usage_rate", "contact_intensity", "minutes",
                 "opponent_strength", "home_flag",
                 "days_rest", "b2b_flag", "game_in_week", "cumulative_minutes_last_3"]
SYMPTOM_FEATURES = ["symptoms_high", "recovery_high"]


def _multiclass_brier(y_true_idx, proba, n_classes) -> float:
    onehot = np.eye(n_classes)[y_true_idx]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _fit_eval(df, features, label):
    X = df[features].fillna(df[features].median())
    y = df["cycle_phase"].astype("category")
    y_idx = y.cat.codes.values
    classes = list(y.cat.categories)
    Xtr, Xte, ytr, yte = train_test_split(X, y_idx, test_size=0.3,
                                          random_state=0, stratify=y_idx)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=0.5)  # multinomial is the default
    clf.fit(scaler.transform(Xtr), ytr)
    proba = clf.predict_proba(scaler.transform(Xte))
    pred = proba.argmax(1)
    acc = float((pred == yte).mean())
    base_rate = float(pd.Series(yte).value_counts(normalize=True).max())
    brier = _multiclass_brier(yte, proba, len(classes))
    # calibration: bin max-confidence vs hit rate
    conf = proba.max(1)
    hit = (pred == yte).astype(int)
    bins = pd.cut(conf, [0, .4, .5, .6, .7, 1.01])
    calib = pd.DataFrame({"conf_bin": bins, "hit": hit, "pred_conf": conf}) \
        .groupby("conf_bin", observed=True).agg(
            n=("hit", "size"), accuracy=("hit", "mean"),
            mean_conf=("pred_conf", "mean")).round(3)
    return {"label": label, "accuracy": round(acc, 3), "base_rate": round(base_rate, 3),
            "brier": round(brier, 3), "n_test": len(yte)}, calib, (clf, scaler, classes, features)


def run_classifier(df_sensitive: pd.DataFrame):
    """Fit performance-only and +symptoms models; return metrics + fitted bundle."""
    m_perf, calib_perf, _ = _fit_eval(df_sensitive, PERF_FEATURES, "performance_only")
    m_sym, calib_sym, bundle = _fit_eval(
        df_sensitive, PERF_FEATURES + SYMPTOM_FEATURES, "with_symptoms")
    metrics = pd.DataFrame([m_perf, m_sym])
    return metrics, calib_sym, bundle


def example_player_probabilities(df_sensitive, bundle, n=8) -> pd.DataFrame:
    clf, scaler, classes, features = bundle
    pid = df_sensitive["player_id"].iloc[0]
    d = df_sensitive[df_sensitive["player_id"] == pid].head(n).copy()
    proba = clf.predict_proba(scaler.transform(d[features].fillna(0)))
    out = d[["player_name", "date", "cycle_phase"]].reset_index(drop=True)
    for i, c in enumerate(classes):
        out[f"P({c})"] = proba[:, i].round(2)
    out["most_likely"] = [classes[i] for i in proba.argmax(1)]
    return out


def simulate_phase_deltas(df_sensitive, n_boot=2000, seed=1) -> pd.DataFrame:
    """Bootstrap mean delta by phase -> shows effect size + overlap (uncertainty)."""
    rng = np.random.default_rng(seed)
    rows = []
    for metric in ["delta_TS", "delta_REB", "delta_PIR", "delta_pts"]:
        for phase in PHASES:
            vals = df_sensitive.loc[df_sensitive["cycle_phase"] == phase, metric].dropna().values
            if len(vals) < 5:
                continue
            boots = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
            rows.append({"metric": metric, "phase": phase,
                         "mean_delta": round(float(np.mean(vals)), 3),
                         "ci90_low": round(float(np.percentile(boots, 5)), 3),
                         "ci90_high": round(float(np.percentile(boots, 95)), 3),
                         "n": len(vals)})
    return pd.DataFrame(rows)
