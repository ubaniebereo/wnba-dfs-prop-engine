"""Internal 'fair' game models (win, margin->spread, total) on REAL data.

Trains on completed games, then projects upcoming games from each team's
pre-game rolling state + current Elo. Margin and total are regressions whose
prediction + residual SD give a Normal from which we read P(cover) and P(over).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from betting.team_features import (FEATURES, _elo, _load, _team_timeline,
                                   build_model_frame)
from src.utils import get_logger

log = get_logger(__name__)
TOTAL_FEATURES = ["h_off10", "a_off10", "h_def10", "a_def10"]


def train_game_models(test_fraction=0.25) -> dict:
    df = build_model_frame()
    df["total"] = df["home_score"] + df["away_score"]
    dates = np.sort(df["game_date"].unique())
    cut = dates[int(len(dates) * (1 - test_fraction))]
    tr, te = df[df["game_date"] < cut], df[df["game_date"] >= cut]

    scaler = StandardScaler().fit(tr[FEATURES].fillna(0))
    win = LogisticRegression(max_iter=1000, C=0.5).fit(
        scaler.transform(tr[FEATURES].fillna(0)), tr["home_win"])
    margin = Ridge(alpha=1.0).fit(scaler.transform(tr[FEATURES].fillna(0)), tr["margin"])
    total = GradientBoostingRegressor(n_estimators=120, max_depth=2,
                                      learning_rate=0.05).fit(
        tr[TOTAL_FEATURES].fillna(0), tr["total"])

    # residual SDs from the held-out games (drive cover/over probabilities)
    m_pred = margin.predict(scaler.transform(te[FEATURES].fillna(0)))
    t_pred = total.predict(te[TOTAL_FEATURES].fillna(0))
    margin_sd = float(np.std(te["margin"] - m_pred)) or 12.0
    total_sd = float(np.std(te["total"] - t_pred)) or 15.0
    # de-bias: shift predictions so held-out mean error is zero
    margin_bias = float(m_pred.mean() - te["margin"].mean())
    total_bias = float(t_pred.mean() - te["total"].mean())

    p_win = win.predict_proba(scaler.transform(te[FEATURES].fillna(0)))[:, 1]
    metrics = {
        "win_accuracy": round(accuracy_score(te["home_win"], p_win > 0.5), 3),
        "win_log_loss": round(log_loss(te["home_win"], np.clip(p_win, 1e-6, 1 - 1e-6)), 3),
        "margin_MAE": round(mean_absolute_error(te["margin"], m_pred), 2),
        "total_MAE": round(mean_absolute_error(te["total"], t_pred), 2),
        "margin_sd": round(margin_sd, 2), "total_sd": round(total_sd, 2),
        "n_train": len(tr), "n_test": len(te),
    }
    # how much of the variance does the total model actually explain? (honesty)
    metrics["total_r2_vs_mean"] = round(
        1 - np.var(te["total"] - t_pred) / np.var(te["total"]), 3)
    metrics["margin_r2_vs_mean"] = round(
        1 - np.var(te["margin"] - m_pred) / np.var(te["margin"]), 3)
    return {"scaler": scaler, "win": win, "margin": margin, "total": total,
            "margin_sd": margin_sd, "total_sd": total_sd,
            "margin_bias": margin_bias, "total_bias": total_bias, "metrics": metrics}


def current_state() -> tuple[pd.DataFrame, dict]:
    """Latest pre-game rolling state per team + current Elo ratings."""
    games, tgs = _load()
    tl = _team_timeline(tgs)
    latest = tl.sort_values("game_date").groupby("team").tail(1).set_index("team")
    # current Elo = ratings after replaying every completed game
    ratings: dict[str, float] = {}
    k, home_adv, base = 20.0, 60.0, 1500.0
    for _, gm in games.iterrows():
        h, a = gm["home_team"], gm["away_team"]
        rh, ra = ratings.get(h, base), ratings.get(a, base)
        eh = 1.0 / (1.0 + 10 ** (-((rh + home_adv) - ra) / 400))
        hw = int(gm["home_score"] > gm["away_score"])
        ratings[h] = rh + k * (hw - eh)
        ratings[a] = ra + k * ((1 - hw) - (1 - eh))
    return latest, ratings


def upcoming_features(home: str, away: str, gdate: str, latest: pd.DataFrame,
                      elo: dict) -> dict | None:
    if home not in latest.index or away not in latest.index:
        return None
    h, a = latest.loc[home], latest.loc[away]

    def rest(row):
        d = (pd.to_datetime(gdate) - pd.to_datetime(row["game_date"])).days
        return float(np.clip(d, 1, 7))

    feats = {
        "d_form": h["form5"] - a["form5"], "d_off": h["off10"] - a["off10"],
        "d_def": h["def10"] - a["def10"], "d_net": h["net10"] - a["net10"],
        "d_rest": rest(h) - rest(a), "d_b2b": int(rest(h) == 1) - int(rest(a) == 1),
        "elo_diff": elo.get(home, 1500) - elo.get(away, 1500),
    }
    total_feats = {"h_off10": h["off10"], "a_off10": a["off10"],
                   "h_def10": h["def10"], "a_def10": a["def10"]}
    return {"game": feats, "total": total_feats}


def project_game(bundle: dict, feats: dict) -> dict:
    """Return model probabilities/means for one upcoming game."""
    sc = bundle["scaler"]
    Xg = pd.DataFrame([feats["game"]])[FEATURES].fillna(0)
    Xt = pd.DataFrame([feats["total"]])[TOTAL_FEATURES].fillna(0)
    p_home = float(bundle["win"].predict_proba(sc.transform(Xg))[0, 1])
    mu_margin = float(bundle["margin"].predict(sc.transform(Xg))[0]) - bundle.get("margin_bias", 0)
    mu_total = float(bundle["total"].predict(Xt)[0]) - bundle.get("total_bias", 0)
    return {"p_home_win": p_home, "mu_margin": mu_margin, "mu_total": mu_total,
            "margin_sd": bundle["margin_sd"], "total_sd": bundle["total_sd"]}


def p_home_cover(proj: dict, home_line: float) -> float:
    """P(home covers). home_line e.g. -3.5 => home must win by >3.5."""
    need = -home_line
    return float(1 - norm.cdf(need, loc=proj["mu_margin"], scale=proj["margin_sd"]))


def p_over(proj: dict, total_line: float) -> float:
    return float(1 - norm.cdf(total_line, loc=proj["mu_total"], scale=proj["total_sd"]))
