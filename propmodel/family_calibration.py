"""Per-prop-family models + calibration with an auditable 'why chosen' summary.

For each family we (1) train a minutes x rate model, (2) pick the DISTRIBUTION
empirically from residual dispersion (Normal for points; NB if over-dispersed,
else Poisson for low/under-dispersed counts like steals/blocks), and (3) pick the
CALIBRATION method (raw vs isotonic vs Platt) by held-out Brier. Every choice is
recorded with a plain-language reason so it's not a black box.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from propmodel import distributions, models
from propmodel.featureset import EXTENDED_STATS, STATS, build

FAMILIES = STATS + EXTENDED_STATS
LABEL = {"points": "points", "rebounds": "rebounds", "assists": "assists",
         "tpm": "threes_made", "steals": "steals", "blocks": "blocks",
         "turnovers": "turnovers"}


# ---- distribution probability ------------------------------------------
def prob_over(mu: float, line: float, dist: str, param) -> float:
    mu = max(float(mu), 1e-6)
    if dist == "normal":
        return float(1 - norm.cdf(line, mu, max(np.sqrt(param), 1e-6)))
    k = int(np.ceil(line))
    if dist == "poisson":
        return float(1 - poisson.cdf(k - 1, mu))
    r = max(float(param), 1e-6)            # negative binomial
    return float(1 - nbinom.cdf(k - 1, r, r / (r + mu)))


def choose_distribution(family: str, actual: np.ndarray, mu: np.ndarray) -> tuple:
    """Return (dist_name, param, reason)."""
    if family == "points":
        v = float(np.var(actual - mu))
        return "normal", v, f"continuous-scale scoring -> Normal(var={v:.1f})"
    resid_var = float(np.var(actual - mu))
    mean_mu = float(np.mean(mu))
    ratio = resid_var / max(mean_mu, 1e-6)
    if ratio > 1.15:
        r = distributions.fit_nb_dispersion(actual, mu)
        return "nb", r, f"counts over-dispersed (resid var/mean={ratio:.2f}) -> NB(r={r:.1f})"
    return "poisson", None, (f"low/near-Poisson counts (resid var/mean={ratio:.2f}, "
                             f"not over-dispersed) -> Poisson, NB not justified")


# ---- calibration -------------------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _pairs(rows, actual_col, mu, dist, param):
    P, Y = [], []
    for mu_i, act in zip(mu, rows[actual_col].values):
        for k in (np.floor(mu_i) - 0.5, np.floor(mu_i) + 0.5, np.floor(mu_i) + 1.5):
            if k <= 0:
                continue
            P.append(prob_over(mu_i, k, dist, param))
            Y.append(int(act > k))
    return np.array(P), np.array(Y)


def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def fit_calibration(p_cal, y_cal, p_test, y_test) -> dict:
    """Fit isotonic + Platt on calib; pick the lower-Brier method on test."""
    raw_b = _brier(p_test, y_test)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, y_cal)
    iso_b = _brier(np.clip(iso.predict(p_test), 0, 1), y_test)
    platt = LogisticRegression().fit(_logit(p_cal).reshape(-1, 1), y_cal)
    platt_b = _brier(platt.predict_proba(_logit(p_test).reshape(-1, 1))[:, 1], y_test)

    options = {"raw": raw_b, "isotonic": iso_b, "platt": platt_b}
    best = min(options, key=options.get)
    # only prefer a calibrated method if it beats raw by a meaningful margin
    if best != "raw" and raw_b - options[best] < 0.001:
        best = "raw"
    return {"method": best, "brier": options, "iso": iso, "platt": platt}


# ---- driver ------------------------------------------------------------
def calibrate_all(test_frac=0.2, calib_frac=0.2) -> tuple[pd.DataFrame, dict]:
    from src.database import get_engine
    feat = build(get_engine())
    d = feat[feat.history >= 5].dropna(subset=["minutes"]).sort_values("game_date")
    dates = np.sort(d.game_date.unique())
    c1 = dates[int(len(dates) * (1 - test_frac - calib_frac))]
    c2 = dates[int(len(dates) * (1 - test_frac))]
    tr, cal, te = d[d.game_date < c1], d[(d.game_date >= c1) & (d.game_date < c2)], d[d.game_date >= c2]

    mm = models.train_minutes(tr)
    rms = {f: models.train_rate(tr, f) for f in FAMILIES}
    proj_cal = models.project(cal, mm, rms)
    proj_te = models.project(te, mm, rms)

    report, store = [], {"minutes": mm, "rates": rms, "families": {}}
    for f in FAMILIES:
        col = f"E_{f}"
        mc = cal.merge(proj_cal[["game_id", "player_id", col]], on=["game_id", "player_id"])
        mt = te.merge(proj_te[["game_id", "player_id", col]], on=["game_id", "player_id"])
        dist, param, dist_reason = choose_distribution(f, mt[f].values, mt[col].values)
        p_cal, y_cal = _pairs(mc, f, mc[col].values, dist, param)
        p_te, y_te = _pairs(mt, f, mt[col].values, dist, param)
        cal_res = fit_calibration(p_cal, y_cal, p_te, y_te)
        b = cal_res["brier"]
        why = (f"{dist_reason}. Calibration: raw Brier {b['raw']:.4f}, "
               f"isotonic {b['isotonic']:.4f}, Platt {b['platt']:.4f} -> "
               f"chose {cal_res['method'].upper()} "
               f"({'no method beat raw' if cal_res['method'] == 'raw' else 'best held-out Brier'}).")
        store["families"][f] = {"dist": dist, "param": param,
                                "calib_method": cal_res["method"],
                                "iso": cal_res["iso"], "platt": cal_res["platt"]}
        report.append({"family": LABEL[f], "n_test": len(mt), "distribution": dist,
                       "mean_proj": round(float(mt[col].mean()), 2),
                       "MAE": round(float(np.mean(np.abs(mt[f] - mt[col]))), 2),
                       "calib_method": cal_res["method"],
                       "brier_raw": round(b["raw"], 4),
                       "brier_chosen": round(b[cal_res["method"]], 4),
                       "why_chosen": why})
    return pd.DataFrame(report), store


def calibrated_prob_over(store: dict, family: str, mu: float, line: float):
    """Raw distribution prob -> per-family calibrated prob (None if family absent)."""
    fam = store["families"].get(family)
    if not fam or mu is None:
        return None, None
    raw = prob_over(float(mu), float(line), fam["dist"], fam["param"])
    return apply_calibration(store, family, raw), fam["calib_method"]


def sim_params_from_store(store: dict) -> dict:
    """Build the {nb_r, var_stat} params dict the combo simulator expects."""
    fams = store.get("families", {})
    nb_r, var_stat = {}, {}
    for f in ("rebounds", "assists"):
        if fams.get(f, {}).get("dist") == "nb":
            nb_r[f] = fams[f]["param"]
    if fams.get("points", {}).get("dist") == "normal":
        var_stat["points"] = fams["points"]["param"]
    return {"nb_r": nb_r, "var_stat": var_stat or {"points": 36.0}}


def apply_calibration(store: dict, family: str, raw_p: float) -> float:
    fam = store["families"].get(family)
    if not fam:
        return raw_p
    if fam["calib_method"] == "isotonic":
        return float(np.clip(fam["iso"].predict([raw_p])[0], 1e-4, 1 - 1e-4))
    if fam["calib_method"] == "platt":
        return float(fam["platt"].predict_proba(_logit(np.array([raw_p])).reshape(-1, 1))[0, 1])
    return raw_p
