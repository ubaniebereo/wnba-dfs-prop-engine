"""Distributional prop modeling (Stage 3, Section 4).

Minutes x rate variance via law of total variance, and proper count
distributions (Negative Binomial via MLE with position-cluster shrinkage).
  Var(X=M*R) ~ E[M]^2 Var(R) + E[R]^2 Var(M) + Var(M) Var(R)
  points          -> Normal(E_stat, Var_stat)
  rebounds/assists-> NegBinomial(mean=E_stat, dispersion r)   [Var = mu + mu^2/r]
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import nbinom, norm

COUNT_STATS = {"rebounds", "assists"}


# --- Negative Binomial dispersion via MLE -------------------------------
def _nb_nll(r: float, actual: np.ndarray, mu: np.ndarray) -> float:
    r = max(r, 1e-6)
    mu = np.clip(mu, 1e-6, None)
    return -np.sum(gammaln(actual + r) - gammaln(r) - gammaln(actual + 1)
                   + r * np.log(r / (r + mu)) + actual * np.log(mu / (r + mu)))


def fit_nb_dispersion(actual: np.ndarray, mu: np.ndarray) -> float:
    actual = np.asarray(actual, float)
    mu = np.asarray(mu, float)
    res = minimize_scalar(_nb_nll, bounds=(0.5, 300), method="bounded",
                          args=(actual, mu))
    return float(res.x)


def fit_nb_by_cluster(actual, mu, cluster, shrink=0.3) -> dict:
    """Per-cluster dispersion, shrunk toward the global MLE (partial pooling)."""
    actual, mu = np.asarray(actual, float), np.asarray(mu, float)
    cluster = np.asarray(cluster)
    r_global = fit_nb_dispersion(actual, mu)
    out = {"_global": r_global}
    for cl in np.unique(cluster):
        m = cluster == cl
        if m.sum() < 50:
            out[cl] = r_global
        else:
            r_cl = fit_nb_dispersion(actual[m], mu[m])
            out[cl] = shrink * r_global + (1 - shrink) * r_cl
    return out


# --- law of total variance ----------------------------------------------
def total_variance(e_min, var_min, e_rate, var_rate):
    return e_min ** 2 * var_rate + e_rate ** 2 * var_min + var_min * var_rate


# --- probabilities -------------------------------------------------------
def prob_over_normal(mu: float, var: float, line: float) -> float:
    return float(1 - norm.cdf(line, loc=mu, scale=max(np.sqrt(var), 1e-6)))


def prob_over_nb(mu: float, r: float, line: float) -> float:
    mu, r = max(mu, 1e-6), max(r, 1e-6)
    p = r / (r + mu)
    k = int(np.ceil(line))                       # X.5 line -> need >= ceil
    return float(1 - nbinom.cdf(k - 1, r, p))


def prob_over(stat: str, mu: float, line: float, params: dict) -> float:
    rmap = params.get("nb_r", {}).get(stat)
    if stat in COUNT_STATS and rmap is not None:
        r = rmap.get(params.get("cluster"), rmap["_global"]) if isinstance(rmap, dict) else rmap
        return prob_over_nb(mu, r, line)
    return prob_over_normal(mu, params["var_stat"][stat], line)   # Normal fallback
