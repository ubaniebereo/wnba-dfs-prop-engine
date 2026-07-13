"""Monte Carlo simulation layer (Section 7).

Single props: draw from the fitted marginal (Normal for points, NB-implied SD for
counts) -> percentiles + over/under. Combos (PRA/PR/PA/RA): draw points, rebounds,
assists JOINTLY using the pooled within-player same-game correlation, so combo
probabilities respect covariance instead of naive summing. Fantasy score uses the
same joint draw with a (partial) scoring formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.database import get_engine

COMPONENTS = ["points", "rebounds", "assists"]
COMBO_PARTS = {"player_pra": ["points", "rebounds", "assists"],
               "player_pr": ["points", "rebounds"],
               "player_pa": ["points", "assists"],
               "player_ra": ["rebounds", "assists"]}
# WNBA-style fantasy weights (pts/reb/ast portion; stl/blk/to not modeled yet)
FANTASY_W = {"points": 1.0, "rebounds": 1.2, "assists": 1.5}

_CORR_CACHE: dict = {}


def pooled_component_corr() -> np.ndarray:
    """3x3 within-player same-game correlation of (points, rebounds, assists)."""
    if "R" in _CORR_CACHE:
        return _CORR_CACHE["R"]
    df = pd.read_sql("SELECT player_id, points, rebounds, assists FROM "
                     "player_game_stats WHERE minutes > 0", get_engine())
    for c in COMPONENTS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=COMPONENTS)
    # demean within player to isolate same-game co-movement
    dem = df.groupby("player_id")[COMPONENTS].transform(lambda s: s - s.mean())
    R = np.corrcoef(dem[COMPONENTS].values.T)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    _CORR_CACHE["R"] = R
    return R


def _sd_for(market: str, mu: float, params: dict) -> float:
    if market in ("rebounds", "assists"):
        r = params["nb_r"].get(market) if isinstance(params.get("nb_r"), dict) else None
        r = r if r else 20.0
        return float(np.sqrt(max(mu, 1e-6) + mu ** 2 / r))   # NB variance
    return float(np.sqrt(params["var_stat"].get(market, 36.0)))  # Normal


def single_prop_sim(market: str, mu: float, line: float, params: dict,
                    n=20000, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    sd = _sd_for(market, mu, params)
    draws = np.clip(rng.normal(mu, sd, n), 0, None)
    pct = np.percentile(draws, [10, 25, 50, 75, 90])
    return {"mean": round(float(draws.mean()), 2),
            "p10": round(pct[0], 1), "p25": round(pct[1], 1), "p50": round(pct[2], 1),
            "p75": round(pct[3], 1), "p90": round(pct[4], 1),
            "prob_over": round(float((draws > line).mean()), 3),
            "sd": round(sd, 2)}


def _joint_draw(means: dict, params: dict, n: int, rng) -> dict:
    """Correlated draws of points/rebounds/assists for one player."""
    R = pooled_component_corr()
    L = np.linalg.cholesky(R + 1e-9 * np.eye(3))
    z = (L @ rng.standard_normal((3, n)))
    out = {}
    for i, c in enumerate(COMPONENTS):
        mu = means.get(c)
        if mu is None or pd.isna(mu):
            return {}
        out[c] = np.clip(mu + _sd_for(c, mu, params) * z[i], 0, None)
    return out


def combo_sim(canonical: str, means: dict, line: float, params: dict,
              n=20000, rng=None) -> dict | None:
    rng = rng or np.random.default_rng(0)
    draws = _joint_draw(means, params, n, rng)
    if not draws:
        return None
    if canonical == "player_fantasy_score":
        total = sum(FANTASY_W[c] * draws[c] for c in COMPONENTS)
    else:
        total = sum(draws[c] for c in COMBO_PARTS[canonical])
    pct = np.percentile(total, [10, 25, 50, 75, 90])
    return {"mean": round(float(total.mean()), 2),
            "p10": round(pct[0], 1), "p25": round(pct[1], 1), "p50": round(pct[2], 1),
            "p75": round(pct[3], 1), "p90": round(pct[4], 1),
            "prob_over": round(float((total > line).mean()), 3),
            "approx": canonical == "player_fantasy_score"}
