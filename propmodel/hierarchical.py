"""Hierarchical partial pooling for per-minute rates (Stage 3, Section 5).

statsmodels MixedLM: rate ~ C(position_cluster) + (1 | player). Player random
intercepts (BLUPs) shrink low-sample players toward their position/league mean
while letting stars keep elevated rates. Faster than full Bayesian at ~30k rows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")


def fit_player_pool(df: pd.DataFrame, stat: str, min_minutes=8) -> pd.DataFrame:
    """Return per-player pooled per-minute rate (global + position + player RE)."""
    d = df[df["minutes"] >= min_minutes].copy()
    d["rate"] = d[stat] / d["minutes"].clip(lower=1)
    d["position_cluster"] = d["position_group"].fillna("wing")
    md = smf.mixedlm("rate ~ C(position_cluster)", d, groups=d["player_id"])
    res = md.fit(method="lbfgs", disp=False)
    # fixed-effect (global + position) prediction per row, then add player RE
    fe = res.predict(d)
    re = res.random_effects
    base = pd.DataFrame({"player_id": d["player_id"].values, "fe": fe.values})
    pooled = base.groupby("player_id")["fe"].mean()
    pooled = pooled + pd.Series({k: float(np.ravel(v)[0]) for k, v in re.items()})
    out = pd.DataFrame({"player_id": pooled.index,
                        f"pooled_rate_{stat}": pooled.values})
    # raw (unpooled) player rate for comparison
    raw = d.groupby("player_id").apply(lambda x: x[stat].sum() / x["minutes"].sum())
    out[f"raw_rate_{stat}"] = out["player_id"].map(raw)
    out["games"] = out["player_id"].map(d.groupby("player_id").size())
    return out


def evaluate_pooling(df: pd.DataFrame, stat: str, test_frac=0.25) -> dict:
    """Does pooling beat the raw player mean at predicting held-out per-min rate?"""
    d = df[df["minutes"] >= 8].copy()
    dates = np.sort(d["game_date"].unique())
    cut = dates[int(len(dates) * (1 - test_frac))]
    tr, te = d[d.game_date < cut], d[d.game_date >= cut].copy()
    pool = fit_player_pool(tr, stat)
    te["actual_rate"] = te[stat] / te["minutes"].clip(lower=1)
    te = te.merge(pool, on="player_id", how="inner")
    mae_pool = float(np.mean(np.abs(te["actual_rate"] - te[f"pooled_rate_{stat}"])))
    mae_raw = float(np.mean(np.abs(te["actual_rate"] - te[f"raw_rate_{stat}"])))
    low = te[te["games"] <= 10]
    return {"stat": stat, "MAE_pooled": round(mae_pool, 4), "MAE_raw": round(mae_raw, 4),
            "low_sample_rate_sd_raw": round(float(low[f"raw_rate_{stat}"].std()), 4),
            "low_sample_rate_sd_pooled": round(float(low[f"pooled_rate_{stat}"].std()), 4)}
