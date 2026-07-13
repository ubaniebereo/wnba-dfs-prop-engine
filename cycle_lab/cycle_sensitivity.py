"""Cycle-sensitivity modeling: separate a faint cycle signal from fatigue noise.

Pipeline:
  deltas      -> within-player deviations from baseline (the SIGNAL features)
  selection   -> games/phase-coverage/role-stability/fatigue-variation pool
  per-player  -> OLS of delta ~ phase + symptoms + FATIGUE controls + context
  scoring     -> standardized phase-effect magnitude x robustness-after-fatigue
  validation  -> recovery of planted ground truth (precision/recall)
  hierarchical-> pooled MixedLM (random player intercepts) for league effect
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")

METRICS = ["delta_TS", "delta_REB", "delta_PIR", "delta_pts"]
FATIGUE = ["days_rest", "b2b_flag", "game_in_week", "cumulative_minutes_last_3"]
CONTEXT = ["minutes", "opponent_strength", "home_flag", "usage_rate", "contact_intensity"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["position_group"] = np.where(df["position"].isin(["PF", "C"]), "big", "perimeter")
    df["contact_intensity"] = (df["REB"] + df.get("BLK", 0) + df.get("PF", 0)) / df["minutes"].clip(lower=1)
    df["home_flag"] = (df["home_away"] == "home").astype(int)
    df["symptoms_high"] = (df["symptom_score"] >= 6).astype(int)
    df["recovery_high"] = (df.get("perceived_recovery", 0) >= 7).astype(int)
    # within-player baselines -> deltas (the cycle SIGNAL)
    g = df.groupby("player_id")
    for col, d in [("TS", "delta_TS"), ("eFG", "delta_eFG"), ("REB", "delta_REB"),
                   ("PIR", "delta_PIR"), ("points", "delta_pts")]:
        df[d] = df[col] - g[col].transform("mean")
    return df


def select_players(df: pd.DataFrame, min_games=25, min_per_phase=5,
                   role_cv_max=0.45) -> pd.DataFrame:
    """Candidate pool: enough games, phase coverage, stable role, fatigue variation."""
    keep = []
    for pid, d in df.groupby("player_id"):
        if len(d) < min_games:
            continue
        ph = d["cycle_phase"].value_counts()
        if ph.min() < min_per_phase or len(ph) < 3:
            continue
        role_cv = d["minutes"].std() / max(d["minutes"].mean(), 1)
        fatigue_var = d["days_rest"].std() > 0 and d["b2b_flag"].nunique() > 1
        if role_cv > role_cv_max or not fatigue_var:
            continue
        keep.append(pid)
    return df[df["player_id"].isin(keep)].copy(), keep


# ---------------------------------------------------------------------------
# Per-player regression with fatigue control
# ---------------------------------------------------------------------------
def _fit_metric(d: pd.DataFrame, target: str):
    """OLS: delta ~ phase(ref=follicular) + symptoms + fatigue + context.
    Returns (menstruation_coef, luteal_coef, min_phase_p, full_R2, robust_frac)."""
    base = ("C(cycle_phase, Treatment(reference='follicular')) + symptoms_high "
            "+ recovery_high")
    ctrl = " + ".join(FATIGUE + CONTEXT)
    try:
        full = smf.ols(f"{target} ~ {base} + {ctrl}", data=d).fit()
        # model WITHOUT fatigue, to measure how much fatigue control shrinks effect
        nofat = smf.ols(f"{target} ~ {base}", data=d).fit()
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    men_key = [k for k in full.params.index if "menstruation" in k]
    lut_key = [k for k in full.params.index if "luteal" in k]
    men = float(full.params[men_key[0]]) if men_key else np.nan
    lut = float(full.params[lut_key[0]]) if lut_key else np.nan
    pmin = float(np.nanmin([full.pvalues.get(k, np.nan) for k in men_key + lut_key])) \
        if (men_key + lut_key) else np.nan
    # robustness: |effect with fatigue| / |effect without| (1.0 = fully robust)
    men0 = float(nofat.params[men_key[0]]) if men_key else np.nan
    robust = abs(men) / abs(men0) if men0 and not np.isnan(men0) else np.nan
    return men, lut, pmin, full.rsquared, min(robust, 1.0) if robust == robust else np.nan


def fit_players(df_pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, d in df_pool.groupby("player_id"):
        rec = {"player_id": pid, "player_name": d["player_name"].iloc[0],
               "position_group": d["position_group"].iloc[0],
               "n_games": len(d), "med_use": int(d["med_use"].iloc[0]),
               "gt_cycle_sensitive": int(d["gt_cycle_sensitive"].iloc[0])}
        std_effects, robusts, pmins = [], [], []
        for tgt in METRICS:
            men, lut, pmin, r2, robust = _fit_metric(d, tgt)
            sd = d[tgt].std() or 1.0
            rec[f"{tgt}_men_coef"] = round(men, 4) if men == men else np.nan
            # standardized menstruation effect (in within-player SD units)
            std_eff = men / sd if (men == men and sd) else np.nan
            rec[f"{tgt}_men_z"] = round(std_eff, 3) if std_eff == std_eff else np.nan
            if std_eff == std_eff:
                std_effects.append(abs(std_eff))
            if robust == robust:
                robusts.append(robust)
            if pmin == pmin:
                pmins.append(pmin)
        # cycle-sensitivity score: mean |standardized effect| x robustness x significance
        mag = np.mean(std_effects) if std_effects else 0.0
        robust = np.mean(robusts) if robusts else 0.0
        sig = 1 - np.nanmin(pmins) if pmins else 0.0
        rec["robust_after_fatigue"] = round(robust, 3)
        rec["min_phase_p"] = round(np.nanmin(pmins), 4) if pmins else np.nan
        rec["cycle_sensitivity_score"] = round(mag * robust * sig, 4)
        # alignment with research: menstruation should WORSEN efficiency/strength
        eff_signs = [rec.get(f"{m}_men_coef", np.nan) for m in ("delta_TS", "delta_PIR", "delta_REB")]
        neg = np.nansum([1 for s in eff_signs if s == s and s < 0])
        pos = np.nansum([1 for s in eff_signs if s == s and s > 0])
        rec["alignment"] = ("aligned" if neg >= 2 and pos == 0 else
                            "deviates" if pos >= 2 and neg == 0 else "partial")
        rows.append(rec)
    out = pd.DataFrame(rows)
    # flag: score above data-driven threshold AND effect survives fatigue control
    thr = out["cycle_sensitivity_score"].quantile(0.80)
    out["cycle_sensitive"] = ((out["cycle_sensitivity_score"] >= thr) &
                              (out["robust_after_fatigue"] >= 0.5)).astype(bool)
    return out.sort_values("cycle_sensitivity_score", ascending=False)


# ---------------------------------------------------------------------------
# Hierarchical pooled model (league-level cycle effect after fatigue control)
# ---------------------------------------------------------------------------
def hierarchical_effect(df_pool: pd.DataFrame, target="delta_TS") -> pd.DataFrame:
    """MixedLM with random player intercepts; fixed phase + fatigue + context."""
    ctrl = " + ".join(FATIGUE + CONTEXT)
    formula = (f"{target} ~ C(cycle_phase, Treatment(reference='follicular')) "
               f"+ symptoms_high + {ctrl}")
    md = smf.mixedlm(formula, df_pool, groups=df_pool["player_id"])
    res = md.fit(method="lbfgs", disp=False)
    keep = [k for k in res.params.index if "cycle_phase" in k or "symptoms" in k]
    return pd.DataFrame({"term": keep,
                         "coef": [round(res.params[k], 4) for k in keep],
                         "p_value": [round(res.pvalues[k], 4) for k in keep]})
