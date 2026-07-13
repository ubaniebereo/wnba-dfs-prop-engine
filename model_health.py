"""Self-improvement / model-health subsystem (Section 11).

Trains + evaluates the per-family models, scores each family (MAE, calibration
Brier, calibration gain, sample size), extracts rate-model feature importance
(permutation), and generates concrete improvement recommendations. Everything is
persisted to feeds.sqlite so the Model Health dashboard tab can read it.

Tables written:
  model_eval_runs, prop_family_metrics, feature_importance_snapshots,
  calibration_metrics, improvement_recommendations
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from propmodel.family_calibration import LABEL, calibrate_all
from propmodel.featureset import build, rate_features
from src.config import MODELS_DIR
from src.database import get_engine
from src.utils import get_logger

log = get_logger(__name__)


def _feature_importance(store: dict) -> pd.DataFrame:
    """Permutation importance of rate features per family (on a holdout sample)."""
    try:
        feat = build(get_engine())
        d = feat[feat.history >= 5].dropna(subset=["minutes"])
        sample = d.tail(3000)
        rows = []
        for fam, rm in store["rates"].items():
            cols = rate_features(fam)
            cols = [c for c in cols if c in sample.columns]
            X = sample[cols].fillna(sample[cols].median())
            y = (sample[fam] / sample["minutes"].clip(lower=1)) if fam in sample else None
            if y is None:
                continue
            pi = permutation_importance(rm["model"], X, y, n_repeats=3,
                                        random_state=0, n_jobs=-1)
            for c, imp in sorted(zip(cols, pi.importances_mean),
                                 key=lambda t: -t[1])[:5]:
                rows.append({"family": LABEL.get(fam, fam), "feature": c,
                             "importance": round(float(imp), 5)})
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        log.warning("feature importance failed: %s", exc)
        return pd.DataFrame()


def generate_recommendations(report: pd.DataFrame, fi: pd.DataFrame) -> list[dict]:
    recs = []
    rep = report.copy()
    rep["err_ratio"] = rep["MAE"] / rep["mean_proj"].clip(lower=0.1)
    # weakest family by relative error
    worst = rep.sort_values("err_ratio", ascending=False).iloc[0]
    recs.append({"family": worst["family"], "priority": "high",
                 "recommendation": f"Highest relative error (MAE {worst['MAE']} on mean "
                 f"{worst['mean_proj']}); improve mean projection with matchup/usage features."})
    # calibration that actually helped
    helped = rep[rep["brier_raw"] - rep["brier_chosen"] > 0.001]
    for r in helped.itertuples():
        recs.append({"family": r.family, "priority": "info",
                     "recommendation": f"{r.calib_method} calibration improved Brier "
                     f"{r.brier_raw}->{r.brier_chosen}; keep it."})
    # families where no method beat raw (well-calibrated already or need features)
    raw_fams = rep[rep["calib_method"] == "raw"]["family"].tolist()
    if raw_fams:
        recs.append({"family": ",".join(raw_fams), "priority": "info",
                     "recommendation": "No calibration beat raw -> already reasonably "
                     "calibrated; gains must come from better features, not calibration."})
    # under-dispersed counts flagged to Poisson
    pois = rep[rep["distribution"] == "poisson"]["family"].tolist()
    if pois:
        recs.append({"family": ",".join(pois), "priority": "info",
                     "recommendation": "Modeled as Poisson (not over-dispersed). If tails "
                     "underperform, add defensive-activity / minutes-floor features."})
    # thin-sample / high-Brier note for the worst-calibrated relative to base
    hi_brier = rep.sort_values("brier_chosen", ascending=False).iloc[0]
    recs.append({"family": hi_brier["family"], "priority": "medium",
                 "recommendation": f"Highest Brier ({hi_brier['brier_chosen']}); most room "
                 f"to improve probability quality — prioritize feature work here."})
    return recs


def run_health() -> dict:
    run_id = uuid.uuid4().hex
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.info("model-health run: training + evaluating per-family models...")
    report, store = calibrate_all()
    joblib.dump(store, MODELS_DIR / "family_calibration.joblib")   # refresh scanner models
    fi = _feature_importance(store)
    recs = generate_recommendations(report, fi)

    from feeds.storage import ENGINE as eng, init_storage   # dashboard reads feeds.sqlite
    init_storage()
    fam = report.copy()
    fam["run_id"] = run_id
    fam["evaluated_at"] = ts
    fam.to_sql("prop_family_metrics", eng, if_exists="replace", index=False)
    if not fi.empty:
        fi["run_id"] = run_id
        fi.to_sql("feature_importance_snapshots", eng, if_exists="replace", index=False)
    rdf = pd.DataFrame(recs)
    rdf["run_id"] = run_id
    rdf["created_at"] = ts
    rdf.to_sql("improvement_recommendations", eng, if_exists="replace", index=False)
    pd.DataFrame([{"run_id": run_id, "evaluated_at": ts, "n_families": len(report),
                   "mean_MAE": round(float(report["MAE"].mean()), 3),
                   "mean_brier": round(float(report["brier_chosen"].mean()), 4)}]).to_sql(
        "model_eval_runs", eng, if_exists="append", index=False)
    return {"run_id": run_id, "report": report, "feature_importance": fi,
            "recommendations": recs}


if __name__ == "__main__":
    out = run_health()
    pd.set_option("display.width", 200, "display.max_colwidth", 120)
    print("\n=== PER-FAMILY METRICS ===")
    print(out["report"][["family", "distribution", "MAE", "calib_method",
                         "brier_raw", "brier_chosen"]].to_string(index=False))
    print("\n=== TOP FEATURES BY FAMILY ===")
    if not out["feature_importance"].empty:
        print(out["feature_importance"].to_string(index=False))
    print("\n=== IMPROVEMENT RECOMMENDATIONS ===")
    for r in out["recommendations"]:
        print(f"  [{r['priority']:6s}] {r['family']}: {r['recommendation']}")
