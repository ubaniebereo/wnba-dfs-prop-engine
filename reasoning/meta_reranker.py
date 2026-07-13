"""Layer 3c — optional meta-model reranker (Section 7).

When enough LABELED outcomes accrue (beat_close / clv_positive / realized hit),
a small interpretable model (logistic / HistGB) can learn final ranking quality
from the stacked features below. We do NOT have that labeled sample yet (the CLV
backtest is ~46 bets), so this is a transparent hook: it trains if a labeled
table exists, else the decision_engine's rule-based score is used as-is.
"""

from __future__ import annotations

import pandas as pd

META_FEATURES = ["model_fair_prob", "anchor_edge", "model_edge", "confidence_score",
                 "standout_score", "decision_confidence", "trust_adjustment",
                 "is_starter", "starter_rate", "opp_pace", "opp_def_vs_pos",
                 "proj_sd", "line_delta"]
LABEL_TABLE = "labeled_outcomes"   # columns: prediction_id, beat_close (0/1)


def available() -> bool:
    try:
        from feeds.storage import ENGINE
        n = pd.read_sql(f"SELECT COUNT(*) n FROM {LABEL_TABLE}", ENGINE)["n"].iloc[0]
        return int(n) >= 200          # need a real sample before trusting a meta-model
    except Exception:
        return False


def train_and_score(scan_df: pd.DataFrame) -> pd.DataFrame:
    """If a labeled sample exists, fit a logistic meta-model on stacked features
    and write `meta_score`; otherwise return scan_df unchanged (documented)."""
    if not available():
        return scan_df          # not enough labels -> rule-based decision stands
    from sklearn.linear_model import LogisticRegression
    from feeds.storage import ENGINE
    lab = pd.read_sql(f"SELECT * FROM {LABEL_TABLE}", ENGINE)
    feats = [c for c in META_FEATURES if c in scan_df.columns]
    train = scan_df.merge(lab, on="prediction_id", how="inner").dropna(subset=feats)
    if len(train) < 200:
        return scan_df
    clf = LogisticRegression(max_iter=500).fit(train[feats].fillna(0), train["beat_close"])
    scan_df = scan_df.copy()
    scan_df["meta_score"] = clf.predict_proba(scan_df[feats].fillna(0))[:, 1].round(3)
    return scan_df.sort_values("meta_score", ascending=False)
