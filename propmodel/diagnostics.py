"""Stage-3 diagnostics + CLV-prep odds archive (Section 8).

Truthful diagnostics: minutes/rate error, probability calibration (Brier,
log-loss, reliability), edge profile (median, tails, over/under skew), and a CLV
evaluator. Also a snapshot logger so we can build our own closing-line archive.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sqlalchemy import text

from src.database import get_engine


def evaluate_minutes_model(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return {"MAE": round(float(np.mean(np.abs(a - p))), 2),
            "RMSE": round(float(np.sqrt(np.mean((a - p) ** 2))), 2),
            "bias": round(float(np.mean(p - a)), 2)}


def evaluate_calibration(probs, outcomes, bins=8) -> dict:
    p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6)
    y = np.asarray(outcomes, int)
    brier = float(np.mean((p - y) ** 2))
    ll = float(log_loss(y, p, labels=[0, 1])) if len(set(y)) > 1 else float("nan")
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1))
    rel = df.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3)
    return {"brier": round(brier, 4), "log_loss": round(ll, 4), "reliability": rel}


def evaluate_edge_profile(edges: pd.DataFrame) -> dict:
    if edges.empty:
        return {"n": 0}
    e = edges["edge"]
    return {"n": len(edges), "median": round(float(e.median()), 3),
            "p90": round(float(e.quantile(0.9)), 3), "max": round(float(e.max()), 3),
            "share_gt5pct": round(float((e > 0.05).mean()), 2),
            "share_gt7pct": round(float((e > 0.07).mean()), 2),
            "over_share": round(float((edges["side"] == "over").mean()), 2)}


def evaluate_clv(pred_prob, closing_prob, outcomes) -> dict:
    """CLV proxy: did we beat the closing implied probability, on average?"""
    pred, close = np.asarray(pred_prob, float), np.asarray(closing_prob, float)
    return {"mean_prob_edge_vs_close": round(float(np.mean(pred - close)), 4),
            "share_beat_close": round(float(np.mean(pred > close)), 2),
            "n": len(pred)}


def log_odds_snapshot(props: pd.DataFrame) -> int:
    """Append a timestamped multi-book odds snapshot for future CLV backtests."""
    if props is None or props.empty:
        return 0
    eng = get_engine()
    snap = props.copy()
    snap["snapshot_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE IF NOT EXISTS odds_snapshots ("
                       "snapshot_ts TEXT, date TEXT, player TEXT, market TEXT, "
                       "side TEXT, line REAL, book TEXT, odds REAL)"))
    cols = ["snapshot_ts", "date", "player", "market", "side", "line", "book", "odds"]
    snap[[c for c in cols if c in snap.columns]].to_sql(
        "odds_snapshots", eng, if_exists="append", index=False)
    return len(snap)
