"""Ranking signal & alert logic (Stage 5, Section 5).

The ranking signal is CLV-predictive, not model-edge-driven: the primary
component is ANCHOR edge (book price vs sharp Pinnacle/consensus fair value),
because that is what actually correlates with beating the close. Model edge is a
weak secondary; pair mispricing is ranked on |joint - independent|.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

# weights: anchor (vs sharp) dominates; model edge is a weak tiebreak
W_ANCHOR, W_MODEL, W_CLV = 0.75, 0.15, 0.10
SCORE_MIN_SINGLE = 0.025
MISPRICING_MIN_PAIR = 0.04


def _clv_history(player: str, market: str) -> float:
    """Placeholder CLV-history metric (0 until a forward sample accrues)."""
    return 0.0


def rank_singles(anchor_edges: pd.DataFrame,
                 model_edges: pd.DataFrame | None = None) -> pd.DataFrame:
    if anchor_edges is None or anchor_edges.empty:
        return pd.DataFrame()
    df = anchor_edges.copy()
    mmap = {}
    if model_edges is not None and not model_edges.empty:
        for r in model_edges.itertuples():
            mmap[(r.player, r.market, r.side, float(r.line))] = r.edge
    df["model_edge"] = df.apply(
        lambda r: mmap.get((r["player"], r["market"], r["side"], float(r["line"])), 0.0), axis=1)
    df["clv_history"] = df.apply(lambda r: _clv_history(r["player"], r["market"]), axis=1)
    df["score"] = (W_ANCHOR * df["anchor_edge"] + W_MODEL * df["model_edge"]
                   + W_CLV * df["clv_history"])
    df = df[df["score"] >= SCORE_MIN_SINGLE]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def rank_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs is None or pairs.empty:
        return pd.DataFrame()
    p = pairs[pairs["mispricing"].abs() >= MISPRICING_MIN_PAIR].copy()
    p["score"] = p["mispricing"].abs()
    return p.sort_values("score", ascending=False).reset_index(drop=True)


def _reason_single(r) -> str:
    tags = [f"stale line vs {r['anchor_source']}"]
    if r.get("model_edge", 0) > 0.02:
        tags.append("model agrees")
    return ", ".join(tags)


def build_alerts(singles: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    """Construct notification payloads (consumed by Cowork / dashboard)."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    alerts = []
    for r in singles.to_dict("records") if not singles.empty else []:
        alerts.append({
            "ts": ts, "type": "single", "player": r["player"], "market": r["market"],
            "side": r["side"], "line": r["line"], "book": r["book"], "odds": r["odds"],
            "fair_prob": r["anchor_fair"], "edge": r["anchor_edge"],
            "score": round(r["score"], 3), "reason": _reason_single(r)})
    for r in pairs.to_dict("records") if not pairs.empty else []:
        alerts.append({
            "ts": ts, "type": "pair", "team": r["team"],
            "players": [r["player_a"], r["player_b"]], "leg": r["leg"], "rho": r["rho"],
            "p_joint": r["p_joint"], "p_independent": r["p_independent"],
            "mispricing": r["mispricing"], "score": round(r["score"], 3),
            "reason": r["direction"] + f" (corr {r['rho']:+.2f})"})
    return alerts
