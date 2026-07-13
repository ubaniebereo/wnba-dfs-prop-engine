"""Layer 3 — decision engine + annotator (Section 4/6).

Combines model output + evidence + validation into the final decision fields and
attaches grounded English. The raw model output, calibrated probability, the
reasoning decision, and the validation-adjusted trust are kept SEPARATE so the UI
can show all four (Section 6/12).

`annotate(scan_df)` adds every reasoning output column to the scan.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import explanations as ex
from .evidence_builder import build_evidence, evidence_score
from .validation_adjuster import load_family_health, trust_adjustment


def _feature_drivers(ev: dict) -> dict:
    pos = [{"k": i["key"], "v": i["value"], "phrase": i["phrase"]}
           for b in ("opportunity", "context") for i in ev[b] if i["polarity"] > 0]
    neg = [{"k": i["key"], "v": i["value"], "phrase": i["phrase"]}
           for b in ("opportunity", "context") for i in ev[b] if i["polarity"] < 0]
    unc = [{"k": i["key"], "v": i["value"], "phrase": i["phrase"]} for i in ev["reliability"]]
    return {"positive": pos[:5], "negative": neg[:5], "uncertainty": unc[:5]}


def decide_row(row: dict, health: dict) -> dict:
    side = row.get("side", "over")
    ev = build_evidence(row)
    es = evidence_score(ev)
    val = trust_adjustment(row, es, health)

    base = float(row.get("standout_score") or 0.0)
    # reasoning nudges the score by net evidence (bounded), then validation scales trust
    reasoned = base * (1 + 0.08 * np.tanh(es["net"]))
    final_score = round(float(reasoned * val["trust_adjustment"]), 1)

    base_conf = float(row.get("confidence_score") or 50.0)
    decision_conf = round(float(np.clip(
        base_conf * val["trust_adjustment"] - 6 * es["reliability_penalty"], 0, 100)), 1)

    drivers = _feature_drivers(ev)
    verdict = ("LEAN " + side.upper()) if final_score >= 18 and es["net"] >= 0 else \
              ("WATCH" if final_score >= 10 else "PASS")
    return {
        "final_standout_score": final_score,
        "final_verdict": verdict,
        "decision_confidence": decision_conf,
        "trust_adjustment": val["trust_adjustment"],
        "validation_status": val["validation_status"],
        "verdict_summary": ex.verdict_summary(row, ev, side, final_score, decision_conf),
        "why_it_stands_out": " • ".join(ex.why_it_stands_out(ev, side)) or "—",
        "why_confidence_is_limited": " • ".join(ex.why_confidence_is_limited(ev)) or "—",
        "game_context_summary": ex.game_context_summary(row, ev),
        "reason_tags": ",".join(ex.reason_tags(ev)),
        "risk_tags": ",".join(ex.risk_tags(ev, row)),
        "feature_drivers_json": json.dumps(drivers),
        "evidence_scores_json": json.dumps(es),
    }


def annotate(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Add reasoning/explanation columns to every scan row (grounded English)."""
    if scan_df is None or scan_df.empty:
        return scan_df
    health = load_family_health()
    out = scan_df.copy()
    decisions = [decide_row(r, health) for r in out.to_dict("records")]
    dec = pd.DataFrame(decisions, index=out.index)
    for c in dec.columns:
        out[c] = dec[c]
    # re-rank by the reasoning-adjusted score (raw standout kept for comparison)
    return out.sort_values("final_standout_score", ascending=False).reset_index(drop=True)
