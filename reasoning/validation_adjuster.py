"""Layer 4 — validation adjustment (Section 4/6).

Adjusts trust using prop-family model health (calibration Brier + whether
calibration helped) and the row's reliability evidence. Returns a multiplier in
~[0.6, 1.1] applied to the standout score, plus a short rationale.
"""

from __future__ import annotations

import pandas as pd

# family-metrics use short names; scan markets use these too
_FAMILY_KEYS = {"points": "points", "rebounds": "rebounds", "assists": "assists",
                "player_points": "points", "player_rebounds": "rebounds",
                "player_assists": "assists"}


def load_family_health() -> dict:
    """Map family -> {brier, calib_method, validation} from prop_family_metrics."""
    try:
        from feeds.storage import ENGINE
        df = pd.read_sql("SELECT * FROM prop_family_metrics", ENGINE)
    except Exception:
        return {}
    out = {}
    for r in df.to_dict("records"):
        b = r.get("brier_chosen", None)
        out[r["family"]] = {
            "brier": b, "calib_method": r.get("calib_method"),
            "validation": ("Validated" if b is not None and b < 0.18 else
                           "Promising" if b is not None and b < 0.21 else
                           "Noisy" if b is not None and b < 0.24 else "Model-only")}
    return out


def trust_adjustment(row: dict, evidence_scores: dict, health: dict) -> dict:
    fam = _FAMILY_KEYS.get(row.get("market_type"), row.get("market_type"))
    h = health.get(fam, {})
    adj, reasons = 1.0, []

    brier = h.get("brier")
    if brier is not None:
        if brier < 0.18:
            adj *= 1.08; reasons.append(f"{fam} model is well-calibrated (Brier {brier})")
        elif brier > 0.235:
            adj *= 0.85; reasons.append(f"{fam} model is only moderately calibrated (Brier {brier})")
    # reliability penalties from evidence (high variance, wide range, feature gaps)
    pen = evidence_scores.get("reliability_penalty", 0.0)
    if pen > 0:
        adj *= max(0.7, 1 - 0.12 * pen)
        reasons.append("downgraded for reliability concerns")
    adj = round(float(min(1.1, max(0.6, adj))), 3)
    return {"trust_adjustment": adj, "validation_status": h.get("validation", "n/a"),
            "trust_reasons": reasons}
