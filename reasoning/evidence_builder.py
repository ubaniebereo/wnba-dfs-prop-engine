"""Layer 2 — evidence extraction (Section 5).

Turns a scan row's REAL feature values into structured evidence buckets. Every
item is grounded in a value actually present on the row; nothing is invented.
Each item: (key, polarity, weight, value, phrase). Polarity is for the chosen
SIDE of the prop (over/under).
"""

from __future__ import annotations

import numpy as np

# WNBA-ish reference points for context comparisons (transparent, tunable).
PACE_AVG = 96.0          # possessions/game proxy
DEF_RTG_AVG = 100.0
HIGH_VAR_MARKETS = {"steals", "blocks", "turnovers", "threes_made", "player_threes_made"}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_evidence(row: dict) -> dict:
    """Return {opportunity:[], context:[], reliability:[], entry:[]} evidence."""
    side = row.get("side", "over")
    over = side == "over"
    ev = {"opportunity": [], "context": [], "reliability": [], "entry": []}

    def add(bucket, key, polarity, weight, value, phrase):
        ev[bucket].append({"key": key, "polarity": polarity, "weight": weight,
                           "value": value, "phrase": phrase})

    # ---- A. opportunity ----
    ae = _num(row.get("anchor_edge")) or 0.0
    if ae >= 0.02:
        add("opportunity", "stale_vs_anchor", +1, 1.0, round(ae, 3),
            f"the price beats the sharp market by {ae*100:.1f}% (line looks stale)")
    # NOTE: news on a row is about THAT player. Their own OUT/doubtful status is an
    # availability RISK (they may not play), NOT an edge. 'Usage shift' would require
    # teammate-out tracking, which we don't have per-row -> we don't claim it.
    nsv = row.get("news_severity")
    ns = nsv.lower() if isinstance(nsv, str) else ""   # column may be NaN(float) when empty
    if ns in ("out", "scratch"):
        add("reliability", "player_out", -1, 1.5, ns,
            "player is flagged OUT — likely won't play (avoid; line may be voided)")
    elif ns in ("doubtful", "questionable"):
        add("reliability", "availability", -1, 1.0, ns,
            f"availability is uncertain ({ns} tonight)")
    elif row.get("news_flag"):
        add("opportunity", "news", +1, 0.5, ns,
            "recent news flagged for this player — monitor for a role change")
    if row.get("line_move_flag"):
        add("opportunity", "line_move", +1, 0.6, row.get("line_delta"),
            "the line moved recently (possible information edge)")

    # ---- B. basketball context (grounded in real feature values) ----
    l3, savg = _num(row.get("last_3_min")), _num(row.get("season_avg_min"))
    if l3 is not None and savg is not None:
        d = l3 - savg
        if d >= 1.5:
            add("context", "minutes_trend", +1 if over else -1, 0.9, round(d, 1),
                f"minutes are trending up (last-3 {l3:.0f} vs season {savg:.0f})")
        elif d <= -1.5:
            add("context", "minutes_trend", -1 if over else +1, 0.9, round(d, 1),
                f"minutes are trending down (last-3 {l3:.0f} vs season {savg:.0f})")
    if row.get("is_starter") == 1:
        add("context", "role", +1 if over else -1, 0.5, 1, "is in the starting five")
    sr = _num(row.get("starter_rate"))
    if sr is not None and 0.2 < sr < 0.8:
        add("context", "role_unstable", 0, 0.6, round(sr, 2),
            "rotation role is not fully stable")
    pace = _num(row.get("opp_pace"))
    if pace is not None:
        if pace >= PACE_AVG + 2:
            add("context", "pace", +1 if over else -1, 0.6, round(pace, 1),
                "opponent plays at a fast pace (extra possessions)")
        elif pace <= PACE_AVG - 2:
            add("context", "pace", -1 if over else +1, 0.6, round(pace, 1),
                "opponent plays slow (fewer possessions)")
    dvp = _num(row.get("opp_def_vs_pos"))
    if dvp is not None:
        stat = row.get("market_type", "this stat")
        # higher allowance to the position = weaker defense for that stat
        ref = {"points": 14, "rebounds": 5, "assists": 3.5}.get(stat, dvp)
        if dvp >= ref * 1.1:
            add("context", "matchup", +1 if over else -1, 0.8, round(dvp, 1),
                f"opponent is weak defending {stat} for this position ({dvp:.1f} allowed)")
        elif dvp <= ref * 0.9:
            add("context", "matchup", -1 if over else +1, 0.8, round(dvp, 1),
                f"opponent defends {stat} well for this position ({dvp:.1f} allowed)")

    # ---- C. reliability ----
    if row.get("market_type") in HIGH_VAR_MARKETS:
        add("reliability", "volatility", -1, 0.8, "high",
            "this is a high-variance market, so confidence is capped")
    mean, sd = _num(row.get("proj_mean")), _num(row.get("proj_sd"))
    if mean and sd and mean > 0 and sd / mean > 0.55:
        add("reliability", "wide_range", -1, 0.6, round(sd, 1),
            "the projected outcome range is wide")
    miss = sum(1 for c in ("last_3_min", "opp_pace", "opp_def_vs_pos", "is_starter")
               if _num(row.get(c)) is None and row.get(c) is None)
    if miss >= 2:
        add("reliability", "feature_gaps", -1, 0.5, miss,
            "limited context data for this player tonight")

    # ---- D. entry fit ----
    if row.get("pair_flag"):
        add("entry", "pair", 0, 0.5, row.get("pair_type"),
            f"part of a correlated pair ({row.get('pair_type')})")
    return ev


def evidence_score(ev: dict) -> dict:
    """Net signed weight per bucket + overall (for the decision engine)."""
    out = {}
    for b, items in ev.items():
        out[b] = round(sum(i["polarity"] * i["weight"] for i in items), 2)
    out["net"] = round(sum(out.values()), 2)
    out["reliability_penalty"] = round(-sum(min(0, i["polarity"]) * i["weight"]
                                            for i in ev["reliability"]), 2)
    return out
