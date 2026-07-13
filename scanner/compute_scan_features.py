"""Compute per-prop scan features + the standout/confidence scores.

The standout score is deliberately NOT model-edge-dominated: our CLV work showed
big model edges are mostly noise. The weighted composite leans on anchor-relative
staleness, news, fresh line movement, and pairs — the signals that actually
predict beating the close. Weights are transparent with a TODO hook to replace
them with a learned beat-close meta-model once a labeled CLV sample accrues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edge.odds_espn import american_to_decimal
from feeds.normalizers import norm_name
from propmodel import distributions
from propmodel.devig import devig, implied

# --- standout component weights (sum ~1.0). TODO: replace with learned weights ---
W = {"anchor": 0.35, "news": 0.20, "move": 0.15, "pair": 0.10, "stale": 0.10, "model": 0.10}
ANCHOR_FULL = 0.05      # anchor edge that maps to a full component score
MODEL_FULL = 0.06


def _best_side(grp: pd.DataFrame, side: str):
    s = grp[grp["side"] == side]
    if s.empty:
        return None
    r = s.loc[s["odds"].idxmax()]            # best (highest) price = line shop
    return r


def _pget(pj, key, col):
    if key in pj.index and col in pj.columns:
        v = pj.loc[key, col]
        return None if pd.isna(v) else v
    return None


def _model_prob_over(market, e_stat, line, params):
    if e_stat is None or pd.isna(e_stat):
        return None
    return distributions.prob_over(market, float(e_stat), float(line), params)


def compute(props: pd.DataFrame, anchor: pd.DataFrame, proj: pd.DataFrame,
            params: dict, news_map: dict, pair_players: dict,
            open_lines: dict, scanned_at: str) -> pd.DataFrame:
    if props.empty:
        return pd.DataFrame()
    a = anchor.set_index(["player", "market", "line"]) if not anchor.empty else None
    pj = proj.copy()
    pj["key"] = pj["player_name"].map(norm_name)
    pj = pj.drop_duplicates("key").set_index("key")

    rows = []
    for (player, market, line), grp in props.groupby(["player", "market", "line"]):
        over, under = _best_side(grp, "over"), _best_side(grp, "under")
        if over is None or under is None:
            continue
        fair_o, fair_u = devig(float(over["odds"]), float(under["odds"]), "shin")
        key = norm_name(player)
        e_stat = _pget(pj, key, f"E_{market}")
        m_over = _model_prob_over(market, e_stat, line, params)
        from propmodel import simulate
        sd_stat = simulate._sd_for(market, float(e_stat), params) if e_stat is not None else None

        akey = (player, market, float(line))
        anc_o = float(a.loc[akey, "anchor_fair_over"]) if (a is not None and akey in a.index) else None
        anc_src = a.loc[akey, "anchor_source"] if (a is not None and akey in a.index) else None

        news = news_map.get(key, {})
        pflag = pair_players.get(key)

        # evaluate both sides, keep the one worth attention (max standout)
        best = None
        for side, row, fair_book, m_p, anc_p in (
            ("over", over, fair_o, m_over, anc_o),
            ("under", under, fair_u, (1 - m_over) if m_over is not None else None,
             (1 - anc_o) if anc_o is not None else None)):
            offered = float(row["odds"])
            model_edge = (m_p - fair_book) if m_p is not None else 0.0
            anchor_edge = (anc_p - implied(offered)) if anc_p is not None else 0.0
            mvkey = (player, market, side, line)
            o_line = open_lines.get(mvkey)
            line_delta = (offered - o_line) if o_line is not None else 0.0
            move_fresh = abs(line_delta) >= 8 and o_line is not None

            comp = {
                "anchor": float(np.clip(anchor_edge / ANCHOR_FULL, 0, 1)),
                "news": float(news.get("severity", 0.0)),
                "move": 1.0 if move_fresh else float(np.clip(abs(line_delta) / 25, 0, 0.6)),
                "pair": 0.7 if pflag else 0.0,
                "stale": 1.0 if anchor_edge >= 0.02 else 0.0,
                "model": float(np.clip(model_edge / MODEL_FULL, 0, 1)),
            }
            standout = round(100 * sum(W[k] * comp[k] for k in W), 1)
            conf = _confidence(news, anc_src, market, e_stat)
            tags = _tags(comp, anchor_edge, news, pflag, move_fresh)
            cand = {
                "scanned_at": scanned_at, "event_id": over["event_id"],
                "game_time": over.get("date"), "player_name": player,
                "team": _pget(pj, key, "team"),
                "opponent": _pget(pj, key, "opponent_id"),
                "position": _pget(pj, key, "position_group"),
                "market_type": market, "side": side, "line_value": float(line),
                "best_book": row["book"], "best_odds_american": int(offered),
                "best_odds_decimal": round(american_to_decimal(offered), 3),
                "model_fair_prob": round(m_p, 3) if m_p is not None else None,
                "anchor_fair_prob": round(anc_p, 3) if anc_p is not None else None,
                "proj_mean": round(float(e_stat), 2) if e_stat is not None else None,
                "proj_sd": round(float(sd_stat), 2) if sd_stat is not None else None,
                # basketball-context features (for grounded reasoning/explanations)
                "last_3_min": _pget(pj, key, "last_3_min"),
                "season_avg_min": _pget(pj, key, "season_avg_min"),
                "is_starter": _pget(pj, key, "is_starter"),
                "starter_rate": _pget(pj, key, "starter_rate"),
                "opp_pace": _pget(pj, key, "opp_pace"),
                "opp_def_rtg": _pget(pj, key, "opp_def_rtg"),
                "opp_def_vs_pos": _pget(pj, key, "opp_def_{}_vs_pos".format(
                    {"points": "pts", "rebounds": "reb", "assists": "ast"}.get(market, "pts"))),
                "model_edge": round(model_edge, 3), "anchor_edge": round(anchor_edge, 3),
                "standout_score": standout, "confidence_score": conf,
                "clv_history_score": 0.0,   # TODO: fill from forward CLV sample
                "news_flag": int(bool(news)), "news_severity": news.get("status"),
                "pair_flag": int(bool(pflag)), "pair_type": pflag if pflag else None,
                "pair_edge_estimate": None,
                "line_move_flag": int(move_fresh), "line_delta": round(line_delta, 1),
                "open_line": o_line, "current_line": offered,
                "devig_method": "shin",
                "stale_vs_anchor": int(anchor_edge >= 0.02),
                "anchor_source": anc_src, "reason_tags_json": tags,
            }
            if best is None or cand["standout_score"] > best["standout_score"]:
                best = cand
        rows.append(best)
    df = pd.DataFrame(rows)
    return df.sort_values("standout_score", ascending=False).reset_index(drop=True) if not df.empty else df


def _confidence(news: dict, anchor_src, market, e_stat) -> float:
    """Trust, not edge: lineup certainty + data completeness + market type."""
    base = 0.55                                  # global calibration baseline
    if news.get("status") in ("questionable", "doubtful"):
        base -= 0.25                             # minutes uncertain
    elif news.get("status") == "out":
        base -= 0.4
    if anchor_src and str(anchor_src).startswith("pinnacle"):
        base += 0.15                             # sharpest anchor present
    elif anchor_src:
        base += 0.05
    if market in ("rebounds", "assists"):
        base += 0.05                             # NB well-calibrated for counts
    if e_stat is None:
        base -= 0.1                              # no model projection
    return round(float(np.clip(base, 0, 1)) * 100, 1)


def _tags(comp, anchor_edge, news, pflag, move_fresh) -> list[str]:
    t = []
    if anchor_edge >= 0.02:
        t.append(f"stale vs anchor +{anchor_edge*100:.1f}%")
    if news:
        t.append(f"news: {news.get('status','?')}")
    if move_fresh:
        t.append("fresh line move")
    if pflag:
        t.append(f"pair: {pflag}")
    if comp["model"] > 0.5:
        t.append("model agrees")
    return t or ["no standout signal"]
