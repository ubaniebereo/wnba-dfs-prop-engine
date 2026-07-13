"""Compare PrizePicks lines to the sharp sportsbook market + model.

The DFS edge: PrizePicks lines are often soft/slow vs sportsbooks. For each PP
line we compute (a) the sportsbook CONSENSUS line for that player+market and the
line gap, and (b) the model's P(over the PP line). A PP line well below the sharp
line => 'MORE' value; well above => 'LESS' value. Breakeven on pick'em is ~0.5,
so model probs far from 0.5 also flag value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feeds.normalizers import norm_name
from propmodel import distributions


def compute_pp_scan(pp_df: pd.DataFrame, props: pd.DataFrame, proj: pd.DataFrame,
                    params: dict, scanned_at: str, store: dict | None = None) -> pd.DataFrame:
    if pp_df.empty:
        return pd.DataFrame()
    # sportsbook consensus line per (player, market) from the Odds API props
    book_line = {}
    if not props.empty:
        for (player, market), g in props.groupby(["player", "market"]):
            book_line[(norm_name(player), market)] = float(g["line"].median())
    pj = proj.copy()
    pj["key"] = pj["player_name"].map(norm_name)
    pj = pj.drop_duplicates("key").set_index("key")

    from propmodel import simulate
    from propmodel.family_calibration import calibrated_prob_over
    CANON2COMP = {"player_points": "points", "player_rebounds": "rebounds",
                  "player_assists": "assists", "player_threes_made": "tpm",
                  "player_steals": "steals", "player_blocks": "blocks",
                  "player_turnovers": "turnovers"}
    COMBO_MKTS = set(simulate.COMBO_PARTS) | {"player_fantasy_score"}

    def _E(key, comp):
        col = f"E_{comp}"
        if key in pj.index and col in pj.columns and not pd.isna(pj.loc[key, col]):
            return float(pj.loc[key, col])
        return None

    rows = []
    for r in pp_df.itertuples():
        key = norm_name(r.player)
        market = r.market               # canonical (player_points / player_pra / ...)
        pp_line = float(r.line)
        comp = CANON2COMP.get(market)
        model_p_over, coverage, calib_method = None, "display-only", None
        if comp:                         # core single -> per-family model + calibration
            e = _E(key, comp)
            if e is not None and store is not None:
                model_p_over, calib_method = calibrated_prob_over(store, comp, e, pp_line)
                coverage = "modeled"
            elif e is not None:
                model_p_over = distributions.prob_over(comp, e, pp_line, params)
                coverage = "modeled"
        elif market in COMBO_MKTS:       # combo -> Monte Carlo joint simulation
            means = {c: _E(key, c) for c in ("points", "rebounds", "assists")}
            if all(v is not None for v in means.values()):
                sim = simulate.combo_sim(market, means, pp_line, params)
                if sim:
                    model_p_over = sim["prob_over"]
                    coverage = "simulated"
        # sportsbook line gap only meaningful for core singles
        bl = book_line.get((key, comp)) if comp else None
        line_diff = (bl - pp_line) if bl is not None else None   # +ve => PP line is LOWER

        # demon/goblin lines are INTENTIONALLY off-market (altered payout) -> a big
        # line gap is by design, not value. Only use the line gap for STANDARD lines.
        is_std = (r.odds_type == "standard")
        dg_flag = int(not is_std)
        use_gap = line_diff if (is_std and line_diff is not None) else None

        side = "—"
        if not is_std:
            side = f"variant: {r.odds_type} (altered payout — EV needs its multiplier)"
        elif model_p_over is not None:
            if model_p_over >= 0.55 or (use_gap is not None and use_gap >= 1.0):
                side = "MORE (over)"
            elif model_p_over <= 0.45 or (use_gap is not None and use_gap <= -1.0):
                side = "LESS (under)"
        # value: standard lines get line-gap + model deviation; variants get neither
        gap_comp = min(abs(use_gap) / 3.0, 1.0) if use_gap is not None else 0.0
        prob_comp = (abs(model_p_over - 0.5) * 2) if (model_p_over is not None and is_std) else 0.0
        value = round(100 * (0.6 * gap_comp + 0.4 * prob_comp), 1)

        rows.append({
            "scanned_at": scanned_at, "player_name": r.player, "team": r.team,
            "stat_type": r.stat_type, "market": market, "is_combo": int(r.is_combo),
            "pp_line": pp_line, "odds_type": r.odds_type,
            "book_line": bl, "line_diff": round(line_diff, 1) if line_diff is not None else None,
            "model_prob_over": round(model_p_over, 3) if model_p_over is not None else None,
            "coverage": coverage, "calib_method": calib_method,
            "demon_goblin_flag": dg_flag,
            "recommend": side, "pp_value_score": value,
            "why": _why(r.stat_type, pp_line, bl, line_diff, model_p_over,
                        coverage, calib_method, dg_flag, side),
            "start_time": getattr(r, "start_time", None),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("pp_value_score", ascending=False).reset_index(drop=True)


def _why(stat_type, pp_line, book_line, line_diff, p_over, coverage,
         calib_method, dg_flag, side) -> str:
    """Plain-language summary of why this prop is (or isn't) interesting."""
    if dg_flag:
        return (f"{stat_type} {pp_line}: demon/goblin VARIANT — line is altered by design "
                f"with a non-standard payout, so a gap vs the book is expected, not value.")
    parts = []
    if coverage == "simulated":
        parts.append(f"combo simulated from component models (P(more)={p_over:.0%})"
                     if p_over is not None else "combo simulated")
    elif coverage == "modeled":
        cm = f" then {calib_method}-calibrated" if calib_method and calib_method != "raw" else ""
        if p_over is not None:
            parts.append(f"per-family model{cm} -> P(more)={p_over:.0%}")
    else:
        return f"{stat_type} {pp_line}: no model for this stat yet — line shown for reference only."
    if book_line is not None and line_diff is not None and abs(line_diff) >= 0.5:
        dirw = "BELOW" if line_diff > 0 else "ABOVE"
        parts.append(f"PrizePicks {pp_line} is {abs(line_diff):.1f} {dirw} the sharp book line "
                     f"({book_line}) — favors {'MORE' if line_diff > 0 else 'LESS'}")
    elif book_line is not None:
        parts.append(f"line matches the sharp book ({book_line}) — little line-shop edge")
    if "MORE" in side or "LESS" in side:
        parts.append(f"lean: {side}")
    return "; ".join(parts) if parts else "no standout signal"
