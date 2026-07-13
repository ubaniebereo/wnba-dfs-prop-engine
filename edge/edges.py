"""Compare internal fair probabilities to de-vigged book probabilities -> edges.

edge = p_model - p_book_fair ; EV per $1 = p_model*(decimal-1) - (1-p_model).
Only markets with edge >= threshold are flagged, each tagged with a short reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .game_models import (current_state, p_home_cover, p_over, project_game,
                          train_game_models, upcoming_features)
from .odds_espn import american_to_implied, devig_two_way, fetch_upcoming_odds
from src.utils import get_logger

log = get_logger(__name__)


def _ev(p_model: float, decimal: float) -> float:
    return p_model * (decimal - 1) - (1 - p_model)


def _reason(market, side, feats, proj, line, line_move) -> str:
    g = feats["game"]
    tags = []
    if market == "moneyline" or market == "spread":
        if (side == "home" and g["d_rest"] >= 1.5) or (side == "away" and g["d_rest"] <= -1.5):
            tags.append("rest edge")
        model_fav_home = proj["p_home_win"] > 0.5
        if (side == "home") == model_fav_home and abs(g["elo_diff"]) > 40:
            tags.append("model rating gap")
    if market == "total":
        if side == "over" and proj["mu_total"] > line:
            tags.append("pace/offense lean over")
        if side == "under" and proj["mu_total"] < line:
            tags.append("defense lean under")
    if line_move and abs(line_move) >= 0.02:
        tags.append("line moved " + ("toward" if line_move > 0 else "away"))
    return ", ".join(tags) or "model disagrees with price"


def find_game_edges(days_ahead=5, edge_threshold=0.03,
                    odds_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """odds_df: tidy per-side odds (e.g. Odds API best line). If None, use ESPN."""
    bundle = train_game_models()
    latest, elo = current_state()
    odds = odds_df if odds_df is not None else fetch_upcoming_odds(days_ahead)
    if odds is None or odds.empty:
        return pd.DataFrame(), bundle["metrics"]

    cands = []
    for gid, gdf in odds.groupby("game_id"):
        home = gdf["home_team"].iloc[0]
        away = gdf["away_team"].iloc[0]
        gdate = gdf["date"].iloc[0]
        feats = upcoming_features(home, away, gdate, latest, elo)
        if feats is None:
            continue
        proj = project_game(bundle, feats)

        for market in ("moneyline", "spread", "total"):
            m = gdf[gdf["market_type"] == market]
            if len(m) < 2:
                continue
            if market == "total":
                a_side, b_side = "over", "under"
            else:
                a_side, b_side = "home", "away"
            ra = m[m["side"] == a_side]
            rb = m[m["side"] == b_side]
            if ra.empty or rb.empty:
                continue
            ra, rb = ra.iloc[0], rb.iloc[0]
            fair_a, fair_b = devig_two_way(american_to_implied(ra["odds"]),
                                           american_to_implied(rb["odds"]))

            # model probability for each side
            if market == "moneyline":
                pa, pb = proj["p_home_win"], 1 - proj["p_home_win"]
            elif market == "spread":
                pa = p_home_cover(proj, ra["line_value"])
                pb = 1 - pa
            else:
                pa = p_over(proj, ra["line_value"])
                pb = 1 - pa

            for side_row, p_model, fair in ((ra, pa, fair_a), (rb, pb, fair_b)):
                edge = p_model - fair
                if edge < edge_threshold:
                    continue
                # an honest plausibility gate: edges this large vs a sharp book
                # almost always mean the model is wrong, not the market.
                flag = "plausible" if edge <= 0.10 else "⚠ likely model error"
                cands.append({
                    "game_id": gid, "date": gdate, "matchup": f"{away}@{home}",
                    "market": market, "side": side_row["side"],
                    "line": side_row["line_value"], "book": side_row["book"],
                    "odds": int(side_row["odds"]),
                    "p_model": round(p_model, 3), "p_book_fair": round(fair, 3),
                    "edge": round(edge, 3),
                    "EV_per_$1": round(_ev(p_model, side_row["decimal"]), 3),
                    "plausibility": flag,
                    "reason": _reason(market, side_row["side"], feats, proj,
                                      side_row["line_value"],
                                      side_row.get("line_move", 0.0)),
                })
    out = pd.DataFrame(cands)
    if not out.empty:
        out = out.sort_values("edge", ascending=False).reset_index(drop=True)
    return out, bundle["metrics"]
