"""Compare model prop projections to REAL book prop lines -> candidate bets.

For each player/market/line offered (per book), de-vig the over/under pair, turn
our projection (mean + residual SD) into P(over)/P(under) via a Normal, and flag
sides where model probability exceeds the book's fair probability by >= threshold.
Edges above a plausibility cap are tagged as likely model error, not value.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import norm

from .odds_api import norm_name
from .odds_espn import american_to_decimal, american_to_implied, devig_two_way


def _market_bias(proj: pd.DataFrame, props: pd.DataFrame) -> dict:
    """Per-market systematic offset (mean projection - consensus line).

    The sharp book line is a better mean estimate than our model, so we center
    our projections on it. This removes the one-sided 'everything is an under'
    artifact; only genuine disagreement remains as edge.
    """
    bias = {}
    for market in ("points", "rebounds", "assists"):
        col = f"proj_{market}"
        if col not in proj.columns:
            continue
        cons = props[props["market"] == market].groupby("key")["line"].median()
        j = proj[["key", col]].merge(cons.rename("line"), on="key", how="inner")
        bias[market] = float((j[col] - j["line"]).mean()) if len(j) else 0.0
    return bias


def prop_edges(projections: pd.DataFrame, props: pd.DataFrame,
               edge_threshold=0.03, max_plausible=0.07, date_tol_days=2) -> pd.DataFrame:
    # max_plausible=0.07: against sharp prop books, real edges are small (~2-5%).
    # Anything larger is almost certainly model error, not value.
    if projections.empty or props.empty:
        return pd.DataFrame()
    proj = projections.copy()
    proj["key"] = proj["player_name"].map(norm_name)
    props = props.copy()
    props["key"] = props["player"].map(norm_name)
    bias = _market_bias(proj, props)   # market-calibration offsets

    out = []
    grp_cols = ["key", "market", "line", "book", "date"]
    for (key, market, line, book, pdate), grp in props.groupby(grp_cols):
        if f"proj_{market}" not in proj.columns:
            continue
        cand = proj[proj["key"] == key]
        if cand.empty:
            continue
        cand = cand.assign(dd=(pd.to_datetime(cand["date"]) -
                               pd.to_datetime(pdate)).abs())
        m = cand.sort_values("dd").iloc[0]
        if m["dd"].days > date_tol_days:
            continue
        mean, sd = m[f"proj_{market}"], m[f"sd_{market}"]
        if pd.isna(mean) or pd.isna(sd):
            continue
        mean = mean - bias.get(market, 0.0)   # center on the market
        over = grp[grp["side"] == "over"]
        under = grp[grp["side"] == "under"]
        if over.empty or under.empty:
            continue
        oo, uo = float(over.iloc[0]["odds"]), float(under.iloc[0]["odds"])
        fair_over, fair_under = devig_two_way(american_to_implied(oo),
                                              american_to_implied(uo))
        p_over = float(1 - norm.cdf(float(line), loc=mean, scale=max(sd, 1e-6)))

        for side, price, fair, p_model in (("over", oo, fair_over, p_over),
                                           ("under", uo, fair_under, 1 - p_over)):
            edge = p_model - fair
            if edge < edge_threshold:
                continue
            dec = american_to_decimal(price)
            out.append({
                "date": pdate, "player": m["player_name"], "market": market,
                "side": side, "line": float(line), "book": book, "odds": int(price),
                "proj_adj": round(float(mean), 1), "p_model": round(p_model, 3),
                "p_book_fair": round(fair, 3), "edge": round(edge, 3),
                "EV_per_$1": round(p_model * (dec - 1) - (1 - p_model), 3),
                "plausibility": "plausible" if edge <= max_plausible
                else "⚠ likely model error",
            })
    out = pd.DataFrame(out)
    if out.empty:
        return out
    # dedupe to ONE row per unique bet, keeping the best price (line shopping)
    out = (out.sort_values("EV_per_$1", ascending=False)
           .drop_duplicates(subset=["player", "market", "side", "line"], keep="first"))
    return out.sort_values("edge", ascending=False).reset_index(drop=True)
