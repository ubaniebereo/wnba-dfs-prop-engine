"""Module 7 — live edge engine on the NEW model (calibrated + market-blended).

Projects upcoming-game players with the minutes x rate model, applies isotonic
calibration and market-prior blending, removes OUT players, and compares to real
Odds API prop lines. Reports an EDGE PROFILE (count, share <=5%, over/under
balance) so we can see edges shrink into a believable range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import advanced, calibration, models
from . import featureset as F
from .featureset import STATS
from src.predict import _recent_roster, upcoming_games
from src.utils import get_logger

log = get_logger(__name__)
SA = {"points": "pts", "rebounds": "reb", "assists": "ast"}


def _team_latest_context(hist: pd.DataFrame, pos: pd.Series) -> dict:
    tb = advanced.team_box(hist).sort_values(["team_id", "game_date"])
    g = tb.groupby("team_id")
    ctx = pd.DataFrame({"def_rtg": g["def_rtg"].mean(), "pace": g["pace"].mean()})
    dvp = advanced.defense_vs_position(hist, pos).sort_values(["team_id", "game_date"])
    dvp_last = dvp.groupby("team_id").mean(numeric_only=True)
    return {"ctx": ctx, "dvp": dvp_last}


def project_upcoming(engine, mm, rms, days_ahead=3, out_ids: set | None = None):
    out_ids = out_ids or set()
    games = upcoming_games(days_ahead)
    if games.empty:
        return pd.DataFrame()
    hist = pd.read_sql("SELECT * FROM player_game_stats WHERE minutes > 0", engine)
    pos = advanced.position_group(hist)
    tstate = _team_latest_context(hist, pos)

    # schedule comes from ESPN (ESPN team ids) but history is BDL (BDL ids);
    # resolve via the shared team ABBREVIATION instead of mismatched ids.
    abbr2id = hist.drop_duplicates("team").set_index("team")["team_id"].to_dict()
    fut = []
    for _, gm in games.iterrows():
        for side, t_abbr, opp_abbr in (
            ("home", gm["home_team"], gm["away_team"]),
            ("away", gm["away_team"], gm["home_team"])):
            tid, opp_id, opp = abbr2id.get(t_abbr), abbr2id.get(opp_abbr), opp_abbr
            if tid is None:
                continue
            for _, p in _recent_roster(hist, tid).iterrows():
                if str(p["player_id"]) in out_ids:
                    continue
                last = hist[hist["player_id"] == p["player_id"]].iloc[-1]
                fut.append({**{c: np.nan for c in hist.columns},
                            "stat_id": f"FUT_{gm['game_id']}_{p['player_id']}",
                            "game_id": f"FUT_{gm['game_id']}", "game_date": gm["game_date"],
                            "player_id": p["player_id"], "player_name": p["player_name"],
                            "team_id": tid, "team": p["team"], "opponent_id": opp_id,
                            "opponent": opp, "is_home": int(side == "home"),
                            "starter": last["starter"]})
    futdf = pd.DataFrame(fut)
    combined = pd.concat([hist, futdf], ignore_index=True)
    feat = F.build(df=combined)
    fr = feat[feat["game_id"].astype(str).str.startswith("FUT")].copy()

    # fill opponent context (no future team box exists) from latest team state
    fr["position_group"] = fr["player_id"].map(pos).fillna("wing")
    fr["opp_def_rtg"] = fr["opponent_id"].map(tstate["ctx"]["def_rtg"])
    fr["opp_pace"] = fr["opponent_id"].map(tstate["ctx"]["pace"])
    for sa in ("pts", "reb", "ast"):
        fr[f"opp_def_{sa}_vs_pos"] = np.nan
        for grp in ("big", "wing", "guard"):
            col = f"def_{sa}_vs_{grp}"
            if col in tstate["dvp"].columns:
                mask = fr["position_group"] == grp
                fr.loc[mask, f"opp_def_{sa}_vs_pos"] = fr.loc[mask, "opponent_id"].map(
                    tstate["dvp"][col])
    proj = models.project(fr, mm, rms)
    # carry basketball-context features through so the reasoning/explanation layer
    # can cite REAL values (minutes trend, pace, opponent defense, role).
    ctx_cols = ["player_id", "last_3_min", "last_5_min", "season_avg_min",
                "is_starter", "starter_rate", "days_rest", "opp_pace", "opp_def_rtg",
                "opp_def_pts_vs_pos", "opp_def_reb_vs_pos", "opp_def_ast_vs_pos"]
    ctx = fr[[c for c in ctx_cols if c in fr.columns]].drop_duplicates("player_id")
    return proj.merge(ctx, on="player_id", how="left")


def edges(projections: pd.DataFrame, props: pd.DataFrame, params: dict,
          iso, w_blend=0.5, edge_threshold=0.03, max_plausible=0.07,
          date_tol_days=2, devig_method="shin") -> pd.DataFrame:
    """Stage 4: NB(MLE)/LTV-Normal probabilities + Shin/logit devig.

    params = {'nb_r': {stat: r}, 'var_stat': {stat: var}} for distributions.prob_over.
    """
    from edge.odds_api import norm_name
    from edge.odds_espn import american_to_decimal
    from . import devig as devig_mod
    from . import distributions
    if projections.empty or props.empty:
        return pd.DataFrame()
    proj = projections.copy()
    proj["key"] = proj["player_name"].map(norm_name)
    props = props.copy()
    props["key"] = props["player"].map(norm_name)

    # center projections on the market: remove the systematic per-market offset
    # (the sharp line is the better mean estimate; this kills the under-skew).
    bias = {}
    for market in STATS:
        col = f"E_{market}"
        if col not in proj.columns:
            continue
        cons = props[props["market"] == market].groupby("key")["line"].median()
        j = proj[["key", col]].merge(cons.rename("line"), on="key", how="inner")
        bias[market] = float((j[col] - j["line"]).mean()) if len(j) else 0.0

    out = []
    for (key, market, line, book, pdate), grp in props.groupby(
            ["key", "market", "line", "book", "date"]):
        if f"E_{market}" not in proj.columns:
            continue
        cand = proj[proj["key"] == key]
        if cand.empty:
            continue
        cand = cand.assign(dd=(pd.to_datetime(cand["game_date"]) - pd.to_datetime(pdate)).abs())
        m = cand.sort_values("dd").iloc[0]
        if m["dd"].days > date_tol_days:
            continue
        e_centered = float(m[f"E_{market}"]) - bias.get(market, 0.0)
        mu = calibration.market_blend(e_centered, float(line), w_blend)
        over = grp[grp["side"] == "over"]
        under = grp[grp["side"] == "under"]
        if over.empty or under.empty:
            continue
        oo, uo = float(over.iloc[0]["odds"]), float(under.iloc[0]["odds"])
        fair_o, fair_u = devig_mod.devig(oo, uo, method=devig_method)   # Shin/logit
        p_over_raw = distributions.prob_over(market, mu, float(line), params)  # NB/LTV
        p_over = calibration.calibrate(iso, p_over_raw)
        for side, price, fair, p_model in (("over", oo, fair_o, p_over),
                                           ("under", uo, fair_u, 1 - p_over)):
            edge = p_model - fair
            if edge < edge_threshold:
                continue
            dec = american_to_decimal(price)
            out.append({"date": pdate, "player": m["player_name"], "market": market,
                        "side": side, "line": float(line), "book": book,
                        "odds": int(price), "E_blend": round(mu, 1),
                        "p_model": round(p_model, 3), "p_book_fair": round(fair, 3),
                        "edge": round(edge, 3),
                        "EV_per_$1": round(p_model * (dec - 1) - (1 - p_model), 3),
                        "plausibility": "plausible" if edge <= max_plausible
                        else "⚠ likely model error"})
    out = pd.DataFrame(out)
    if out.empty:
        return out
    out = (out.sort_values("EV_per_$1", ascending=False)
           .drop_duplicates(["player", "market", "side", "line"], keep="first"))
    return out.sort_values("edge", ascending=False).reset_index(drop=True)


def edge_profile(edges_df: pd.DataFrame) -> dict:
    if edges_df.empty:
        return {"n_edges": 0}
    return {"n_edges": len(edges_df),
            "median_edge": round(float(edges_df["edge"].median()), 3),
            "max_edge": round(float(edges_df["edge"].max()), 3),
            "share_<=5%": round(float((edges_df["edge"] <= 0.05).mean()), 2),
            "share_plausible": round(float((edges_df["plausibility"] == "plausible").mean()), 2),
            "over_share": round(float((edges_df["side"] == "over").mean()), 2)}
