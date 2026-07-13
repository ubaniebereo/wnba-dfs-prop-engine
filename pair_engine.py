"""Correlated pairs / same-game value (Stage 5, Section 4).

Books often price same-game prop combos near INDEPENDENCE. If the true
correlation is non-zero, a combo is mispriced. We estimate teammate/opponent
correlations from the multi-season history, build a Gaussian-copula joint
probability from sharp marginals (anchor fair P(over)), and compare to the
independence product to flag correlation mispricing.

Note: The Odds API does not expose actual SGP prices, so pair flags identify
WHERE correlation is unpriced (joint vs independent), not a confirmed book edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from src.database import get_engine

MIN_SHARED_GAMES = 25
_CORR_CACHE: dict = {}


def _load() -> pd.DataFrame:
    eng = get_engine()
    df = pd.read_sql("SELECT game_id, game_date, team_id, team, opponent, player_id, "
                     "player_name, points, rebounds, assists, minutes FROM "
                     "player_game_stats WHERE minutes > 0", eng)
    return df


def teammate_correlations(stat: str = "points") -> pd.DataFrame:
    """Pairwise within-team correlation of a stat across shared games."""
    df = _load()
    rows = []
    for tid, g in df.groupby("team_id"):
        wide = g.pivot_table(index="game_id", columns="player_name", values=stat)
        # keep players with enough games
        wide = wide.loc[:, wide.notna().sum() >= MIN_SHARED_GAMES]
        if wide.shape[1] < 2:
            continue
        corr = wide.corr(min_periods=MIN_SHARED_GAMES)
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                rho = corr.loc[a, b]
                shared = wide[[a, b]].dropna().shape[0]
                if pd.notna(rho) and shared >= MIN_SHARED_GAMES:
                    rows.append({"team_id": tid, "player_a": a, "player_b": b,
                                 "stat": stat, "rho": round(float(rho), 3),
                                 "shared_games": shared})
    out = pd.DataFrame(rows)
    return out.sort_values("rho") if not out.empty else out


def correlation_summary(stat="points") -> dict:
    c = teammate_correlations(stat)
    if c.empty:
        return {"n_pairs": 0}
    return {"stat": stat, "n_pairs": len(c),
            "mean_rho": round(float(c["rho"].mean()), 3),
            "share_positive": round(float((c["rho"] > 0.05).mean()), 2),
            "share_negative": round(float((c["rho"] < -0.05).mean()), 2),
            "most_negative": c.iloc[0][["player_a", "player_b", "rho"]].to_dict(),
            "most_positive": c.iloc[-1][["player_a", "player_b", "rho"]].to_dict()}


def gaussian_copula_joint(p_over_a: float, p_over_b: float, rho: float) -> float:
    """P(A over AND B over) under a Gaussian copula with correlation rho."""
    pa_u, pb_u = 1 - p_over_a, 1 - p_over_b
    za, zb = norm.ppf(np.clip(pa_u, 1e-4, 1 - 1e-4)), norm.ppf(np.clip(pb_u, 1e-4, 1 - 1e-4))
    both_under = float(multivariate_normal.cdf([za, zb], mean=[0, 0],
                                               cov=[[1, rho], [rho, 1]]))
    # inclusion-exclusion: P(both over) = 1 - P(a under) - P(b under) + P(both under).
    # (the old code dropped p_over_b, so positive rho could wrongly give joint<indep)
    return float(1 - pa_u - pb_u + both_under)


def opponent_correlation() -> float:
    """Pooled cross-team (same-game) game-environment correlation of scoring.

    Players on opposing teams co-move via the game's pace/total, NOT via shared
    usage. Estimated from team points-for vs points-against per game, then shrunk
    to player level (individual scoring adds idiosyncratic noise).
    """
    if "opp" in _CORR_CACHE:
        return _CORR_CACHE["opp"]
    tg = pd.read_sql("SELECT points_for, points_against FROM team_game_stats "
                     "WHERE points_for IS NOT NULL AND points_against IS NOT NULL",
                     get_engine())
    rho_team = float(np.corrcoef(tg["points_for"], tg["points_against"])[0, 1]) \
        if len(tg) > 30 else 0.0
    rho = round(float(np.clip(rho_team * 0.45, -0.4, 0.4)), 3)   # team -> player shrink
    _CORR_CACHE["opp"] = rho
    return rho


def find_pairs(anchor: pd.DataFrame, proj: pd.DataFrame, props: pd.DataFrame,
               opp_corr: float | None = None, min_abs_mispricing=0.03) -> pd.DataFrame:
    """Flag DIFFERENT-TEAM prop pairs whose joint != independence.

    Per the product rule, both players must be on different teams. Real cross-team
    correlation only exists for SAME-GAME opponents (shared pace/total) -> we use
    the pooled opponent correlation there; different-game cross-team pairs are
    ~independent and aren't flagged (no correlation edge to exploit).
    """
    if anchor.empty or proj.empty:
        return pd.DataFrame()
    if opp_corr is None:
        opp_corr = opponent_correlation()
    from edge.odds_api import TEAM_ABBR     # full team name -> abbreviation
    # tonight's matchups in abbreviations: team -> opponent team
    team_opp = {}
    for _, p in props.iterrows():
        h, a = TEAM_ABBR.get(p.get("home")), TEAM_ABBR.get(p.get("away"))
        if h and a:
            team_opp[h] = a
            team_opp[a] = h
    name2team = (proj.dropna(subset=["team"]).drop_duplicates("player_name")
                 .set_index("player_name")["team"].to_dict())

    a = anchor[anchor.market == "points"].copy()
    a["team"] = a["player"].map(name2team)
    a = a.dropna(subset=["team"])
    players = a.to_dict("records")
    rows = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pa, pb = players[i], players[j]
            if pa["team"] == pb["team"]:
                continue                                  # DIFFERENT TEAMS ONLY
            same_game = team_opp.get(pa["team"]) == pb["team"]
            rho = opp_corr if same_game else 0.0
            if abs(rho) < 0.05:
                continue                                  # no cross-team edge -> skip
            joint = gaussian_copula_joint(pa["anchor_fair_over"], pb["anchor_fair_over"], rho)
            indep = pa["anchor_fair_over"] * pb["anchor_fair_over"]
            mis = joint - indep
            if abs(mis) >= min_abs_mispricing:
                rows.append({
                    "team": f"{pa['team']} vs {pb['team']}",
                    "player_a": pa["player"], "player_b": pb["player"],
                    "leg": "both_over_points", "rho": rho, "same_game": int(same_game),
                    "p_independent": round(indep, 3), "p_joint": round(joint, 3),
                    "mispricing": round(mis, 3),
                    "direction": "dual-over underpriced (game environment)" if mis > 0
                    else "dual-over overpriced (split lean)"})
    out = pd.DataFrame(rows)
    return out.reindex(out.mispricing.abs().sort_values(ascending=False).index) \
        if not out.empty else out
