"""Player prop models: project a stat's mean + SD, then P(over/under line).

Targets: points, rebounds, assists. Features are leakage-free per-player rolling
form + minutes + rest + a generic opponent-difficulty proxy. The model predicts
the mean; the residual SD turns it into a Normal from which prop probabilities
are read. Real prop odds are consumed from `lines.csv` if present; none are
fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from src import database
from src.config import MAX_DAYS_REST, MIN_HISTORY_GAMES
from src.features import add_team_proxies, load_player_games, load_team_games
from src.predict import _recent_roster, upcoming_games
from src.utils import get_logger

log = get_logger(__name__)

STAT_TO_MARKET = {"points": "player_points", "rebounds": "player_rebounds",
                  "assists": "player_assists"}
# map our SQLite columns
STAT_COL = {"points": "points", "rebounds": "rebounds", "assists": "assists"}


def _rolling(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date"]).copy()
    df["season"] = df["game_date"].str[:4].astype(int)
    g = df.groupby("player_id", group_keys=False)
    for w in (3, 5):
        df[f"last_{w}_{stat}"] = g[stat].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"last_{w}_min"] = g["minutes"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    sg = df.groupby(["player_id", "season"], group_keys=False)
    df[f"season_avg_{stat}"] = sg[stat].apply(lambda s: s.shift(1).expanding().mean())
    df["season_avg_min"] = sg["minutes"].apply(lambda s: s.shift(1).expanding().mean())
    df["home_flag"] = df["is_home"].astype(int)
    df["history"] = g.cumcount()
    prev = g["game_date"].shift(1)
    df["days_rest"] = (pd.to_datetime(df["game_date"]) - pd.to_datetime(prev)).dt.days.clip(upper=MAX_DAYS_REST)
    return df


def _feature_cols(stat: str) -> list[str]:
    return [f"last_3_{stat}", f"last_5_{stat}", "last_3_min", "last_5_min",
            f"season_avg_{stat}", "season_avg_min", "home_flag", "days_rest",
            "opp_allowed_proxy"]


def train_prop_model(engine, stat: str):
    players = load_player_games(engine)
    teams = add_team_proxies(load_team_games(engine))
    if players.empty:
        return None
    df = _rolling(players, STAT_COL[stat])
    proxy = teams[["game_id", "team_id", "team_prior_allowed"]].rename(
        columns={"team_id": "opponent_id", "team_prior_allowed": "opp_allowed_proxy"})
    df = df.merge(proxy, on=["game_id", "opponent_id"], how="left")
    cols = _feature_cols(STAT_COL[stat])
    train = df[df["history"] >= MIN_HISTORY_GAMES].dropna(subset=[STAT_COL[stat]])
    train = train.sort_values("game_date")
    if len(train) < 100:
        return None
    medians = train[cols].median()
    X = train[cols].fillna(medians)
    y = train[STAT_COL[stat]].astype(float)
    n_te = max(50, int(len(train) * 0.2))
    Xtr, ytr, Xte, yte = X.iloc[:-n_te], y.iloc[:-n_te], X.iloc[-n_te:], y.iloc[-n_te:]
    model = RandomForestRegressor(n_estimators=300, max_depth=10,
                                  min_samples_leaf=5, n_jobs=-1, random_state=42)
    model.fit(Xtr, ytr)
    resid_sd = float(np.std(yte - model.predict(Xte))) or float(y.std())
    model.fit(X, y)  # refit on all
    mae = float(np.mean(np.abs(yte - model.predict(Xte))))
    return {"stat": stat, "model": model, "cols": cols, "medians": medians,
            "resid_sd": resid_sd, "mae": round(mae, 2), "n": len(train),
            "players": players, "teams": teams}


def _asof_features(prior: pd.DataFrame, stat: str, *, is_home: int,
                   gdate: str, teams: pd.DataFrame, opponent_id: str) -> dict:
    prior = prior.sort_values("game_date")
    last_date = prior["game_date"].iloc[-1]
    days_rest = min((pd.to_datetime(gdate) - pd.to_datetime(last_date)).days, MAX_DAYS_REST)
    season = int(gdate[:4])
    sp = prior[prior["game_date"].str[:4].astype(int) == season]
    opp = teams[(teams["team_id"] == str(opponent_id)) & (teams["game_date"] < gdate)]
    opp_proxy = opp["team_prior_allowed"].tail(1).mean() if not opp.empty else np.nan
    return {
        f"last_3_{stat}": prior[stat].tail(3).mean(), f"last_5_{stat}": prior[stat].tail(5).mean(),
        "last_3_min": prior["minutes"].tail(3).mean(), "last_5_min": prior["minutes"].tail(5).mean(),
        f"season_avg_{stat}": (sp[stat].mean() if not sp.empty else prior[stat].mean()),
        "season_avg_min": (sp["minutes"].mean() if not sp.empty else prior["minutes"].mean()),
        "home_flag": int(is_home), "days_rest": float(days_rest),
        "opp_allowed_proxy": opp_proxy, "history": len(prior),
    }


def project_props(engine, bundles: dict, days_ahead=5) -> pd.DataFrame:
    games = upcoming_games(days_ahead)
    if games.empty or not bundles:
        return pd.DataFrame()
    any_b = next(iter(bundles.values()))
    players, teams = any_b["players"], any_b["teams"]
    rows = []
    for _, g in games.iterrows():
        for side, team_id, opp_id, opp_abbr in (
            ("home", g["home_team_id"], g["away_team_id"], g["away_team"]),
            ("away", g["away_team_id"], g["home_team_id"], g["home_team"])):
            roster = _recent_roster(players, team_id)
            for _, p in roster.iterrows():
                prior = players[(players["player_id"] == p["player_id"]) &
                                (players["game_date"] < g["game_date"])]
                if len(prior) < MIN_HISTORY_GAMES:
                    continue
                row = {"game_id": g["game_id"], "date": g["game_date"],
                       "player_id": p["player_id"], "player_name": p["player_name"],
                       "team": p["team"], "opponent": opp_abbr}
                for stat, b in bundles.items():
                    feat = _asof_features(prior, STAT_COL[stat], is_home=(side == "home"),
                                          gdate=g["game_date"], teams=teams, opponent_id=opp_id)
                    X = pd.DataFrame([feat])[b["cols"]].fillna(b["medians"])
                    row[f"proj_{stat}"] = round(float(b["model"].predict(X)[0]), 1)
                    row[f"sd_{stat}"] = round(b["resid_sd"], 1)
                rows.append(row)
    return pd.DataFrame(rows)


def prop_prob_over(proj_mean: float, sd: float, line: float) -> float:
    return float(1 - norm.cdf(line, loc=proj_mean, scale=max(sd, 1e-6)))


def prop_edges_from_lines(projections: pd.DataFrame, lines_path: str,
                          edge_threshold=0.03) -> pd.DataFrame:
    """If a real lines.csv exists, compute prop edges. Never fabricates odds."""
    import os
    if not os.path.exists(lines_path) or projections.empty:
        return pd.DataFrame()
    lines = pd.read_csv(lines_path)
    market_to_stat = {v: k for k, v in STAT_TO_MARKET.items()}
    out = []
    for _, ln in lines.iterrows():
        stat = market_to_stat.get(str(ln.get("market_type")))
        if stat is None:
            continue
        match = projections[(projections["player_id"].astype(str) == str(ln.get("player_id"))) &
                            (projections["game_id"].astype(str) == str(ln.get("game_id")))]
        if match.empty:
            continue
        m = match.iloc[0]
        mean, sd = m[f"proj_{stat}"], m[f"sd_{stat}"]
        line = float(ln["line_value"])
        odds = float(ln["odds"])
        dec = (1 + odds / 100 if odds > 0 else 1 + 100 / -odds) if str(
            ln.get("odds_format", "american")).startswith("a") else odds
        implied = 1 / dec
        side = str(ln.get("side", "over")).lower()
        p_over = prop_prob_over(mean, sd, line)
        p_model = p_over if side == "over" else 1 - p_over
        edge = p_model - implied
        if edge >= edge_threshold:
            out.append({"player": m["player_name"], "market": stat, "side": side,
                        "line": line, "book": ln.get("sportsbook_name"), "odds": odds,
                        "proj": mean, "p_model": round(p_model, 3),
                        "implied": round(implied, 3), "edge": round(edge, 3)})
    return pd.DataFrame(out).sort_values("edge", ascending=False) if out else pd.DataFrame()
