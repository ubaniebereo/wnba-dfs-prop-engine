"""CLV backtest on REAL historical WNBA props (Stage 4, Section 3).

The Odds API v4 historical endpoints give prop odds at any past timestamp, so we
can measure whether the model's flagged bets beat the CLOSING line — the honest
test of value. For each sampled historical game we fetch a 'placed' snapshot
(hours before tip) and a 'closing' snapshot (~tip), generate model bets at the
placed odds, then compute CLV vs closing and the realized hit rate.

Endpoints (cost ~10 credits/market/snapshot):
  /v4/historical/sports/basketball_wnba/events?date=ISO
  /v4/historical/sports/basketball_wnba/events/{id}/odds?markets=...&date=ISO
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from edge.odds_api import norm_name
from propmodel import devig as devig_mod
from src.utils import get_logger

log = get_logger(__name__)
HBASE = "https://api.the-odds-api.com/v4/historical/sports/basketball_wnba"
MARKETS = "player_points,player_rebounds,player_assists"
MARKET2STAT = {"player_points": "points", "player_rebounds": "rebounds",
               "player_assists": "assists"}


def _key() -> str:
    k = os.environ.get("ODDS_API_KEY", "")
    if not k:
        raise RuntimeError("ODDS_API_KEY not set")
    return k


def _get(path: str, params: dict) -> dict:
    params = dict(params); params["apiKey"] = _key()
    r = requests.get(f"{HBASE}{path}", params=params, timeout=40)
    rem = r.headers.get("x-requests-remaining")
    if rem is not None:
        log.info("historical odds: %s credits remaining", rem)
    r.raise_for_status()
    return r.json()


def historical_events(date_iso: str) -> list[dict]:
    return _get("/events", {"date": date_iso}).get("data", []) or []


def event_props(event_id: str, date_iso: str) -> list[dict]:
    payload = _get(f"/events/{event_id}/odds",
                   {"regions": "us", "markets": MARKETS,
                    "oddsFormat": "american", "date": date_iso})
    data = payload.get("data") or {}
    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            stat = MARKET2STAT.get(m["key"])
            if not stat:
                continue
            for o in m.get("outcomes", []):
                rows.append({"event_id": event_id, "book": bk["key"], "market": stat,
                             "player": o.get("description"), "side": o["name"].lower(),
                             "line": o.get("point"), "odds": float(o["price"])})
    return rows


def _best_two_way(props: pd.DataFrame):
    """Collapse to one over/under pair per (player,market,line): best price each side."""
    if props.empty:
        return props
    idx = props.groupby(["player", "market", "line", "side"])["odds"].idxmax()
    return props.loc[idx]


def collect_snapshots(date: str, placed_hours_before=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (placed_props, closing_props) for all games on a date."""
    midday = f"{date}T18:00:00Z"
    placed_rows, closing_rows = [], []
    for ev in historical_events(midday):
        commence = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        placed_ts = (commence - timedelta(hours=placed_hours_before)).strftime("%Y-%m-%dT%H:%M:%SZ")
        closing_ts = (commence - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for ts, bucket in ((placed_ts, placed_rows), (closing_ts, closing_rows)):
            for r in event_props(ev["id"], ts):
                r["date"] = date
                r["home"] = ev["home_team"]; r["away"] = ev["away_team"]
                bucket.append(r)
    return pd.DataFrame(placed_rows), pd.DataFrame(closing_rows)


def build_bets(placed: pd.DataFrame, closing: pd.DataFrame, projections: pd.DataFrame,
               params, outcomes: pd.DataFrame, edge_threshold=0.03,
               devig_method="shin") -> pd.DataFrame:
    """Model bets at placed odds + CLV vs closing + realized outcome."""
    from propmodel import distributions
    if placed.empty:
        return pd.DataFrame()
    placed = _best_two_way(placed.dropna(subset=["line"]))
    closing = _best_two_way(closing.dropna(subset=["line"])) if not closing.empty else closing
    proj = projections.copy(); proj["key"] = proj["player_name"].map(norm_name)
    out = outcomes.copy(); out["key"] = out["player_name"].map(norm_name)

    bets = []
    for (player, market, line), grp in placed.groupby(["player", "market", "line"]):
        over = grp[grp.side == "over"]; under = grp[grp.side == "under"]
        if over.empty or under.empty:
            continue
        key = norm_name(player)
        pr = proj[proj.key == key]
        if pr.empty:
            continue
        mu = float(pr.iloc[0][f"E_{market}"])
        oo, uo = float(over.iloc[0].odds), float(under.iloc[0].odds)
        fair_o, fair_u = devig_mod.devig(oo, uo, method=devig_method)
        p_over = distributions.prob_over(market, mu, float(line), params)
        for side, price, fair, p_model in (("over", oo, fair_o, p_over),
                                           ("under", uo, fair_u, 1 - p_over)):
            edge = p_model - fair
            if edge < edge_threshold:
                continue
            # closing price for this exact bet
            cl = closing[(closing.player == player) & (closing.market == market) &
                         (closing.line == line) & (closing.side == side)] if not closing.empty else closing
            if cl is None or cl.empty:
                continue
            close_odds = float(cl.iloc[0].odds)
            placed_imp = devig_mod.implied(price)
            close_imp = devig_mod.implied(close_odds)
            # realized outcome from box score (outcomes pre-filtered to this date)
            oc = out[out.key == key]
            actual = float(oc.iloc[0][market]) if not oc.empty else np.nan
            won = (int(actual > line) if side == "over" else int(actual < line)) if actual == actual else np.nan
            bets.append({"player": player, "market": market, "side": side, "line": line,
                         "placed_odds": price, "close_odds": close_odds,
                         "p_model": round(p_model, 3), "edge": round(edge, 3),
                         "placed_implied": round(placed_imp, 3),
                         "close_implied": round(close_imp, 3),
                         "clv_prob": round(close_imp - placed_imp, 4),  # >0 = beat close
                         "beat_close": int(close_imp > placed_imp), "won": won})
    return pd.DataFrame(bets)


def evaluate_clv(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"n": 0}
    b = bets.dropna(subset=["clv_prob"])
    return {"n_bets": len(b),
            "mean_clv_prob": round(float(b["clv_prob"].mean()), 4),
            "median_clv_prob": round(float(b["clv_prob"].median()), 4),
            "share_beat_close": round(float(b["beat_close"].mean()), 2),
            "realized_hit_rate": round(float(b["won"].dropna().mean()), 2)
            if b["won"].notna().any() else None,
            "mean_model_p": round(float(b["p_model"].mean()), 3)}
