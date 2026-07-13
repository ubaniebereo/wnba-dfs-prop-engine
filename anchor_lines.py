"""Sharp anchor lines & consensus fair value (Stage 5, Section 2).

Pinnacle is the sharpest book; on The Odds API it appears in the `eu` region for
WNBA props. We de-vig Pinnacle to a fair probability per prop and treat that as
fair value. Other books' offered prices are then compared to this anchor — the
real edge is a STALE book line vs the sharp fair, not our model vs the book.
If Pinnacle is missing for a prop, fall back to a de-vigged consensus of sharp books.
"""

from __future__ import annotations

import os

import pandas as pd
import requests

from propmodel.devig import devig, implied
from src.utils import get_logger

log = get_logger(__name__)
BASE = "https://api.the-odds-api.com/v4/sports/basketball_wnba"
PROP_MARKETS = "player_points,player_rebounds,player_assists"
SHARP_CONSENSUS = ["pinnacle", "betonlineag", "draftkings", "fanduel"]
MARKET2STAT = {"player_points": "points", "player_rebounds": "rebounds",
               "player_assists": "assists"}


def _key() -> str:
    k = os.environ.get("ODDS_API_KEY", "")
    if not k:
        raise RuntimeError("ODDS_API_KEY not set")
    return k


def fetch_props_multi_region(regions=("us", "eu")) -> pd.DataFrame:
    """All books (US + EU incl Pinnacle) for current WNBA props."""
    key = _key()
    events = requests.get(f"{BASE}/events/", params={"apiKey": key}, timeout=25).json()
    rows = []
    for e in events:
        for reg in regions:
            try:
                d = requests.get(f"{BASE}/events/{e['id']}/odds/",
                                 params={"apiKey": key, "regions": reg,
                                         "markets": PROP_MARKETS,
                                         "oddsFormat": "american"}, timeout=30).json()
            except requests.RequestException:
                continue
            for bk in d.get("bookmakers", []):
                for m in bk.get("markets", []):
                    stat = MARKET2STAT.get(m["key"])
                    if not stat:
                        continue
                    for o in m.get("outcomes", []):
                        rows.append({"event_id": e["id"], "date": e["commence_time"][:10],
                                     "home": e["home_team"], "away": e["away_team"],
                                     "book": bk["key"], "market": stat,
                                     "player": o.get("description"),
                                     "side": o["name"].lower(), "line": o.get("point"),
                                     "odds": float(o["price"])})
    return pd.DataFrame(rows).drop_duplicates(
        ["event_id", "book", "market", "player", "side", "line"]) if rows else pd.DataFrame()


def build_anchor(props: pd.DataFrame) -> pd.DataFrame:
    """Per (player, market, line): anchor fair P(over)/P(under) + source."""
    out = []
    for (player, market, line), grp in props.groupby(["player", "market", "line"]):
        fair_over = source = None
        # prefer Pinnacle's de-vigged two-way
        pin = grp[grp.book == "pinnacle"]
        po, pu = pin[pin.side == "over"], pin[pin.side == "under"]
        if not po.empty and not pu.empty:
            fair_over, _ = devig(float(po.iloc[0].odds), float(pu.iloc[0].odds), "shin")
            source = "pinnacle"
        else:
            # consensus: mean de-vigged fair across sharp books that have both sides
            fos = []
            for bk in SHARP_CONSENSUS:
                b = grp[grp.book == bk]
                bo, bu = b[b.side == "over"], b[b.side == "under"]
                if not bo.empty and not bu.empty:
                    fos.append(devig(float(bo.iloc[0].odds), float(bu.iloc[0].odds), "shin")[0])
            if fos:
                fair_over = sum(fos) / len(fos)
                source = f"consensus({len(fos)})"
        if fair_over is not None:
            out.append({"player": player, "market": market, "line": float(line),
                        "anchor_fair_over": round(fair_over, 4),
                        "anchor_fair_under": round(1 - fair_over, 4),
                        "anchor_source": source})
    return pd.DataFrame(out)


def line_shop_edges(props: pd.DataFrame, anchor: pd.DataFrame,
                    min_edge=0.02) -> pd.DataFrame:
    """Stale-line finder: book price that beats the sharp anchor fair value."""
    if props.empty or anchor.empty:
        return pd.DataFrame()
    a = anchor.set_index(["player", "market", "line"])
    rows = []
    for r in props.itertuples():
        if r.book == "pinnacle":
            continue                              # don't bet the anchor against itself
        key = (r.player, r.market, float(r.line))
        if key not in a.index:
            continue
        fair_side = a.loc[key, "anchor_fair_over"] if r.side == "over" else a.loc[key, "anchor_fair_under"]
        edge = float(fair_side) - implied(r.odds)   # +EV if sharp fair > offered implied
        if edge >= min_edge:
            rows.append({"player": r.player, "market": r.market, "side": r.side,
                         "line": float(r.line), "book": r.book, "odds": int(r.odds),
                         "anchor_fair": round(float(fair_side), 3),
                         "book_implied": round(implied(r.odds), 3),
                         "anchor_edge": round(edge, 3),
                         "anchor_source": a.loc[key, "anchor_source"]})
    out = pd.DataFrame(rows)
    return out.sort_values("anchor_edge", ascending=False).reset_index(drop=True) if not out.empty else out
