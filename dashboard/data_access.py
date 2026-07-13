"""Data access layer (Section 3B) — reads precomputed scan results from SQLite.

The Dash UI never computes models or polls odds; it only reads these helpers,
keeping callbacks fast.
"""

from __future__ import annotations

import pandas as pd

from feeds.storage import ENGINE as FEEDS_DB


def _safe(query: str, default_cols=None) -> pd.DataFrame:
    try:
        return pd.read_sql(query, FEEDS_DB)
    except Exception:
        return pd.DataFrame(columns=default_cols or [])


def get_scan() -> pd.DataFrame:
    df = _safe("SELECT * FROM current_prop_scan ORDER BY standout_score DESC")
    return df


def get_pairs() -> pd.DataFrame:
    return _safe("SELECT * FROM current_pair_scan")


def get_prizepicks() -> pd.DataFrame:
    return _safe("SELECT * FROM prizepicks_scan ORDER BY pp_value_score DESC")


def get_entries() -> pd.DataFrame:
    return _safe("SELECT * FROM current_entries ORDER BY entry_ev DESC")


def get_family_metrics() -> pd.DataFrame:
    return _safe("SELECT * FROM prop_family_metrics")


def get_recommendations() -> pd.DataFrame:
    return _safe("SELECT * FROM improvement_recommendations")


def get_feature_importance() -> pd.DataFrame:
    return _safe("SELECT * FROM feature_importance_snapshots")


def get_source_health() -> pd.DataFrame:
    return _safe("SELECT * FROM source_health")


def last_scan_time() -> str | None:
    df = _safe("SELECT MAX(scanned_at) t FROM current_prop_scan")
    return df["t"].iloc[0] if not df.empty and df["t"].iloc[0] else None


def get_kpis() -> dict:
    df = get_scan()
    if df.empty:
        return {"props": 0}
    return {
        "props": len(df),
        "standouts": int((df["standout_score"] >= 20).sum()),
        "mean_standout": round(float(df["standout_score"].mean()), 1),
        "mean_anchor_edge": round(float(df["anchor_edge"].mean()) * 100, 2),
        "news_flagged": int(df["news_flag"].sum()),
        "pair_flagged": int(df["pair_flag"].sum()),
        "stale_vs_anchor": int(df["stale_vs_anchor"].sum()),
        "last_scan": last_scan_time(),
    }


def filter_scan(df: pd.DataFrame, team=None, market=None, min_standout=0,
                min_conf=0, news_only=False, stale_only=False, pair_only=False,
                movers_only=False) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if team and team != "ALL":
        out = out[out["team"] == team]
    if market and market != "ALL":
        # map a readable family label to the singles market; non-singles families
        # (combos/threes) only live on the PrizePicks board, so singles filters empty.
        mt = MARKET_TO_SINGLE.get(market, market.lower())
        out = out[out["market_type"] == mt]
    out = out[out["standout_score"] >= (min_standout or 0)]
    out = out[out["confidence_score"] >= (min_conf or 0)]
    if news_only:
        out = out[out["news_flag"] == 1]
    if stale_only:
        out = out[out["stale_vs_anchor"] == 1]
    if pair_only:
        out = out[out["pair_flag"] == 1]
    if movers_only:
        out = out[out["line_move_flag"] == 1]
    return out


def get_sparkline(player_name: str, market_type: str, side: str) -> pd.DataFrame:
    """Recent odds history for one prop (for the detail-panel sparkline)."""
    q = ("SELECT captured_at, line_value, odds_american FROM odds_snapshots "
         f"WHERE player_name = '{player_name.replace(chr(39), chr(39)*2)}' "
         f"AND market_type LIKE '%{market_type}' AND side = '{side}' "
         "ORDER BY captured_at")
    return _safe(q)


def get_all_books(player_name: str, market_type: str) -> pd.DataFrame:
    q = ("SELECT bookmaker, side, line_value, odds_american, is_anchor_book, captured_at "
         "FROM odds_snapshots WHERE player_name = "
         f"'{player_name.replace(chr(39), chr(39)*2)}' AND market_type LIKE '%{market_type}' "
         "ORDER BY captured_at DESC LIMIT 40")
    df = _safe(q)
    return df.drop_duplicates(["bookmaker", "side"]) if not df.empty else df


def get_recent_news(limit=12) -> pd.DataFrame:
    return _safe("SELECT captured_at, event_type, parsed_status, player_name, raw_text "
                 f"FROM news_events WHERE player_name IS NOT NULL "
                 f"ORDER BY captured_at DESC LIMIT {limit}")


_EXCLUDE_TEAMS = {"AUS", "BRAZIL", "JAPAN", "PUERTORICO", "USA", "EAST", "WEST",
                  "WNBASTARS", "CLA", "COL", "DEL", "PAR", "STE", "WIL", "TBD", ""}


def teams_list() -> list[str]:
    """All real WNBA franchises (from the full roster, not just the current scan)."""
    try:
        from src.database import get_engine
        df = pd.read_sql("SELECT DISTINCT team FROM player_game_stats "
                         "WHERE team IS NOT NULL", get_engine())
        teams = sorted(t for t in df["team"].dropna().unique()
                       if t and t not in _EXCLUDE_TEAMS)
    except Exception:
        teams = []
    return ["ALL"] + teams


# PrizePicks readable label -> singles sportsbook market (for the Soft Lines board)
MARKET_TO_SINGLE = {"Points": "points", "Rebounds": "rebounds", "Assists": "assists"}


def markets_list() -> list[str]:
    """All prop families actually present (singles + PrizePicks combos/others)."""
    labels = []
    pp = _safe("SELECT DISTINCT stat_type FROM prizepicks_scan")
    if not pp.empty:
        labels = [s for s in pp["stat_type"].dropna().unique().tolist() if s]
    for core in ("Points", "Rebounds", "Assists"):
        if core not in labels:
            labels.append(core)
    # stable, readable order: singles first, then combos, then the rest
    order = {"Points": 0, "Rebounds": 1, "Assists": 2, "Pts+Rebs+Asts": 3,
             "Pts+Rebs": 4, "Pts+Asts": 5, "Rebs+Asts": 6, "3-PT Made": 7}
    labels.sort(key=lambda s: (order.get(s, 99), s))
    return ["ALL"] + labels
