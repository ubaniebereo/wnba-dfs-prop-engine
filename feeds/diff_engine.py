"""Diff engine & line-move detection (Stage 6, Sec 7) — the core of latency study.

Compares consecutive odds_snapshots to emit line_move_events, links each move to a
preceding news_event within a window, and measures how long soft books lag the
sharp anchor (Pinnacle) after news.
"""

from __future__ import annotations

import pandas as pd

from .storage import insert, read
from src.utils import get_logger

log = get_logger(__name__)
KEY = ["event_id", "bookmaker", "player_name", "market_type", "side"]


def detect_moves(min_odds_delta=5.0) -> int:
    """Emit a line_move_event whenever a book's line or price changed vs prior snapshot."""
    snaps = read("SELECT * FROM odds_snapshots ORDER BY captured_at")
    if snaps.empty:
        return 0
    snaps = snaps.sort_values("captured_at")
    moves = []
    for _, g in snaps.groupby(KEY):
        g = g.sort_values("captured_at")
        prev = None
        for r in g.itertuples():
            if prev is not None:
                line_chg = float(r.line_value) != float(prev.line_value)
                odds_chg = abs(float(r.odds_american) - float(prev.odds_american)) >= min_odds_delta
                if line_chg or odds_chg:
                    moves.append({
                        "event_id": r.event_id, "bookmaker": r.bookmaker,
                        "player_name": r.player_name, "market_type": r.market_type,
                        "side": r.side, "old_line": float(prev.line_value),
                        "new_line": float(r.line_value), "old_odds": float(prev.odds_american),
                        "new_odds": float(r.odds_american), "moved_at": r.captured_at,
                        "metadata_json": {}})
            prev = r
    n = insert("line_move_events", moves)
    log.info("detected %d line moves", n)
    return n


def link_moves_to_news(window_minutes=20) -> pd.DataFrame:
    """Associate each move with a player news event in the preceding window."""
    moves = read("SELECT * FROM line_move_events")
    news = read("SELECT * FROM news_events WHERE player_name IS NOT NULL")
    if moves.empty or news.empty:
        return moves
    moves["moved_dt"] = pd.to_datetime(moves["moved_at"], utc=True, errors="coerce")
    news["news_dt"] = pd.to_datetime(news["captured_at"], utc=True, errors="coerce")
    linked = []
    for m in moves.itertuples():
        cand = news[(news["player_name"] == m.player_name) &
                    (news["news_dt"] <= m.moved_dt) &
                    (news["news_dt"] >= m.moved_dt - pd.Timedelta(minutes=window_minutes))]
        trig = cand.sort_values("news_dt").iloc[-1] if not cand.empty else None
        d = m._asdict()
        d["trigger_news_event_id"] = trig["news_event_id"] if trig is not None else None
        d["lag_minutes"] = ((m.moved_dt - trig["news_dt"]).total_seconds() / 60
                            if trig is not None else None)
        linked.append(d)
    return pd.DataFrame(linked)


def latency_metrics() -> dict:
    """Lag from anchor (Pinnacle) move to soft-book move on the same player/market."""
    moves = read("SELECT * FROM line_move_events")
    if moves.empty:
        return {"moves": 0}
    moves["moved_dt"] = pd.to_datetime(moves["moved_at"], utc=True, errors="coerce")
    lags = []
    for _, g in moves.groupby(["event_id", "player_name", "market_type", "side"]):
        anchor = g[g["bookmaker"] == "pinnacle"].sort_values("moved_dt")
        soft = g[g["bookmaker"] != "pinnacle"].sort_values("moved_dt")
        if not anchor.empty and not soft.empty:
            lag = (soft["moved_dt"].iloc[0] - anchor["moved_dt"].iloc[0]).total_seconds() / 60
            lags.append(lag)
    if not lags:
        return {"moves": len(moves), "anchor_to_soft_pairs": 0}
    s = pd.Series(lags)
    return {"moves": len(moves), "anchor_to_soft_pairs": len(s),
            "median_lag_min": round(float(s.median()), 1),
            "share_soft_slower": round(float((s > 0).mean()), 2)}
