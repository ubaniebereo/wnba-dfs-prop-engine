"""Query helpers + alert-ready views (Section 14).

alert_candidates: the standout props worth pushing, with a dedup key so a future
notifier (Claude Cowork / webhook) won't spam the same (player, market, side,
book, line) repeatedly.
"""

from __future__ import annotations

import pandas as pd

from feeds.storage import ENGINE as FEEDS_DB


def alert_candidates(min_standout=30, min_conf=40) -> pd.DataFrame:
    try:
        df = pd.read_sql("SELECT * FROM current_prop_scan", FEEDS_DB)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    out = df[(df["standout_score"] >= min_standout) &
             (df["confidence_score"] >= min_conf)].copy()
    out["dedup_key"] = (out["player_name"].astype(str) + "|" + out["market_type"] + "|" +
                        out["side"] + "|" + out["best_book"].astype(str) + "|" +
                        out["line_value"].astype(str))
    cols = ["dedup_key", "player_name", "team", "market_type", "side", "line_value",
            "best_book", "best_odds_american", "standout_score", "confidence_score",
            "model_edge", "anchor_edge", "news_flag", "pair_flag", "line_move_flag",
            "reason_tags_json", "scanned_at"]
    return out[[c for c in cols if c in out.columns]].sort_values(
        "standout_score", ascending=False).reset_index(drop=True)


def payload_for_alert(row: dict) -> dict:
    """Notification-ready payload (Claude Cowork / webhook)."""
    return {"dedup_key": row.get("dedup_key"), "player": row.get("player_name"),
            "market": row.get("market_type"), "side": row.get("side"),
            "line": row.get("line_value"), "book": row.get("best_book"),
            "odds": row.get("best_odds_american"), "standout": row.get("standout_score"),
            "confidence": row.get("confidence_score"),
            "anchor_edge": row.get("anchor_edge"),
            "reasons": row.get("reason_tags_json"), "ts": row.get("scanned_at")}
