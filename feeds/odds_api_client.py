"""Odds snapshot ingestion (Stage 6, Sec 5/A) — reuses the verified Odds API path.

Captures timestamped multi-book prop odds (incl Pinnacle = anchor) into
odds_snapshots so the diff engine can detect line moves over time.
"""

from __future__ import annotations

import anchor_lines
from .normalizers import norm_market, norm_side
from .storage import now_iso
from edge.odds_espn import american_to_decimal
from propmodel.devig import implied
from src.utils import get_logger

log = get_logger(__name__)
ANCHOR_BOOK = "pinnacle"


def snapshot_props() -> list[dict]:
    """One timestamped row per (book, player, market, side, line) currently posted."""
    props = anchor_lines.fetch_props_multi_region()
    if props.empty:
        return []
    ts = now_iso()
    rows = []
    for r in props.itertuples():
        rows.append({
            "captured_at": ts, "source": "the_odds_api", "bookmaker": r.book,
            "event_id": r.event_id, "player_name": r.player,
            "market_type": norm_market(r.market) or r.market,
            "side": norm_side(r.side) or r.side, "line_value": float(r.line),
            "odds_american": float(r.odds), "odds_decimal": american_to_decimal(r.odds),
            "implied_prob_raw": round(implied(r.odds), 4),
            "is_anchor_book": (r.book == ANCHOR_BOOK),
            "metadata_json": {"home": r.home, "away": r.away, "date": r.date},
        })
    return rows


def capture_snapshot() -> int:
    from .storage import init_storage, insert
    init_storage()
    rows = snapshot_props()
    n = insert("odds_snapshots", rows)
    log.info("captured %d odds-snapshot rows", n)
    return n
