"""Durable time-series storage for the live intelligence layer (Stage 6, Sec 4).

SQLAlchemy Core schema for odds snapshots, news, lineups, SGP quotes, and line
moves, with indexes tuned for recent-by-event lookups, news->move joins, and
reconstructing the last pre-tip close.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (JSON, Boolean, Column, Float, Index, Integer, MetaData,
                        String, Table, create_engine, text)

from src.config import DATA_DIR

ENGINE = create_engine(f"sqlite:///{DATA_DIR / 'feeds.sqlite'}", future=True)
meta = MetaData()


def _id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


odds_snapshots = Table(
    "odds_snapshots", meta,
    Column("snapshot_id", String, primary_key=True),
    Column("captured_at", String, index=True), Column("source", String),
    Column("bookmaker", String), Column("event_id", String, index=True),
    Column("game_id", String), Column("player_id", String), Column("team_id", String),
    Column("player_name", String), Column("market_type", String), Column("side", String),
    Column("line_value", Float), Column("odds_american", Float),
    Column("odds_decimal", Float), Column("implied_prob_raw", Float),
    Column("implied_prob_devig", Float), Column("is_anchor_book", Boolean),
    Column("metadata_json", JSON),
)
news_events = Table(
    "news_events", meta,
    Column("news_event_id", String, primary_key=True), Column("source", String),
    Column("captured_at", String, index=True), Column("event_type", String),
    Column("player_id", String), Column("player_name", String), Column("team_id", String),
    Column("game_id", String), Column("raw_text", String),
    Column("parsed_status", String), Column("confidence", Float),
    Column("metadata_json", JSON),
)
lineup_events = Table(
    "lineup_events", meta,
    Column("lineup_event_id", String, primary_key=True), Column("source", String),
    Column("captured_at", String, index=True), Column("game_id", String, index=True),
    Column("player_id", String), Column("player_name", String), Column("team_id", String),
    Column("is_starter", Integer), Column("is_active", Integer),
    Column("projected_vs_confirmed", String), Column("metadata_json", JSON),
)
sgp_quotes = Table(
    "sgp_quotes", meta,
    Column("quote_id", String, primary_key=True), Column("captured_at", String, index=True),
    Column("bookmaker", String), Column("event_id", String),
    Column("legs_json", JSON), Column("parlay_odds_american", Float),
    Column("parlay_odds_decimal", Float), Column("implied_prob_raw", Float),
    Column("implied_prob_devig", Float), Column("metadata_json", JSON),
)
line_move_events = Table(
    "line_move_events", meta,
    Column("move_id", String, primary_key=True), Column("event_id", String, index=True),
    Column("bookmaker", String), Column("player_id", String), Column("player_name", String),
    Column("market_type", String), Column("side", String),
    Column("old_line", Float), Column("new_line", Float),
    Column("old_odds", Float), Column("new_odds", Float),
    Column("moved_at", String, index=True), Column("trigger_news_event_id", String),
    Column("anchor_already_moved", Boolean), Column("metadata_json", JSON),
)

Index("ix_odds_recent", odds_snapshots.c.event_id, odds_snapshots.c.bookmaker,
      odds_snapshots.c.player_name, odds_snapshots.c.market_type, odds_snapshots.c.captured_at)
Index("ix_news_player_time", news_events.c.player_name, news_events.c.captured_at)


def init_storage():
    meta.create_all(ENGINE)
    # WAL lets the dashboard READ while the background scanner WRITES (no locks).
    with ENGINE.begin() as c:
        c.exec_driver_sql("PRAGMA journal_mode=WAL")
    return ENGINE


def insert(table_name: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    table = meta.tables[table_name]
    cols = {c.name for c in table.columns}
    clean = []
    for r in rows:
        rr = {k: (json.dumps(v) if isinstance(v, (dict, list)) and k.endswith("_json") else v)
              for k, v in r.items() if k in cols}
        rr.setdefault(list(table.primary_key.columns)[0].name, _id())
        clean.append(rr)
    with ENGINE.begin() as c:
        c.execute(table.insert(), clean)
    return len(clean)


def read(query: str, params=None) -> pd.DataFrame:
    with ENGINE.connect() as c:
        return pd.read_sql(text(query), c, params=params)
