"""SQLite schema + persistence helpers (SQLAlchemy Core).

Five tables: games, player_game_stats, team_game_stats, model_features,
predictions. Each has a primary key so we can upsert idempotently with
SQLite's INSERT OR REPLACE — re-running ingestion never duplicates rows.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import (Boolean, Column, Float, Integer, MetaData, String,
                        Table, create_engine, text)
from sqlalchemy.engine import Engine

from . import config
from .utils import get_logger

log = get_logger(__name__)
metadata = MetaData()


games = Table(
    "games", metadata,
    Column("game_id", String, primary_key=True),
    Column("game_date", String, index=True),
    Column("season", Integer),
    Column("status", String),
    Column("completed", Boolean),
    Column("home_team_id", String),
    Column("home_team", String),
    Column("home_score", Integer),
    Column("away_team_id", String),
    Column("away_team", String),
    Column("away_score", Integer),
)

player_game_stats = Table(
    "player_game_stats", metadata,
    Column("stat_id", String, primary_key=True),       # game_id + '_' + player_id
    Column("game_id", String, index=True),
    Column("game_date", String, index=True),
    Column("player_id", String, index=True),
    Column("player_name", String),
    Column("team_id", String),
    Column("team", String),
    Column("opponent_id", String),
    Column("opponent", String),
    Column("is_home", Integer),
    Column("starter", Integer),
    Column("minutes", Integer),
    Column("points", Integer),
    Column("fgm", Integer), Column("fga", Integer),
    Column("tpm", Integer), Column("tpa", Integer),
    Column("ftm", Integer), Column("fta", Integer),
    Column("rebounds", Integer), Column("oreb", Integer), Column("dreb", Integer),
    Column("assists", Integer), Column("turnovers", Integer),
    Column("steals", Integer), Column("blocks", Integer),
    Column("fouls", Integer), Column("plus_minus", Float),
)

team_game_stats = Table(
    "team_game_stats", metadata,
    Column("team_game_id", String, primary_key=True),  # game_id + '_' + team_id
    Column("game_id", String, index=True),
    Column("game_date", String, index=True),
    Column("team_id", String, index=True),
    Column("team", String),
    Column("opponent_id", String),
    Column("opponent", String),
    Column("is_home", Integer),
    Column("points_for", Integer),
    Column("points_against", Integer),
)

model_features = Table(
    "model_features", metadata,
    Column("stat_id", String, primary_key=True),
    Column("game_id", String, index=True),
    Column("game_date", String, index=True),
    Column("player_id", String, index=True),
    Column("player_name", String),
    Column("team", String),
    Column("opponent", String),
    Column("points", Integer),                         # target (NULL for upcoming)
    Column("last_3_points", Float),
    Column("last_5_points", Float),
    Column("last_3_minutes", Float),
    Column("last_5_minutes", Float),
    Column("season_avg_points", Float),
    Column("season_avg_minutes", Float),
    Column("home_away_flag", Integer),
    Column("days_rest", Float),
    Column("opponent_points_allowed_proxy", Float),
    Column("opponent_pace_proxy", Float),
    Column("history_games", Integer),                  # prior games available
)

predictions = Table(
    "predictions", metadata,
    Column("prediction_id", String, primary_key=True),  # game_id + '_' + player_id
    Column("game_id", String, index=True),
    Column("game_date", String, index=True),
    Column("player_id", String, index=True),
    Column("player_name", String),
    Column("team", String),
    Column("opponent", String),
    Column("predicted_points", Float),
    Column("model_name", String),
    Column("created_at", String),
)

# Map table-name -> (Table, primary-key column) for the generic upsert helper.
_PK = {
    "games": "game_id",
    "player_game_stats": "stat_id",
    "team_game_stats": "team_game_id",
    "model_features": "stat_id",
    "predictions": "prediction_id",
}


def get_engine() -> Engine:
    """Create (or reuse) the SQLite engine."""
    return create_engine(config.DB_URL, future=True)


def init_db(engine: Engine | None = None) -> Engine:
    """Create all tables if they do not already exist."""
    engine = engine or get_engine()
    metadata.create_all(engine)
    log.info("Initialized database at %s", config.DB_PATH)
    return engine


def upsert(engine: Engine, table_name: str, df: pd.DataFrame) -> int:
    """Idempotently write a DataFrame using INSERT OR REPLACE on the PK.

    Returns the number of rows written. Extra DataFrame columns not in the
    table are ignored; missing columns are filled by the table defaults.
    """
    if df is None or df.empty:
        return 0
    if table_name not in _PK:
        raise ValueError(f"Unknown table {table_name!r}")
    table = metadata.tables[table_name]
    cols = [c.name for c in table.columns if c.name in df.columns]
    clean = df[cols].where(pd.notnull(df[cols]), None)
    records = clean.to_dict(orient="records")

    placeholders = ", ".join(f":{c}" for c in cols)
    col_list = ", ".join(cols)
    stmt = text(f"INSERT OR REPLACE INTO {table_name} ({col_list}) "
                f"VALUES ({placeholders})")
    with engine.begin() as conn:
        conn.execute(stmt, records)
    log.debug("Upserted %d rows into %s", len(records), table_name)
    return len(records)


def read_sql(engine: Engine, query: str) -> pd.DataFrame:
    """Convenience wrapper returning a DataFrame for a SQL string."""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)
