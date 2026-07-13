"""Ingest WNBA games + box scores from ESPN into SQLite.

Flow:
  1. For each season, scan calendar dates in the season window.
  2. Pull each day's scoreboard -> game rows (and event ids).
  3. For completed games, pull the summary -> player + team box scores.
  4. Upsert everything into SQLite (idempotent; safe to re-run).

CLI:
  python -m src.ingest                       # default seasons from config
  python -m src.ingest --seasons 2025
  python -m src.ingest --start 2025-06-01 --end 2025-06-30
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

import pandas as pd

from . import api_espn, config, database
from .utils import daterange, get_logger

log = get_logger(__name__)


def _season_window(season: int) -> tuple[date, date]:
    start = date(season, *config.SEASON_START_MMDD)
    end = date(season, *config.SEASON_END_MMDD)
    today = date.today()
    return start, min(end, today)


def ingest_dates(engine, start: date, end: date) -> dict[str, int]:
    """Ingest every game between start and end (inclusive). Returns counters."""
    counts = {"games": 0, "player_rows": 0, "team_rows": 0, "summaries": 0}
    for day in daterange(start, end):
        payload = api_espn.fetch_scoreboard(day)
        games_df = api_espn.scoreboard_to_games(payload)
        if games_df.empty:
            continue
        counts["games"] += database.upsert(engine, "games", games_df)
        log.info("%s: %d games", day.isoformat(), len(games_df))

        for _, g in games_df.iterrows():
            if not g["completed"]:
                continue  # box score only exists for finished games
            summary = api_espn.fetch_summary(g["game_id"])
            if summary is None:
                continue
            counts["summaries"] += 1
            players = api_espn.summary_to_player_stats(
                summary, g["game_id"], g["game_date"])
            teams = api_espn.summary_to_team_stats(
                summary, g["game_id"], g["game_date"])
            counts["player_rows"] += database.upsert(
                engine, "player_game_stats", players)
            counts["team_rows"] += database.upsert(
                engine, "team_game_stats", teams)
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest WNBA data from ESPN.")
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=config.DEFAULT_SEASONS)
    parser.add_argument("--start", help="explicit start date YYYY-MM-DD")
    parser.add_argument("--end", help="explicit end date YYYY-MM-DD")
    args = parser.parse_args(argv)

    engine = database.init_db()
    totals = {"games": 0, "player_rows": 0, "team_rows": 0, "summaries": 0}

    try:
        if args.start and args.end:
            s = datetime.strptime(args.start, "%Y-%m-%d").date()
            e = datetime.strptime(args.end, "%Y-%m-%d").date()
            log.info("Ingesting explicit window %s..%s", s, e)
            c = ingest_dates(engine, s, e)
            for k in totals:
                totals[k] += c[k]
        else:
            for season in args.seasons:
                s, e = _season_window(season)
                log.info("Ingesting season %d (%s..%s)", season, s, e)
                c = ingest_dates(engine, s, e)
                for k in totals:
                    totals[k] += c[k]
    except KeyboardInterrupt:
        log.warning("Interrupted — partial data is already committed.")

    log.info("Done. games=%d summaries=%d player_rows=%d team_rows=%d",
             totals["games"], totals["summaries"],
             totals["player_rows"], totals["team_rows"])


if __name__ == "__main__":
    main()
