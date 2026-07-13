"""BallDontLie -> our SQLite schema (Module 0 ETL).

Re-sources games + player_game_stats from BallDontLie so ALL history uses one
consistent set of IDs (no cross-source mapping). Exhibition/All-Star squads are
filtered out. Opponent / is_home / a starter proxy (top-5 minutes per team-game)
are derived. Also ingests player positions (Module 1) into `player_positions`.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from balldontlie_client import BallDontLieClient
from src.database import get_engine, init_db, upsert
from src.utils import get_logger

log = get_logger(__name__)

# non-franchise squads to drop (All-Star, national teams, exhibitions)
EXHIBITION = {"AUS", "BRAZIL", "JAPAN", "PUERTORICO", "USA", "EAST", "WEST",
              "WNBASTARS", "CLA", "COL", "DEL", "PAR", "STE", "WIL", "TBD"}


def parse_min(value) -> float:
    if value in (None, ""):
        return 0.0
    s = str(value)
    if ":" in s:
        mm, ss = (s.split(":") + ["0"])[:2]
        return round(int(mm) + int(ss) / 60.0, 2)
    try:
        return float(s)
    except ValueError:
        return 0.0


def _abbr(team: dict) -> str:
    return (team or {}).get("abbreviation", "")


def ingest_games(client: BallDontLieClient, season: int) -> pd.DataFrame:
    rows = []
    for g in client.get_wnba_games(season):
        h, v = _abbr(g.get("home_team")), _abbr(g.get("visitor_team"))
        if h in EXHIBITION or v in EXHIBITION or not h or not v:
            continue
        rows.append({
            "game_id": str(g["id"]), "game_date": (g.get("date") or "")[:10],
            "season": g.get("season"), "status": g.get("status"),
            # BDL marks finished games "post"/"final"; scores present == completed
            "completed": (g.get("home_score") is not None
                          and g.get("away_score") is not None
                          and str(g.get("status", "")).lower() in ("post", "final")),
            "home_team_id": str((g.get("home_team") or {}).get("id")), "home_team": h,
            "home_score": g.get("home_score"),
            "away_team_id": str((g.get("visitor_team") or {}).get("id")), "away_team": v,
            "away_score": g.get("away_score"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        upsert(get_engine(), "games", df)
    return df


def ingest_player_stats(client: BallDontLieClient, season: int,
                        games_df: pd.DataFrame) -> int:
    gmap = {r.game_id: r for r in games_df.itertuples()}
    rows = []
    for s in client.get_wnba_player_stats(season):
        gid = str((s.get("game") or {}).get("id"))
        g = gmap.get(gid)
        if g is None:                       # skipped exhibition game
            continue
        team = s.get("team") or {}
        tid, tabbr = str(team.get("id")), _abbr(team)
        is_home = int(tabbr == g.home_team)
        opp_id = g.away_team_id if is_home else g.home_team_id
        opp = g.away_team if is_home else g.home_team
        p = s.get("player") or {}
        pid = str(p.get("id"))
        rows.append({
            "stat_id": f"{gid}_{pid}", "game_id": gid, "game_date": g.game_date,
            "player_id": pid,
            "player_name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
            "team_id": tid, "team": tabbr, "opponent_id": opp_id, "opponent": opp,
            "is_home": is_home, "starter": None,
            "minutes": parse_min(s.get("min")),
            "points": s.get("pts"), "fgm": s.get("fgm"), "fga": s.get("fga"),
            "tpm": s.get("fg3m"), "tpa": s.get("fg3a"),
            "ftm": s.get("ftm"), "fta": s.get("fta"),
            "rebounds": s.get("reb"), "oreb": s.get("oreb"), "dreb": s.get("dreb"),
            "assists": s.get("ast"), "turnovers": s.get("turnover"),
            "steals": s.get("stl"), "blocks": s.get("blk"),
            "fouls": s.get("pf"), "plus_minus": s.get("plus_minus"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    # starter proxy: top-5 minutes per (game, team)
    df["starter"] = (df.groupby(["game_id", "team_id"])["minutes"]
                     .rank(method="first", ascending=False) <= 5).astype(int)
    upsert(get_engine(), "player_game_stats", df)
    return len(df)


def rebuild_team_game_stats() -> int:
    """Derive team_game_stats (points_for/against) from the games table."""
    eng = get_engine()
    g = pd.read_sql("SELECT * FROM games WHERE completed=1 AND home_score IS NOT NULL", eng)
    rows = []
    for _, r in g.iterrows():
        for is_home, tid, t, oid, o, pf, pa in (
            (1, r.home_team_id, r.home_team, r.away_team_id, r.away_team, r.home_score, r.away_score),
            (0, r.away_team_id, r.away_team, r.home_team_id, r.home_team, r.away_score, r.home_score)):
            rows.append({"team_game_id": f"{r.game_id}_{tid}", "game_id": r.game_id,
                         "game_date": r.game_date, "team_id": tid, "team": t,
                         "opponent_id": oid, "opponent": o, "is_home": is_home,
                         "points_for": pf, "points_against": pa})
    df = pd.DataFrame(rows)
    if not df.empty:
        upsert(eng, "team_game_stats", df)
    return len(df)


def ingest_positions(client: BallDontLieClient) -> pd.DataFrame:
    rows = []
    for p in client.get_wnba_players():
        pos = (p.get("position_abbreviation") or p.get("position") or "").upper()
        cluster = ("big" if "C" in pos else "guard" if pos == "G" else "wing")
        rows.append({"player_id": str(p["id"]),
                     "player_name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
                     "position": pos, "position_cluster": cluster})
    df = pd.DataFrame(rows).drop_duplicates("player_id")
    eng = get_engine()
    with eng.begin() as c:
        c.execute(text("CREATE TABLE IF NOT EXISTS player_positions "
                       "(player_id TEXT PRIMARY KEY, player_name TEXT, "
                       "position TEXT, position_cluster TEXT)"))
        c.execute(text("DELETE FROM player_positions"))
    df.to_sql("player_positions", eng, if_exists="append", index=False)
    return df


def _clear(tables):
    eng = get_engine()
    with eng.begin() as c:
        for t in tables:
            c.execute(text(f"DELETE FROM {t}"))


def backfill(seasons, replace=True) -> pd.DataFrame:
    """Backfill games + player_game_stats for the given seasons from BallDontLie."""
    init_db()
    client = BallDontLieClient()
    if replace:
        _clear(["games", "player_game_stats", "team_game_stats"])
    summary = []
    for yr in seasons:
        gdf = ingest_games(client, yr)
        n_stats = ingest_player_stats(client, yr, gdf) if not gdf.empty else 0
        summary.append({"season": yr, "games": len(gdf), "player_rows": n_stats})
        log.info("season %d: %d games, %d player-rows", yr, len(gdf), n_stats)
    rebuild_team_game_stats()
    pos = ingest_positions(client)
    log.info("positions ingested: %d players", len(pos))
    return pd.DataFrame(summary)
