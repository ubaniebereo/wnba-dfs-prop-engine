"""Build leakage-free rolling features for the points model.

Leakage rule (critical): every feature for a target game uses ONLY games that
finished strictly before that game's date. This is enforced with a per-player
`.shift(1)` before any rolling/expanding window, so the target game's own stats
never leak into its own features.

Produces the `model_features` table (and a CSV). The same as-of helper functions
are reused by predict.py so historical and future features are computed
identically.

CLI:
  python -m src.features
"""

from __future__ import annotations

import pandas as pd

from . import config, database
from .utils import get_logger

log = get_logger(__name__)

W3, W5 = config.ROLL_WINDOWS[0], config.ROLL_WINDOWS[1]
PROXY_W = config.TEAM_PROXY_WINDOW


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_player_games(engine) -> pd.DataFrame:
    df = database.read_sql(engine, "SELECT * FROM player_game_stats")
    if not df.empty:
        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return df


def load_team_games(engine) -> pd.DataFrame:
    df = database.read_sql(engine, "SELECT * FROM team_game_stats")
    if not df.empty:
        df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Team (opponent) proxies
# ---------------------------------------------------------------------------
def add_team_proxies(teams: pd.DataFrame) -> pd.DataFrame:
    """Add each team's pre-game rolling points-allowed and pace (shift(1))."""
    teams = teams.sort_values(["team_id", "game_date"]).copy()
    teams["pace_raw"] = teams["points_for"].fillna(0) + teams["points_against"].fillna(0)
    grp = teams.groupby("team_id", group_keys=False)
    teams["team_prior_allowed"] = grp["points_against"].apply(
        lambda s: s.shift(1).rolling(PROXY_W, min_periods=1).mean())
    teams["team_prior_pace"] = grp["pace_raw"].apply(
        lambda s: s.shift(1).rolling(PROXY_W, min_periods=1).mean())
    return teams


def team_proxy_asof(teams: pd.DataFrame, team_id: str, as_of_date: str
                    ) -> tuple[float | None, float | None]:
    """Opponent proxy for a FUTURE game: rolling stats over games before date."""
    prior = teams[(teams["team_id"] == str(team_id)) &
                  (teams["game_date"] < as_of_date)].tail(PROXY_W)
    if prior.empty:
        return None, None
    allowed = prior["points_against"].mean()
    pace = (prior["points_for"].fillna(0) + prior["points_against"].fillna(0)).mean()
    return float(allowed), float(pace)


# ---------------------------------------------------------------------------
# Player rolling features
# ---------------------------------------------------------------------------
def engineer(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Vectorized leakage-free features for every completed player-game row."""
    if players.empty:
        return pd.DataFrame()
    df = players.sort_values(["player_id", "game_date"]).copy()
    df["season"] = df["game_date"].str[:4].astype(int)
    g = df.groupby("player_id", group_keys=False)

    df["last_3_points"] = g["points"].apply(lambda s: s.shift(1).rolling(W3, min_periods=1).mean())
    df["last_5_points"] = g["points"].apply(lambda s: s.shift(1).rolling(W5, min_periods=1).mean())
    df["last_3_minutes"] = g["minutes"].apply(lambda s: s.shift(1).rolling(W3, min_periods=1).mean())
    df["last_5_minutes"] = g["minutes"].apply(lambda s: s.shift(1).rolling(W5, min_periods=1).mean())

    sg = df.groupby(["player_id", "season"], group_keys=False)
    df["season_avg_points"] = sg["points"].apply(lambda s: s.shift(1).expanding().mean())
    df["season_avg_minutes"] = sg["minutes"].apply(lambda s: s.shift(1).expanding().mean())

    df["home_away_flag"] = df["is_home"].astype(int)
    df["history_games"] = g.cumcount()
    prev_date = g["game_date"].shift(1)
    df["days_rest"] = (pd.to_datetime(df["game_date"]) -
                       pd.to_datetime(prev_date)).dt.days.clip(upper=config.MAX_DAYS_REST)

    # opponent proxies: join the opponent's SAME-game pre-game rolling row
    teams_p = add_team_proxies(teams)
    proxy = (teams_p[["game_id", "team_id", "team_prior_allowed", "team_prior_pace"]]
             .rename(columns={"team_id": "opponent_id"}))
    df = df.merge(proxy, on=["game_id", "opponent_id"], how="left")
    df["opponent_points_allowed_proxy"] = df["team_prior_allowed"]
    df["opponent_pace_proxy"] = df["team_prior_pace"]

    keep = (["stat_id", "game_id", "game_date", "player_id", "player_name",
             "team", "opponent", "points"] + config.FEATURE_COLUMNS +
            ["history_games"])
    return df[keep]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    engine = database.init_db()
    players = load_player_games(engine)
    teams = load_team_games(engine)
    if players.empty:
        log.error("No player_game_stats found — run `python -m src.ingest` first.")
        return

    feats = engineer(players, teams)
    database.upsert(engine, "model_features", feats)
    out = config.OUTPUT_DIR / "model_features.csv"
    feats.to_csv(out, index=False)
    usable = int((feats["history_games"] >= config.MIN_HISTORY_GAMES).sum())
    log.info("Built %d feature rows (%d with >=%d prior games). Saved %s",
             len(feats), usable, config.MIN_HISTORY_GAMES, out)


if __name__ == "__main__":
    main()
