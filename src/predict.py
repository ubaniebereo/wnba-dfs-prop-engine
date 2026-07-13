"""Generate points predictions for upcoming scheduled games.

For each upcoming game we infer the likely active players (those who appeared in
their team's most recent games), build the SAME leakage-free features used in
training as of the game date, and predict points with the saved model.

CLI:
  python -m src.predict                 # next 7 days
  python -m src.predict --days-ahead 3
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import joblib
import pandas as pd

from . import api_espn, config, database, features
from .utils import get_logger

log = get_logger(__name__)

RECENT_TEAM_GAMES = 3   # infer a team's roster from its last N games


def load_bundle():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {config.MODEL_PATH}. Run `python -m src.train` first.")
    return joblib.load(config.MODEL_PATH)


def upcoming_games(days_ahead: int) -> pd.DataFrame:
    """Scoreboard scan for not-yet-completed games over the next N days."""
    frames = []
    today = date.today()
    for i in range(days_ahead + 1):
        payload = api_espn.fetch_scoreboard(today + timedelta(days=i))
        g = api_espn.scoreboard_to_games(payload)
        if not g.empty:
            frames.append(g[~g["completed"].astype(bool)])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("game_id")


def _recent_roster(players: pd.DataFrame, team_id: str) -> pd.DataFrame:
    """Players from a team's most recent RECENT_TEAM_GAMES games."""
    tp = players[players["team_id"] == str(team_id)]
    if tp.empty:
        return tp
    recent_games = sorted(tp["game_date"].unique())[-RECENT_TEAM_GAMES:]
    roster = tp[tp["game_date"].isin(recent_games)]
    return roster.sort_values("game_date").drop_duplicates("player_id", keep="last")


def _player_features(hist: pd.DataFrame, teams: pd.DataFrame, *, player_id: str,
                     team_id: str, opponent_id: str, is_home: int,
                     game_date: str) -> dict | None:
    """As-of feature vector for one player ahead of `game_date` (no leakage)."""
    prior = hist[(hist["player_id"] == str(player_id)) &
                 (hist["game_date"] < game_date)].sort_values("game_date")
    if prior.empty:
        return None
    season = int(game_date[:4])
    season_prior = prior[prior["game_date"].str[:4].astype(int) == season]
    last_date = prior["game_date"].iloc[-1]
    days_rest = min((datetime.fromisoformat(game_date) -
                     datetime.fromisoformat(last_date)).days, config.MAX_DAYS_REST)
    allowed, pace = features.team_proxy_asof(teams, opponent_id, game_date)
    return {
        "last_3_points": prior["points"].tail(config.ROLL_WINDOWS[0]).mean(),
        "last_5_points": prior["points"].tail(config.ROLL_WINDOWS[1]).mean(),
        "last_3_minutes": prior["minutes"].tail(config.ROLL_WINDOWS[0]).mean(),
        "last_5_minutes": prior["minutes"].tail(config.ROLL_WINDOWS[1]).mean(),
        "season_avg_points": (season_prior["points"].mean()
                              if not season_prior.empty else prior["points"].mean()),
        "season_avg_minutes": (season_prior["minutes"].mean()
                               if not season_prior.empty else prior["minutes"].mean()),
        "home_away_flag": int(is_home),
        "days_rest": float(days_rest),
        "opponent_points_allowed_proxy": allowed,
        "opponent_pace_proxy": pace,
        "history_games": int(len(prior)),
    }


def build_prediction_rows(engine, games: pd.DataFrame) -> pd.DataFrame:
    players = features.load_player_games(engine)
    teams = features.add_team_proxies(features.load_team_games(engine))
    if players.empty:
        log.error("No history in DB — run ingest/features first.")
        return pd.DataFrame()

    rows = []
    for _, g in games.iterrows():
        for side, team_id, opp_id in (
            ("home", g["home_team_id"], g["away_team_id"]),
            ("away", g["away_team_id"], g["home_team_id"]),
        ):
            roster = _recent_roster(players, team_id)
            for _, p in roster.iterrows():
                feat = _player_features(
                    players, teams, player_id=p["player_id"], team_id=team_id,
                    opponent_id=opp_id, is_home=(side == "home"),
                    game_date=g["game_date"])
                if feat is None or feat["history_games"] < config.MIN_HISTORY_GAMES:
                    continue
                feat.update({
                    "game_id": g["game_id"], "game_date": g["game_date"],
                    "player_id": p["player_id"], "player_name": p["player_name"],
                    "team": p["team"],
                    "opponent": g["away_team"] if side == "home" else g["home_team"],
                })
                rows.append(feat)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict upcoming WNBA player points.")
    parser.add_argument("--days-ahead", type=int, default=7)
    args = parser.parse_args(argv)

    engine = database.init_db()
    bundle = load_bundle()
    model, feats, medians = bundle["model"], bundle["features"], bundle["medians"]

    games = upcoming_games(args.days_ahead)
    if games.empty:
        log.warning("No upcoming games found in the next %d days.", args.days_ahead)
        return
    log.info("Found %d upcoming games.", len(games))

    rows = build_prediction_rows(engine, games)
    if rows.empty:
        log.warning("No players had enough history to predict.")
        return

    X = rows[feats].fillna(pd.Series(medians))
    rows["predicted_points"] = model.predict(X).round(2)
    rows["model_name"] = bundle["model_name"]
    rows["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows["prediction_id"] = rows["game_id"] + "_" + rows["player_id"]

    database.upsert(engine, "predictions", rows)
    out = config.OUTPUT_DIR / "predictions.csv"
    cols = ["game_date", "player_name", "team", "opponent",
            "predicted_points", "model_name"]
    rows.sort_values("predicted_points", ascending=False)[cols].to_csv(out, index=False)
    log.info("Wrote %d predictions -> %s (and predictions table).", len(rows), out)
    print(rows.sort_values("predicted_points", ascending=False)[cols].head(20)
          .to_string(index=False))


if __name__ == "__main__":
    main()
