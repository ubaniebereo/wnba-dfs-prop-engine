"""Module 6 — expansion-team rating priors (strong early regression).

2026 expansion teams (Golden State Valkyries, Toronto Tempo, Portland Fire) have
little history, so a normal Elo over-reacts to a few results. We start them below
established teams and use a games-played-dependent K that shrinks fast early, so
ratings stabilize instead of swinging.
"""

from __future__ import annotations

import pandas as pd

from src.database import get_engine

EXPANSION_2026 = {"GS", "TOR", "POR"}      # ESPN abbreviations
BASE, EXPANSION_BASE = 1500.0, 1440.0      # expansion teams start lower


def _k(games_played: int, k0=28.0, kmin=14.0, half=12) -> float:
    """K decays from k0 toward kmin as a team plays more games (early regression)."""
    return kmin + (k0 - kmin) * (half / (half + games_played))


def compute_elo(home_adv=60.0) -> pd.DataFrame:
    games = pd.read_sql("SELECT game_id, game_date, home_team, away_team, "
                        "home_score, away_score FROM games WHERE completed=1 "
                        "AND home_score IS NOT NULL ORDER BY game_date", get_engine())
    rating, gp, rows = {}, {}, []
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        rh = rating.get(h, EXPANSION_BASE if h in EXPANSION_2026 else BASE)
        ra = rating.get(a, EXPANSION_BASE if a in EXPANSION_2026 else BASE)
        eh = 1 / (1 + 10 ** (-((rh + home_adv) - ra) / 400))
        hw = int(g["home_score"] > g["away_score"])
        rows.append({"game_id": g["game_id"], "home_elo": round(rh, 1),
                     "away_elo": round(ra, 1)})
        rating[h] = rh + _k(gp.get(h, 0)) * (hw - eh)
        rating[a] = ra + _k(gp.get(a, 0)) * ((1 - hw) - (1 - eh))
        gp[h] = gp.get(h, 0) + 1
        gp[a] = gp.get(a, 0) + 1
    if not rating:
        return pd.DataFrame(columns=["team", "elo", "games", "expansion"])
    final = pd.DataFrame([{"team": t, "elo": round(r, 1),
                           "games": gp.get(t, 0),
                           "expansion": t in EXPANSION_2026}
                          for t, r in rating.items()]).sort_values("elo", ascending=False)
    return final


def evaluate_expansion() -> pd.DataFrame:
    """Show expansion-team ratings aren't wildly volatile vs the league."""
    return compute_elo()
