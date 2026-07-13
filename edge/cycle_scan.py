"""Run the real-data cycle scanner 'just in case' and report honestly.

No cycle labels exist for real players (private medical data), so we cannot use
cycle phase as a feature. What we CAN do is scan each player's real points series
for a 21-35 day periodicity (Lomb-Scargle + permutation) and check whether ANY
survives multiple-comparison correction. Prior runs returned null; this re-checks
on the current ESPN data and returns the counts so the claim stays evidence-based.
"""

from __future__ import annotations

import numpy as np

from cross_season import bh_fdr
from src.database import get_engine, read_sql
from src.utils import get_logger
from wnba_cycles.analysis import flag_player

log = get_logger(__name__)


def scan(min_games=15) -> dict:
    eng = get_engine()
    df = read_sql(eng, "SELECT * FROM player_game_stats WHERE minutes > 0")
    if df.empty:
        return {"players": 0, "raw_p05": 0, "fdr_survivors": 0}
    results = []
    for pid, d in df.groupby("player_id"):
        rows = [{**r, "game": {"date": r["game_date"]}} for r in d.to_dict("records")]
        res = flag_player(str(pid), d["player_name"].iloc[0], rows, n_perm=600)
        if res is not None:
            results.append(res)
    if not results:
        return {"players": 0, "raw_p05": 0, "fdr_survivors": 0}
    pvals = [r.p_value for r in results]
    disc, cut = bh_fdr(pvals, q=0.10)
    return {
        "players": len(results),
        "raw_p05": int(sum(p < 0.05 for p in pvals)),
        "expected_by_chance": round(0.05 * len(results), 1),
        "fdr_survivors": int(disc.sum()),
        "fdr_cutoff": round(cut, 4) if cut else None,
    }
