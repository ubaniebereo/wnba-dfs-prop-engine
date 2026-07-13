#!/usr/bin/env python3
"""Stage 4 CLV backtest on REAL historical WNBA props.

  python run_stage4_clv.py

Trains the prop model on data before a test window, projects test-date games,
fetches placed + closing historical props, builds model bets, and reports CLV
(did we beat the close?) and realized hit rate. Costs Odds API historical credits.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import clv_backtest as clv
from propmodel import models, prop_model
from propmodel.featureset import STATS, build
from src.database import get_engine

pd.set_option("display.width", 180, "display.max_columns", 40)

# a few 2025 in-season dates to sample (kept small to limit credit spend)
TEST_DATES = ["2025-07-08", "2025-07-11", "2025-07-15"]
TEST_START = "2025-07-01"


def main():
    eng = get_engine()
    feat = build(eng)
    feat = feat[feat.history >= 5].dropna(subset=["minutes"])
    train = feat[feat.game_date < TEST_START]
    test = feat[feat.game_date.isin(TEST_DATES)]
    if test.empty:
        print("No test rows for those dates."); sys.exit(0)

    mm = models.train_minutes(train)
    rms = {s: models.train_rate(train, s) for s in STATS}
    params = prop_model.build_params(train, models.project(train, mm, rms),
                                     float(mm["sd"] ** 2))
    proj_all = models.project(test, mm, rms)

    all_bets = []
    for date in TEST_DATES:
        proj_d = proj_all[proj_all.game_date == date]
        out_d = test[test.game_date == date][["player_name"] + STATS]
        if proj_d.empty:
            continue
        print(f"fetching historical props for {date} ...", file=sys.stderr)
        placed, closing = clv.collect_snapshots(date)
        if placed.empty:
            print(f"  no historical props for {date}"); continue
        bets = clv.build_bets(placed, closing, proj_d, params, out_d,
                              edge_threshold=0.03, devig_method="shin")
        print(f"  {date}: {len(bets)} model bets")
        all_bets.append(bets)

    bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    print("\n" + "=" * 80)
    print("CLV BACKTEST RESULT (real historical props)")
    print("=" * 80)
    res = clv.evaluate_clv(bets)
    print(res)
    if not bets.empty:
        print("\nCLV interpretation: clv_prob>0 => placed price beat the close (good).")
        print("Sample bets:")
        cols = ["player", "market", "side", "line", "placed_odds", "close_odds",
                "p_model", "edge", "clv_prob", "beat_close", "won"]
        print(bets.dropna(subset=["clv_prob"])[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
