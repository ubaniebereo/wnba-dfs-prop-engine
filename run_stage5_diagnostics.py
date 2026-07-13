#!/usr/bin/env python3
"""Stage 5 diagnostics — CLV + calibration + edge profile over a real sample.

  python run_stage5_diagnostics.py

Reuses the Stage-4 CLV backtest (real historical props) and adds calibration
(Brier/log-loss/reliability) and edge-distribution diagnostics. Honest scoreboard
for whether the engine is moving toward positive CLV + good calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import clv_backtest as clv
from propmodel import diagnostics, models, prop_model
from propmodel.featureset import STATS, build
from src.database import get_engine

pd.set_option("display.width", 170)
SAMPLE_DATES = ["2025-07-15"]      # keep small to limit credits; extend for 200+ sample
TEST_START = "2025-07-01"


def main():
    eng = get_engine()
    feat = build(eng); feat = feat[feat.history >= 5].dropna(subset=["minutes"])
    train = feat[feat.game_date < TEST_START]
    mm = models.train_minutes(train)
    rms = {s: models.train_rate(train, s) for s in STATS}
    params = prop_model.build_params(train, models.project(train, mm, rms), float(mm["sd"] ** 2))

    all_bets = []
    for date in SAMPLE_DATES:
        test = feat[feat.game_date == date]
        proj = models.project(test, mm, rms)
        out_d = test[["player_name"] + STATS]
        placed, closing = clv.collect_snapshots(date)
        if placed.empty:
            continue
        all_bets.append(clv.build_bets(placed, closing, proj, params, out_d,
                                       edge_threshold=0.03, devig_method="shin"))
    bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    if bets.empty:
        print("No bets in sample."); return

    print("=" * 80)
    print("STAGE 5 DIAGNOSTICS (real historical sample)")
    print("=" * 80)
    print("\nCLV:", clv.evaluate_clv(bets))

    b = bets.dropna(subset=["won"])
    cal = diagnostics.evaluate_calibration(b["p_model"], b["won"])
    print(f"\nCalibration: Brier={cal['brier']} log_loss={cal['log_loss']}")
    print(cal["reliability"].to_string())

    ep = diagnostics.evaluate_edge_profile(bets.rename(columns={"side": "side"}))
    print("\nEdge profile:", ep)
    print("\nTargets (200+ bet sample): >=55-60% beat close, non-neg mean CLV,")
    print("Brier<=0.18-0.20, edges mostly 1-5%, over/under balanced. This is a tiny")
    print("sample — scale SAMPLE_DATES to make these statistically meaningful.")


if __name__ == "__main__":
    main()
