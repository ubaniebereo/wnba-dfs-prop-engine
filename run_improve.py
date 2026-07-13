#!/usr/bin/env python3
"""Improved WNBA prop pipeline — full run + evaluation routines.

  python run_improve.py            # backtest eval + (if ODDS_API_KEY) live edges
  python run_improve.py --backfill # also ingest earlier seasons first

Pipeline: features -> minutes x rate models -> distribution params -> isotonic
calibration -> market blend -> edges vs real props, with diagnostics at each step.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from propmodel import calibration, evaluate, expansion, injuries, live_edge, models
from propmodel.featureset import STATS, build
from src.database import get_engine

pd.set_option("display.width", 170, "display.max_columns", 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-ahead", type=int, default=3)
    ap.add_argument("--backfill", action="store_true")
    args = ap.parse_args()
    eng = get_engine()

    if args.backfill:
        from propmodel.multiseason import backfill
        print("Backfilling earlier seasons (this can take a few minutes)...")
        backfill()

    # ---------- 1. BACKTEST: does the new model beat old RF + recency? ----------
    print("=" * 84)
    print("BACKTEST — minutes x rate model vs old RF vs recency (leakage-free)")
    print("=" * 84)
    bt = evaluate.run(eng)
    print(f"split: {bt['n_train']} train / {bt['n_test']} test (cut {bt['cut_date']})")
    print(f"minutes model: MAE={bt['minutes']['MAE']} RMSE={bt['minutes']['RMSE']}\n")
    print("per-stat error + STAR under-projection bias:")
    print(bt["per_stat"].to_string(index=False))
    print("\ncount-distribution fit:")
    print(bt["dispersion"].to_string(index=False))

    # ---------- 2. CALIBRATION improvement (out-of-sample) ----------
    print("\n" + "=" * 84)
    print("CALIBRATION — isotonic correction (out-of-sample)")
    print("=" * 84)
    feat = build(eng)
    d = feat[feat["history"] >= 5].dropna(subset=["minutes"])
    dates = np.sort(d["game_date"].unique())
    c1, c2 = dates[int(len(dates) * .6)], dates[int(len(dates) * .8)]
    tr, cal, te = d[d.game_date < c1], d[(d.game_date >= c1) & (d.game_date < c2)], d[d.game_date >= c2]
    mm = models.train_minutes(tr)
    rms = {s: models.train_rate(tr, s) for s in STATS}
    params = models.fit_distribution_params(tr, models.project(tr, mm, rms))
    pc = calibration.build_calibration_pairs(cal, models.project(cal, mm, rms), params)
    pt = calibration.build_calibration_pairs(te, models.project(te, mm, rms), params)
    iso = calibration.fit_isotonic(pc)
    print(f"Brier  raw={calibration.brier(pt):.4f}  ->  isotonic={calibration.brier(pt, iso):.4f}")

    # ---------- 3. INJURIES ----------
    print("\n" + "=" * 84)
    print("INJURIES — exclude OUT players from pricing")
    print("=" * 84)
    inj = injuries.evaluate_injuries()
    print(f"  {inj}")
    out_ids = injuries.out_player_ids()

    # ---------- 4. EXPANSION-team ratings ----------
    print("\n" + "=" * 84)
    print("EXPANSION-TEAM ELO (strong early regression)")
    print("=" * 84)
    elo = expansion.evaluate_expansion()
    print(elo.head(16).to_string(index=False))

    # ---------- 5. LIVE EDGES with calibrated + blended new model ----------
    print("\n" + "=" * 84)
    print("LIVE EDGE PROFILE — new model, calibrated + market-blended")
    print("=" * 84)
    if not os.environ.get("ODDS_API_KEY"):
        print("  (No ODDS_API_KEY -> skipping live props.)")
        return
    # refit on ALL data for live use, params + iso from the splits above
    mm_full = models.train_minutes(d)
    rms_full = {s: models.train_rate(d, s) for s in STATS}
    proj = live_edge.project_upcoming(eng, mm_full, rms_full,
                                      days_ahead=args.days_ahead, out_ids=out_ids)
    if proj.empty:
        print("  No upcoming projectable players.")
        return
    from edge.odds_api import fetch_player_props
    props = fetch_player_props()
    e = live_edge.edges(proj, props, params, iso, w_blend=0.5)
    print("  edge profile:", live_edge.edge_profile(e))
    if not e.empty:
        cols = ["player", "market", "side", "line", "book", "odds", "E_blend",
                "p_model", "p_book_fair", "edge", "plausibility"]
        print("\n  Top edges (calibrated + blended):")
        print(e[e.plausibility == "plausible"][cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
