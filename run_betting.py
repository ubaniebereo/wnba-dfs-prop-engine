#!/usr/bin/env python3
"""Cycle-agnostic betting/outcome layer on REAL ingested WNBA data.

  python run_betting.py

Prints win-probability model metrics (vs an Elo baseline), a calibration table,
and point-margin regression errors.
"""

from __future__ import annotations

import pandas as pd

from betting.outcome_models import run

pd.set_option("display.width", 160)


def main():
    print("=" * 78)
    print("BETTING / OUTCOME LAYER  (real ESPN data, cycle-agnostic)")
    print("=" * 78)
    r = run()
    print(f"Games: train={r['n_train']}  test={r['n_test']}  "
          f"(chronological split at {r['cut_date']})\n")

    print("--- Win probability (home team) ---")
    print(r["win_models"].to_string(index=False))

    print("\n--- Calibration (logistic, home-win prob) ---")
    print(r["calibration_logistic"].to_string())

    print("\n--- Point margin (home - away) ---")
    print(r["margin_models"].to_string(index=False))
    print("\nReads: a richer feature set should at least match Elo; on this small")
    print("(~1 season) sample, expect ROC-AUC in the ~0.6-0.7 range typical of")
    print("basketball outcome models, with margin RMSE near the league's game-to-game SD.")


if __name__ == "__main__":
    main()
