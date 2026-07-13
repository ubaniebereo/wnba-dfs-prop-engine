#!/usr/bin/env python3
"""Train per-family models + calibration; print the report with 'why chosen'.

  ./.venv/bin/python run_family_calibration.py

Also persists the fitted models + calibrators to models/family_calibration.joblib
so the scanner can apply calibrated probabilities.
"""

from __future__ import annotations

import joblib
import pandas as pd

from propmodel.family_calibration import calibrate_all
from src.config import MODELS_DIR

pd.set_option("display.width", 200, "display.max_colwidth", 130)


def main():
    print("Training per-family models + calibration (counts -> NB/Poisson, points -> Normal)...\n")
    report, store = calibrate_all()

    print("=" * 100)
    print("PER-FAMILY MODEL + CALIBRATION REPORT")
    print("=" * 100)
    cols = ["family", "n_test", "distribution", "mean_proj", "MAE",
            "calib_method", "brier_raw", "brier_chosen"]
    print(report[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("WHY EACH CHOICE WAS MADE")
    print("=" * 100)
    for r in report.itertuples():
        print(f"\n• {r.family.upper()}")
        print(f"    {r.why_chosen}")

    path = MODELS_DIR / "family_calibration.joblib"
    joblib.dump(store, path)
    print(f"\nSaved per-family models + calibrators -> {path}")


if __name__ == "__main__":
    main()
