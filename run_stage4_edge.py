#!/usr/bin/env python3
"""Stage 4 — NB + Shin devig wired into live edge; re-measure the profile.

  python run_stage4_edge.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import lineups_etl
from propmodel import calibration, live_edge, models, prop_model
from propmodel.featureset import STATS, build
from src.database import get_engine

pd.set_option("display.width", 180, "display.max_columns", 40)
H = "=" * 86


def main():
    eng = get_engine()
    lineups_etl.from_espn_backup()                 # ensure real starters

    feat = build(eng)
    d = feat[feat.history >= 5].dropna(subset=["minutes"])
    dates = np.sort(d.game_date.unique())
    c1, c2 = dates[int(len(dates) * .6)], dates[int(len(dates) * .8)]
    tr = d[d.game_date < c1]
    cal = d[(d.game_date >= c1) & (d.game_date < c2)]

    mm = models.train_minutes(tr)
    rms = {s: models.train_rate(tr, s) for s in STATS}
    params = prop_model.build_params(tr, models.project(tr, mm, rms), float(mm["sd"] ** 2))
    print("Stage-4 distribution params:")
    print(f"  NB dispersion r: rebounds={params['nb_r']['rebounds']:.1f} "
          f"assists={params['nb_r']['assists']:.1f}")
    print(f"  points variance (LTV): {params['var_stat']['points']:.1f}")

    iso = calibration.fit_isotonic(
        calibration.build_calibration_pairs(cal, models.project(cal, mm, rms),
                                            {"r": params["nb_r"], "sd": params["var_stat"]}))

    if not os.environ.get("ODDS_API_KEY"):
        print("\nNo ODDS_API_KEY -> cannot pull live props.")
        return
    mm_full = models.train_minutes(d)
    rms_full = {s: models.train_rate(d, s) for s in STATS}
    from propmodel import injuries
    proj = live_edge.project_upcoming(eng, mm_full, rms_full, days_ahead=3,
                                      out_ids=injuries.out_player_ids())
    from edge.odds_api import fetch_player_props
    props = fetch_player_props()

    print("\n" + H + "\nLIVE EDGE PROFILE — Stage 4 (NB + Shin devig) vs Stage 3 (Normal + mult.)\n" + H)
    for label, method, p in (("Stage3 (Normal+mult)", "multiplicative",
                              {"nb_r": {}, "var_stat": {s: params["var_stat"][s] for s in STATS}}),
                             ("Stage4 (NB+Shin)", "shin", params)):
        e = live_edge.edges(proj, props, p, iso, w_blend=0.5,
                            edge_threshold=0.03, devig_method=method)
        prof = prop_model.evaluate_live_edge_distribution(e)
        print(f"  {label:22s}: {prof}")


if __name__ == "__main__":
    main()
