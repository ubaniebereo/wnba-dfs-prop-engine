#!/usr/bin/env python3
"""Stage 3 — confirmed lineups + variance/distribution + pooling + devig + CLV prep.

  python run_stage3.py

Retrains with real starters, fits Negative-Binomial dispersion + law-of-total-
variance, compares NB vs Normal calibration for counts, evaluates hierarchical
pooling and devig methods, logs an odds snapshot, and prints honest diagnostics.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import lineups_etl
import usage_model
import injury_notes_etl
from propmodel import (calibration, devig, diagnostics, distributions,
                       evaluate, hierarchical, models)
from propmodel.featureset import STATS, build
from src.database import get_engine

pd.set_option("display.width", 180, "display.max_columns", 40)
H = "=" * 86


def main():
    eng = get_engine()

    print(H + "\n1) CONFIRMED LINEUPS — replace top-5-minutes proxy with real ESPN starters\n" + H)
    print("  backfill:", lineups_etl.from_espn_backup())
    print("  lineup check:", lineups_etl.evaluate_lineups())

    print("\n" + H + "\n2) BACKTEST with real starters + rotation size (vs old RF / recency)\n" + H)
    bt = evaluate.run(eng)
    print(f"  split {bt['n_train']}/{bt['n_test']} | minutes MAE={bt['minutes']['MAE']} "
          f"RMSE={bt['minutes']['RMSE']}")
    print(bt["per_stat"].to_string(index=False))

    print("\n" + H + "\n3) DISTRIBUTIONS — NB(MLE) vs Normal calibration for counts (Brier)\n" + H)
    feat = build(eng)
    d = feat[feat.history >= 5].dropna(subset=["minutes"])
    dates = np.sort(d.game_date.unique())
    cut = dates[int(len(dates) * 0.75)]
    tr, te = d[d.game_date < cut], d[d.game_date >= cut]
    mm = models.train_minutes(tr)
    rms = {s: models.train_rate(tr, s) for s in STATS}
    var_min = float(mm["sd"] ** 2)
    proj_tr, proj_te = models.project(tr, mm, rms), models.project(te, mm, rms)
    for stat in ("rebounds", "assists"):
        m = te.merge(proj_te[["game_id", "player_id", f"E_{stat}"]], on=["game_id", "player_id"])
        mu, actual = m[f"E_{stat}"].values, m[stat].values
        r = distributions.fit_nb_dispersion(actual, mu)
        # rate variance for LTV (Normal alt)
        mt = tr.merge(proj_tr[["game_id", "player_id", f"E_{stat}"]], on=["game_id", "player_id"])
        var_rate = float(np.var((mt[stat] / mt["minutes"].clip(lower=1)) -
                                (mt[f"E_{stat}"] / mt["minutes"].clip(lower=1))))
        b_nb, b_no, yy = [], [], []
        for _, row in m.iterrows():
            for k in (np.floor(row[f"E_{stat}"]) - 0.5, np.floor(row[f"E_{stat}"]) + 1.5):
                if k <= 0:
                    continue
                vstat = distributions.total_variance(mm_e := row.get("E_min", mu.mean()),
                                                     var_min, row[f"E_{stat}"] / 25, var_rate)
                b_nb.append(distributions.prob_over_nb(row[f"E_{stat}"], r, k))
                b_no.append(distributions.prob_over_normal(row[f"E_{stat}"], max(vstat, 1.0), k))
                yy.append(int(row[stat] > k))
        yy = np.array(yy)
        print(f"  {stat:9s} NB r={r:5.1f} | Brier  NB={np.mean((np.array(b_nb)-yy)**2):.4f}  "
              f"Normal={np.mean((np.array(b_no)-yy)**2):.4f}  (lower=better)")

    print("\n" + H + "\n4) HIERARCHICAL POOLING (MixedLM) — rate MAE pooled vs raw\n" + H)
    print("  ", hierarchical.evaluate_pooling(d, "rebounds"))

    print("\n" + H + "\n5) DEVIG methods (-110/-110 and -150/+130 examples)\n" + H)
    for a, b in [(-110, -110), (-150, 130)]:
        row = {m: devig.devig(a, b, m) for m in ("multiplicative", "shin", "logit")}
        print(f"  {a}/{b}: " + " | ".join(f"{k}={v[0]:.3f}/{v[1]:.3f}" for k, v in row.items()))

    print("\n" + H + "\n6) USAGE redistribution (WOWY: usage/min by #starters out)\n" + H)
    print(usage_model.evaluate_usage().to_string())

    print("\n" + H + "\n7) INJURY NOTES (minute-cap NLP) + ODDS SNAPSHOT (CLV prep)\n" + H)
    print("  ", injury_notes_etl.evaluate_injury_notes())
    if os.environ.get("ODDS_API_KEY"):
        try:
            from edge.odds_api import fetch_player_props
            props = fetch_player_props()
            print("  odds snapshot logged rows:", diagnostics.log_odds_snapshot(props))
        except Exception as e:
            print("  odds snapshot skipped:", e)


if __name__ == "__main__":
    main()
