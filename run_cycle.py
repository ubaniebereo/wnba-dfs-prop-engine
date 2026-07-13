#!/usr/bin/env python3
"""Cycle-sensitivity layer end-to-end on synthetic, ground-truth-labeled data.

  python run_cycle.py

Prints: selection summary, cycle-sensitivity table, ground-truth recovery
(precision/recall), hierarchical league effect, phase-classifier metrics,
an example player's phase probabilities, and a bootstrap of per-phase deltas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cycle_lab import cycle_sensitivity as cs
from cycle_lab import phase_probabilities as pp
from cycle_lab.synth_data import generate

pd.set_option("display.width", 160, "display.max_columns", 40)


def _precision_recall(summary: pd.DataFrame) -> dict:
    tp = int(((summary.cycle_sensitive) & (summary.gt_cycle_sensitive == 1)).sum())
    fp = int(((summary.cycle_sensitive) & (summary.gt_cycle_sensitive == 0)).sum())
    fn = int(((~summary.cycle_sensitive) & (summary.gt_cycle_sensitive == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(prec, 2), "recall": round(rec, 2)}


def main():
    print("=" * 78)
    print("CYCLE-SENSITIVITY LAYER  (synthetic data, known ground truth)")
    print("=" * 78)
    raw = generate()
    df = cs.engineer(raw)
    pool, kept = cs.select_players(df)
    n_sens_truth = raw.groupby("player_id").gt_cycle_sensitive.first().sum()
    print(f"Players generated: {raw.player_id.nunique()} | planted cycle-sensitive "
          f"(truth): {n_sens_truth}")
    print(f"Passed selection (>=25 gms, phase coverage, role stability, fatigue "
          f"variation): {len(kept)} players, {len(pool)} games\n")

    summary = cs.fit_players(pool)
    show = ["player_name", "position_group", "n_games", "med_use",
            "delta_TS_men_z", "delta_PIR_men_z", "robust_after_fatigue",
            "min_phase_p", "cycle_sensitivity_score", "cycle_sensitive",
            "alignment", "gt_cycle_sensitive"]
    print("--- Cycle-sensitivity summary (top 12 by score) ---")
    print(summary[show].head(12).to_string(index=False))

    pr = _precision_recall(summary)
    print(f"\n--- Ground-truth recovery --- {pr}")
    print("(Detector flags the top-quintile robust scorers; precision/recall show how")
    print(" well that isolates the planted small-effect players from pure noise.)")

    print("\n--- Hierarchical pooled effect on delta_TS (MixedLM, random player "
          "intercepts, fatigue-controlled) ---")
    print(cs.hierarchical_effect(pool, "delta_TS").to_string(index=False))

    # phase classifier on the cycle-sensitive (ground-truth) players
    sens_players = summary.loc[summary.gt_cycle_sensitive == 1, "player_id"].tolist() \
        if (summary.gt_cycle_sensitive == 1).any() else kept
    df_sens = pool[pool.player_id.isin(sens_players)].copy()
    print(f"\n--- Phase classifier (cycle-sensitive players: {len(sens_players)}, "
          f"{len(df_sens)} games) ---")
    metrics, calib, bundle = pp.run_classifier(df_sens)
    print(metrics.to_string(index=False))
    print("(Base rate = guessing the majority phase. Performance-only barely beats it;")
    print(" symptoms add the most signal — exactly as the literature implies.)")

    print("\n--- Example phase probabilities (one cycle-sensitive player) ---")
    print(pp.example_player_probabilities(df_sens, bundle).to_string(index=False))

    print("\n--- Simulation: bootstrap mean delta by TRUE phase (90% CI) ---")
    sim = pp.simulate_phase_deltas(df_sens)
    print(sim.to_string(index=False))
    print("\nNote how the phase CIs overlap heavily: effects are real-but-tiny and")
    print("swamped by per-game variance. That overlap IS the honest result.")


if __name__ == "__main__":
    main()
