#!/usr/bin/env python3
"""Stage 6 — SGP validation study (Sec 9B).

Quantifies how much correlation SHOULD move a same-game pair price away from the
independence assumption books often use. For each same-team points pair with
anchored marginals + empirical correlation, compares the copula joint to the
independence product, grouped by pair category. (Actual book SGP prices must be
supplied via feeds/sgp_scraper for the final confirmation — see viability note.)

  python run_stage6_sgp_validation.py
"""

from __future__ import annotations

import pandas as pd

import anchor_lines
import pair_engine
from feeds.sgp_scraper import viability

pd.set_option("display.width", 170)


def main():
    props = anchor_lines.fetch_props_multi_region()
    anchor = anchor_lines.build_anchor(props)
    games = pair_engine._load()   # has player_name + team (stands in for proj)
    pairs = pair_engine.find_pairs(anchor, games, props, min_abs_mispricing=0.0)

    print("=" * 84)
    print("STAGE 6 SGP VALIDATION — copula vs independence (live same-team points pairs)")
    print("=" * 84)
    if pairs.empty:
        print("No same-team anchored pairs available right now."); return
    pairs["category"] = pairs["rho"].apply(
        lambda r: "positive-corr (dual-over underpriced)" if r > 0.05
        else "negative-corr (dual-over overpriced)" if r < -0.05 else "near-independent")
    summary = pairs.groupby("category").agg(
        n=("mispricing", "size"),
        mean_abs_dev_from_indep=("mispricing", lambda s: round(s.abs().mean(), 3)),
        max_abs_dev=("mispricing", lambda s: round(s.abs().max(), 3))).reset_index()
    print(summary.to_string(index=False))
    print("\nLargest deviations from independence (where book SGP pricing matters most):")
    show = pairs.reindex(pairs.mispricing.abs().sort_values(ascending=False).index)
    print(show[["team", "player_a", "player_b", "rho", "p_independent",
                "p_joint", "mispricing"]].head(8).to_string(index=False))
    print(f"\nNOTE on confirmation: {viability()['reason']}")
    print("Recommended:", "; ".join(viability()["recommended"]))


if __name__ == "__main__":
    main()
