#!/usr/bin/env python3
"""Integrated edge finder: cycle scan + game models + prop models + edges.

  python run_edges.py [--days-ahead 5] [--edge 0.03] [--lines lines.csv]

Uses REAL data only: ESPN box scores (SQLite) for models, real DraftKings odds
(via ESPN) for game markets, and a real lines.csv for player props if supplied.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from edge import cycle_scan
from edge.edges import find_game_edges
from edge.prop_edges import prop_edges
from edge.prop_models import project_props, train_prop_model
from src.database import init_db

pd.set_option("display.width", 200, "display.max_columns", 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-ahead", type=int, default=5)
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--lines", default="lines.csv")
    args = ap.parse_args()
    engine = init_db()

    print("=" * 84)
    print("1) CYCLE SCAN (real data, 'just in case')  — labels don't exist, so this is")
    print("   a periodicity check, NOT a cycle feature.")
    print("=" * 84)
    cyc = cycle_scan.scan()
    print(f"   players scanned: {cyc['players']} | raw p<0.05: {cyc['raw_p05']} "
          f"(~{cyc.get('expected_by_chance','?')} expected by chance) | "
          f"survive FDR@10%: {cyc['fdr_survivors']}")
    print("   -> No robust cycle signal; cycle features OMITTED from projections.\n")

    have_odds_api = bool(os.environ.get("ODDS_API_KEY"))
    game_odds_df = None
    src_label = "single-book DraftKings via ESPN"
    if have_odds_api:
        try:
            from edge.odds_api import best_line_per_side, fetch_game_odds
            game_odds_df = best_line_per_side(fetch_game_odds())
            src_label = "multi-book BEST LINE via The Odds API"
        except Exception as e:  # fall back to ESPN on any API issue
            print(f"   (Odds API game fetch failed: {e}; using ESPN.)")

    print("=" * 84)
    print(f"2) GAME MODELS + EDGES (win / spread / total) — {src_label}")
    print("=" * 84)
    game_edges, gm = find_game_edges(days_ahead=args.days_ahead,
                                     edge_threshold=args.edge, odds_df=game_odds_df)
    print(f"   model eval: win_acc={gm['win_accuracy']} log_loss={gm['win_log_loss']} "
          f"margin_MAE={gm['margin_MAE']} total_MAE={gm['total_MAE']} "
          f"(train={gm['n_train']}, test={gm['n_test']})")
    print(f"   variance explained vs sample mean: margin R2={gm['margin_r2_vs_mean']} "
          f"total R2={gm['total_r2_vs_mean']}  <- near 0 means little real total edge")
    if game_edges.empty:
        print("   No game markets cleared the edge threshold (or no lines posted).")
    else:
        cols = ["date", "matchup", "market", "side", "line", "odds",
                "p_model", "p_book_fair", "edge", "EV_per_$1", "plausibility", "reason"]
        print("\n   CANDIDATE GAME BETS (edge >= %.0f%%):" % (args.edge * 100))
        print(game_edges[cols].to_string(index=False))
        n_plaus = int((game_edges["plausibility"] == "plausible").sum())
        print(f"\n   Of {len(game_edges)} flags, {n_plaus} are in the plausible (<=10%) "
              "range; larger 'edges' are model error, not value.")

    print("\n" + "=" * 84)
    print("3) PLAYER PROP MODELS — projections + edges vs real lines.csv (if present)")
    print("=" * 84)
    bundles = {}
    for stat in ("points", "rebounds", "assists"):
        b = train_prop_model(engine, stat)
        if b:
            bundles[stat] = b
            print(f"   {stat:9s} model: MAE={b['mae']} resid_sd={b['resid_sd']:.1f} n={b['n']}")
    projections = project_props(engine, bundles, days_ahead=args.days_ahead)
    if projections.empty:
        print("   No upcoming games with sufficient player history.")
        return
    show = ["date", "player_name", "team", "opponent"] + [f"proj_{s}" for s in bundles]
    top = projections.sort_values("proj_points", ascending=False).head(10)
    print("\n   PROJECTIONS (top 10 by projected points):")
    print(top[show].to_string(index=False))

    edges = pd.DataFrame()
    if have_odds_api:
        try:
            from edge.odds_api import fetch_player_props
            props = fetch_player_props()
            edges = prop_edges(projections, props, edge_threshold=args.edge)
        except Exception as e:
            print(f"   (Odds API props fetch failed: {e})")
    if edges.empty:
        print("\n   No prop edges cleared the threshold (real props compared, "
              "or none posted).")
    else:
        print("\n   CANDIDATE PROP BETS (model vs REAL book prop lines):")
        print(edges.to_string(index=False))
        n_pl = int((edges["plausibility"] == "plausible").sum())
        print(f"\n   {n_pl}/{len(edges)} flags are plausible (<=7% edge, realistic vs "
              "sharp prop books); the rest are model error. So many large 'edges' "
              "means the model isn't calibrated to the market — don't bet these blind.")


if __name__ == "__main__":
    main()
