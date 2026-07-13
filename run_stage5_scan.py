#!/usr/bin/env python3
"""Stage 5 — live value scan: anchor + model + pairs -> ranked alerts.

  python run_stage5_scan.py

Pulls multi-book WNBA props (incl Pinnacle), builds sharp anchor fair values,
finds stale-line singles and correlation-mispriced pairs, ranks by a
CLV-predictive signal, and emits notification payloads. No picks/staking advice.
"""

from __future__ import annotations

import json
import os

import pandas as pd

import anchor_lines
import news_feeds
import pair_engine
import ranking
from src.database import get_engine

pd.set_option("display.width", 190, "display.max_columns", 40)
H = "=" * 90


def main():
    if not os.environ.get("ODDS_API_KEY"):
        print("ODDS_API_KEY not set."); return
    eng = get_engine()

    print(H + "\nSTAGE 5 LIVE SCAN\n" + H)
    props = anchor_lines.fetch_props_multi_region()
    print(f"props: {len(props)} rows, books={sorted(props.book.unique()) if not props.empty else []}")
    anchor = anchor_lines.build_anchor(props)
    print(f"anchored props: {len(anchor)} "
          f"(pinnacle={int((anchor.anchor_source=='pinnacle').sum()) if not anchor.empty else 0})")

    # --- singles: stale book lines vs sharp anchor ---
    single_edges = anchor_lines.line_shop_edges(props, anchor, min_edge=0.02)
    ranked_singles = ranking.rank_singles(single_edges)

    # --- pairs: correlation mispricing ---
    games = pair_engine._load()   # has player_name + team (stands in for proj)
    pairs = pair_engine.find_pairs(anchor, games, props, min_abs_mispricing=0.03)
    ranked_pairs = ranking.rank_pairs(pairs)

    # --- news bus (latency windows) ---
    bus = news_feeds.NewsBus()
    n_news = news_feeds.poll_espn_injuries(bus)

    print("\n" + H + "\nRANKED SINGLE ALERTS (book price beats sharp fair)\n" + H)
    if ranked_singles.empty:
        print("  none above threshold — market efficient right now (expected in steady state).")
    else:
        print(ranked_singles[["player", "market", "side", "line", "book", "odds",
                              "anchor_fair", "anchor_edge", "score", "anchor_source"]].to_string(index=False))

    print("\n" + H + "\nRANKED PAIR ALERTS (correlation mispricing vs independence)\n" + H)
    if ranked_pairs.empty:
        print("  none above threshold.")
    else:
        print(ranked_pairs[["team", "player_a", "player_b", "rho", "p_independent",
                            "p_joint", "mispricing", "direction"]].head(12).to_string(index=False))

    alerts = ranking.build_alerts(ranked_singles, ranked_pairs)
    print(f"\n{H}\nNOTIFICATION PAYLOADS ({len(alerts)}) + news events ({n_news})\n{H}")
    for a in alerts[:6]:
        print("  " + json.dumps(a))
    # persist for the diagnostics/CLV forward sample
    if alerts:
        pd.DataFrame(alerts).to_json(eng.url.database.replace(".sqlite", "_alerts.jsonl"),
                                     orient="records", lines=True)


if __name__ == "__main__":
    main()
