#!/usr/bin/env python3
"""Stage 6 — anchor vs soft-book study (Sec 9C).

How often, and by how much, do soft books' prop prices deviate from the sharp
Pinnacle/consensus anchor right now? This quantifies the line-shopping edge
surface (the 'stale number' frequency by book).

  python run_stage6_anchor_vs_softbooks.py
"""

from __future__ import annotations

import pandas as pd

import anchor_lines
from propmodel.devig import implied

pd.set_option("display.width", 170)


def main():
    props = anchor_lines.fetch_props_multi_region()
    if props.empty:
        print("No live props posted."); return
    anchor = anchor_lines.build_anchor(props)
    a = anchor.set_index(["player", "market", "line"])

    rows = []
    for r in props.itertuples():
        if r.book == "pinnacle":
            continue
        key = (r.player, r.market, float(r.line))
        if key not in a.index:
            continue
        fair = a.loc[key, "anchor_fair_over"] if r.side == "over" else a.loc[key, "anchor_fair_under"]
        rows.append({"book": r.book, "market": r.market,
                     "edge_vs_anchor": float(fair) - implied(r.odds)})
    d = pd.DataFrame(rows)
    if d.empty:
        print("No comparable props."); return

    print("=" * 80)
    print("ANCHOR vs SOFT-BOOK — stale-number frequency & magnitude (live snapshot)")
    print("=" * 80)
    by_book = d.groupby("book").agg(
        n=("edge_vs_anchor", "size"),
        share_stale_2pct=("edge_vs_anchor", lambda s: round((s >= 0.02).mean(), 3)),
        share_stale_3pct=("edge_vs_anchor", lambda s: round((s >= 0.03).mean(), 3)),
        max_edge=("edge_vs_anchor", lambda s: round(s.max(), 3)),
        mean_abs_dev=("edge_vs_anchor", lambda s: round(s.abs().mean(), 3))).reset_index()
    print(by_book.to_string(index=False))
    print(f"\nTotal comparable book prices: {len(d)} | anchored props: {len(anchor)}")
    print("Read: share_stale = fraction where the book's price beats the sharp fair by >=X.")
    print("Low fractions => market efficient now; real stale windows open after NEWS")
    print("(run the scheduler + diff_engine around lineup news to catch them).")


if __name__ == "__main__":
    main()
