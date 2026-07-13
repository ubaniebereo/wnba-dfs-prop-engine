#!/usr/bin/env python3
"""Cross-season replication: the honest test for a REAL efficiency cycle.

Single-season p-values are contaminated by multiple comparisons (test ~150
players, ~7 will look 'significant' at p<0.05 by chance alone). Two defenses:

  1. Benjamini-Hochberg FDR within each season (controls the false-discovery
     rate across all players).
  2. REPLICATION: a player flagged in BOTH seasons with a consistent cycle
     length is the real prize -- the chance of that happening twice by luck,
     for the same player at the same period, is small.

Usage:
  ./.venv/bin/python cross_season.py --seasons 2024 2025 --perm 1500
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from wnba_cycles.wnba_data import collect_season
from wnba_cycles.analysis import flag_player, MIN_GAMES


def bh_fdr(pvals, q=0.10):
    """Return boolean mask of discoveries under Benjamini-Hochberg at level q."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, n + 1) / n) * q
    passed = p[order] <= thresh
    if not passed.any():
        return np.zeros(n, bool), None
    kmax = np.max(np.where(passed)[0])
    cut = p[order][kmax]
    return p <= cut, float(cut)


def run_season(year, perm):
    print(f"\n=== Season {year}: pulling real box scores ===", file=sys.stderr)
    rows, names = collect_season(year, progress=lambda d, t, r: (
        print(f"  {d}/{t} games", file=sys.stderr) if d % 50 == 0 or d == t else None))
    by_pid = {}
    for r in rows:
        pid = r["player"]["id"]
        if pid is not None:
            by_pid.setdefault(int(pid), []).append(r)
    res = {}
    for pid, grows in by_pid.items():
        r = flag_player(pid, names.get(pid, str(pid)), grows, n_perm=perm)
        if r is not None:
            res[pid] = r
    return res, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs=2, default=[2024, 2025])
    ap.add_argument("--perm", type=int, default=1500)
    ap.add_argument("--period-tol", type=float, default=6.0,
                    help="max |period_a - period_b| days to count as consistent")
    args = ap.parse_args()
    a, b = args.seasons

    res_a, _ = run_season(a, args.perm)
    res_b, names_b = run_season(b, args.perm)

    # FDR within each season
    for yr, res in ((a, res_a), (b, res_b)):
        pids = list(res)
        disc, cut = bh_fdr([res[p].p_value for p in pids], q=0.10)
        n_disc = int(disc.sum())
        n_raw = sum(res[p].p_value < 0.05 for p in pids)
        print(f"\nSeason {yr}: {len(pids)} players tested | "
              f"{n_raw} raw p<0.05 (~{0.05*len(pids):.1f} expected by chance) | "
              f"{n_disc} survive BH-FDR@10% "
              f"(cutoff p<={cut:.4f})" if cut else
              f"\nSeason {yr}: {len(pids)} players | {n_raw} raw p<0.05 "
              f"(~{0.05*len(pids):.1f} by chance) | 0 survive BH-FDR@10%")

    # Replication across both seasons
    print("\n" + "=" * 90)
    print(f"REPLICATORS — significant (p<0.05) in BOTH {a} and {b}, "
          f"with cycle length consistent to <= {args.period_tol:.0f} days")
    print("=" * 90)
    reps = []
    for pid, ra in res_a.items():
        rb = res_b.get(pid)
        if rb is None:
            continue
        if ra.p_value < 0.05 and rb.p_value < 0.05 \
                and abs(ra.best_period - rb.best_period) <= args.period_tol:
            reps.append((pid, ra, rb))
    reps.sort(key=lambda t: t[1].p_value * t[2].p_value)
    if not reps:
        print("\nNONE. No player shows a consistent 21-35d cycle in both seasons.")
        print("Honest read: there is no robust, repeatable league-wide signal here.")
        print("The single-season flags are most likely multiple-comparison noise.")
    else:
        hdr = (f"{'PLAYER':22} {f'PER_{a}':>7} {f'p_{a}':>7} "
               f"{f'PER_{b}':>7} {f'p_{b}':>7} {'DIP_PTS':>7} {'AVG_PTS':>7}")
        print(hdr); print("-" * len(hdr))
        for pid, ra, rb in reps:
            print(f"{names_b.get(pid, ra.name)[:22]:22} "
                  f"{ra.best_period:7.1f} {ra.p_value:7.3f} "
                  f"{rb.best_period:7.1f} {rb.p_value:7.3f} "
                  f"{rb.drop_mean_pts:7.1f} {rb.base_mean_pts:7.1f}")
        print("\nThese are the only names with a repeatable pattern. Even so, treat as a")
        print("prior to combine with matchup/injury/line value -- not a standalone bet.")


if __name__ == "__main__":
    main()
