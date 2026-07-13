#!/usr/bin/env python3
"""End-to-end cycle analysis on FREE real data (data.wnba.com, no API key).

  ./.venv/bin/python run_free.py --season 2025
  ./.venv/bin/python run_free.py --season 2025 --perm 3000 --top 30

Pulls a full season of real box scores, computes per-game Game Score, runs the
21-35 day Lomb-Scargle + permutation cycle test per player, and prints the
players whose efficiency dip cycle is statistically real, with their dip-vs-
normal comparison and next predicted dip dates.
"""

from __future__ import annotations

import argparse
import sys

from wnba_cycles.wnba_data import collect_season
from wnba_cycles.analysis import flag_player, MIN_GAMES, BAND_LO, BAND_HI


def _progress(done, total, rows):
    print(f"  fetched {done}/{total} games  ({rows} player-rows)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--min-games", type=int, default=MIN_GAMES)
    args = ap.parse_args()

    print(f"Pulling {args.season} real box scores from data.wnba.com ...",
          file=sys.stderr)
    rows, names = collect_season(args.season, refresh=args.refresh,
                                 progress=_progress)
    if not rows:
        print("No data returned (feed may be down or season has no games).",
              file=sys.stderr)
        sys.exit(1)
    print(f"  {len(rows)} player-game rows, {len(names)} players", file=sys.stderr)

    by_pid: dict[int, list[dict]] = {}
    for r in rows:
        pid = r["player"]["id"]
        if pid is not None:
            by_pid.setdefault(int(pid), []).append(r)

    results = []
    for pid, grows in by_pid.items():
        r = flag_player(pid, names.get(pid, str(pid)), grows, n_perm=args.perm)
        if r is not None:
            results.append(r)
    results.sort(key=lambda r: r.red_flag_score, reverse=True)

    flagged = [r for r in results if r.significant and r.red_flag_score > 0]
    print("\n" + "=" * 86)
    print(f"SEASON {args.season} | analyzed {len(results)} players with "
          f">={args.min_games} games | FLAGGED {len(flagged)} with a real "
          f"{int(BAND_LO)}-{int(BAND_HI)}d efficiency cycle (perm p<0.05)")
    print("=" * 86)
    hdr = (f"{'PLAYER':22} {'GMS':>3} {'PER':>5} {'p':>6} "
           f"{'DIP_PTS':>7} {'AVG_PTS':>7} {'DIP_GmSc':>8} {'AVG_GmSc':>8} {'FLAG':>6}")
    print(hdr); print("-" * len(hdr))
    for r in flagged[:args.top]:
        print(f"{r.name[:22]:22} {r.n_games:3d} {r.best_period:5.1f} {r.p_value:6.3f} "
              f"{r.drop_mean_pts:7.1f} {r.base_mean_pts:7.1f} "
              f"{r.drop_mean_gmsc:8.1f} {r.base_mean_gmsc:8.1f} {r.red_flag_score:6.2f}")
    if flagged:
        print("\nNext predicted dip windows (top flagged):")
        for r in flagged[:min(args.top, 12)]:
            print(f"  {r.name[:22]:22} {', '.join(r.next_dip_dates[:3])}")
    print("\nDIP_PTS = avg points on predicted-dip games; AVG_PTS = other games.")
    print("PER = detected cycle length (days). FLAG = red-flag score (size x significance).")
    print("A flag = the cycle is real in THIS player's season, not a guarantee any one")
    print("game goes under. Use as one signal; confirm against the posted line.")


if __name__ == "__main__":
    main()
