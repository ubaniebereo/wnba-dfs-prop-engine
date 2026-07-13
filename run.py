#!/usr/bin/env python3
"""WNBA efficiency-cycle pipeline.

Steps (the chain you asked for):
  1. pull every active WNBA player + their per-game stats
  2. compute a per-game efficiency (Game Score) for each player
  3. detect efficiency DROPS vs each player's own baseline
  4. test whether drops recur on a 21-35 day cycle (Lomb-Scargle + permutation)
  5. FLAG players whose cycle is statistically real, and compare their
     predicted-dip games to their normal games
  6. (optional) scan player props for flagged players and rank UNDER candidates

Usage:
  export BALLDONTLIE_API_KEY=...        # required (paid tier for props/odds)
  ./.venv/bin/python run.py analyze --seasons 2024 2025
  ./.venv/bin/python run.py props   --seasons 2024 2025 --days-ahead 7

Add --refresh to bypass the local cache and re-pull from the API.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

from wnba_cycles.client import BDLClient, BDLError
from wnba_cycles.analysis import flag_player, MIN_GAMES, BAND_LO, BAND_HI
from wnba_cycles.props import scan_unders


def _group_stats_by_player(stats: list[dict]) -> dict[int, list[dict]]:
    by_pid: dict[int, list[dict]] = {}
    for s in stats:
        p = s.get("player") or {}
        pid = p.get("id") or s.get("player_id")
        if pid is None:
            continue
        by_pid.setdefault(int(pid), []).append(s)
    return by_pid


def _name_lookup(players: list[dict]) -> dict[int, str]:
    out = {}
    for p in players:
        out[int(p["id"])] = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
    return out


def cmd_analyze(args) -> list:
    client = BDLClient(use_cache=not args.refresh)
    print(f"Pulling active players...", file=sys.stderr)
    players = client.active_players(refresh=args.refresh)
    names = _name_lookup(players)
    print(f"  {len(players)} active players", file=sys.stderr)

    print(f"Pulling player stats for seasons {args.seasons}...", file=sys.stderr)
    stats = client.player_stats(seasons=args.seasons, refresh=args.refresh)
    print(f"  {len(stats)} stat lines", file=sys.stderr)

    by_pid = _group_stats_by_player(stats)
    results = []
    for pid, rows in by_pid.items():
        name = names.get(pid) or (rows[0].get("player") or {}).get("last_name", str(pid))
        r = flag_player(pid, name, rows, n_perm=args.perm)
        if r is not None:
            results.append(r)

    results.sort(key=lambda r: r.red_flag_score, reverse=True)
    _print_report(results, args)
    return results


def _print_report(results, args):
    flagged = [r for r in results if r.significant and r.red_flag_score > 0]
    print()
    print("=" * 78)
    print(f"ANALYZED {len(results)} players with >= {MIN_GAMES} games")
    print(f"FLAGGED  {len(flagged)} with a significant {int(BAND_LO)}-{int(BAND_HI)}"
          f"-day efficiency cycle (perm p<0.05)")
    print("=" * 78)
    hdr = (f"{'PLAYER':22} {'GMS':>3} {'PERIOD':>6} {'p':>6} "
           f"{'DIP_PTS':>7} {'AVG_PTS':>7} {'DIP_GmSc':>8} {'AVG_GmSc':>8} {'FLAG':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in flagged[:args.top]:
        print(f"{r.name[:22]:22} {r.n_games:3d} {r.best_period:6.1f} "
              f"{r.p_value:6.3f} {r.drop_mean_pts:7.1f} {r.base_mean_pts:7.1f} "
              f"{r.drop_mean_gmsc:8.1f} {r.base_mean_gmsc:8.1f} "
              f"{r.red_flag_score:6.2f}")
    if flagged:
        print()
        print("Next predicted dip dates (top flagged):")
        for r in flagged[:min(args.top, 10)]:
            print(f"  {r.name[:22]:22} {', '.join(r.next_dip_dates[:3])}")
    print()
    print("Read: DIP_PTS = avg points on predicted-dip games; AVG_PTS = other games.")
    print("A flag means the dip cycle is statistically real for THAT player — it is")
    print("not a guarantee any single game is under. Treat as one input, not gospel.")


def cmd_props(args):
    results = cmd_analyze(args)
    flagged = [r for r in results if r.significant and r.red_flag_score > 0]
    if not flagged:
        print("\nNo flagged players -> nothing to scan.", file=sys.stderr)
        return
    client = BDLClient(use_cache=not args.refresh)
    today = datetime.utcnow().date()
    end = today + timedelta(days=args.days_ahead)
    print(f"\nPulling upcoming games {today}..{end} for props scan...", file=sys.stderr)
    try:
        games = client.upcoming_games(start_date=str(today), end_date=str(end),
                                       seasons=args.seasons)
    except BDLError as e:
        print(f"Could not pull games: {e}", file=sys.stderr)
        return
    try:
        cands = scan_unders(client, flagged, games)
    except BDLError as e:
        print(f"Props unavailable ({e}). Your API tier may not include odds.",
              file=sys.stderr)
        return

    print("\n" + "=" * 78)
    print(f"UNDER CANDIDATES — flagged players in a predicted dip with a posted line")
    print("=" * 78)
    if not cands:
        print("None — no flagged player has an upcoming game in a dip window with props.")
        return
    hdr = (f"{'PLAYER':22} {'DATE':10} {'MARKET':14} {'LINE':>6} "
           f"{'DIP':>6} {'AVG':>6} {'EDGE':>6} {'FLAG':>5} BOOK")
    print(hdr)
    print("-" * len(hdr))
    for c in cands[:args.top]:
        print(f"{c.player[:22]:22} {c.game_date:10} {c.market[:14]:14} "
              f"{c.line:6.1f} {c.dip_avg:6.1f} {c.season_avg:6.1f} "
              f"{c.edge_vs_line:6.1f} {c.red_flag_score:5.2f} {c.book}")
    print("\nEDGE = predicted-dip average minus the posted line. More negative = the")
    print("model leans further UNDER. This is decision support, not a betting guarantee.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "props"):
        p = sub.add_parser(name)
        p.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025],
                       help="seasons to pull (more games = stronger test)")
        p.add_argument("--perm", type=int, default=2000,
                       help="permutations for significance test")
        p.add_argument("--top", type=int, default=40, help="rows to print")
        p.add_argument("--refresh", action="store_true", help="bypass cache")
        if name == "props":
            p.add_argument("--days-ahead", type=int, default=7,
                           help="how many days of upcoming games to scan")
    args = ap.parse_args()
    try:
        if args.cmd == "analyze":
            cmd_analyze(args)
        else:
            cmd_props(args)
    except BDLError as e:
        print(f"\nAPI error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
