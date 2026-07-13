#!/usr/bin/env python3
"""Run the prop scanner: once (default) or on a loop.

  python run_daily_scan.py            # single scan, write current_prop_scan
  python run_daily_scan.py --loop     # APScheduler loop (every 120s), never
                                      # depends on a browser being open
  python run_daily_scan.py --loop --interval 60
"""

from __future__ import annotations

import argparse

from scanner.run_live_scan import run_loop, run_scan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    if args.loop:
        run_loop(args.interval)
    else:
        print(run_scan())


if __name__ == "__main__":
    main()
