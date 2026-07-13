#!/usr/bin/env python3
"""ONE-COMMAND launcher — scans, serves the dashboard, opens your browser.

    ./.venv/bin/python start.py

It runs a first scan, then keeps scanning every 2 minutes IN THE BACKGROUND,
starts the dashboard, and opens http://127.0.0.1:8050 in your browser
automatically. Keep this window open; press Ctrl-C to stop everything.

You do NOT need a second terminal. You do NOT need to open anything yourself.
"""

from __future__ import annotations

import threading
import webbrowser

from apscheduler.schedulers.background import BackgroundScheduler

from dashboard_app import app
from scanner.run_live_scan import run_scan

URL = "http://127.0.0.1:8050"


def main():
    print("=" * 60)
    print("  WNBA PROP ENGINE  —  starting up")
    print("=" * 60)
    print("\n[1/2] Running the first scan (the first one takes ~10-15s)...\n")
    try:
        print("      scan result:", run_scan())
    except Exception as exc:  # noqa: BLE001
        print("      (first scan had an issue:", exc, "— dashboard still opens)")

    # keep the board fresh automatically, in the background (no 2nd window needed)
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(run_scan, "interval", seconds=120, max_instances=1, coalesce=True)
    sched.start()

    print("\n[2/2] Opening the dashboard in your browser...")
    print(f"      >>>  {URL}  <<<")
    print("\n  KEEP THIS WINDOW OPEN. Press Ctrl-C here to stop.\n")
    threading.Timer(2.0, lambda: webbrowser.open(URL)).start()
    app.run(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()
