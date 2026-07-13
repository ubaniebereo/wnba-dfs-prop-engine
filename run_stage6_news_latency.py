#!/usr/bin/env python3
"""Stage 6 — news latency study (Sec 9A).

Ingests X news + a timestamped odds snapshot, then (over repeated runs) detects
line moves and measures the lag between a news event and book/anchor movement.
A single run seeds the time series; lag metrics accrue as the scheduler captures
more snapshots around lineup news.

  python run_stage6_news_latency.py            # one capture + current metrics
"""

from __future__ import annotations

import pandas as pd

from feeds import diff_engine, odds_api_client
from feeds.normalizers import PlayerResolver
from feeds.storage import init_storage, insert, read
from feeds.x_client import XClient

pd.set_option("display.width", 170)


def capture_cycle() -> dict:
    init_storage()
    resolver = PlayerResolver()
    x = XClient()
    posts = x.fetch_recent_posts(max_results=25)
    news = x.to_news_events(posts, resolver)
    n_news = insert("news_events", news)
    n_odds = odds_api_client.capture_snapshot()
    return {"x_posts": len(posts), "news_events_stored": n_news,
            "odds_rows": n_odds, "unresolved_names": len(resolver.unresolved)}


def main():
    print("=" * 84)
    print("STAGE 6 NEWS-LATENCY STUDY")
    print("=" * 84)
    cyc = capture_cycle()
    print("capture cycle:", cyc)

    n_moves = diff_engine.detect_moves()
    snaps = read("SELECT COUNT(DISTINCT captured_at) n FROM odds_snapshots")["n"].iloc[0]
    print(f"\nodds snapshots in archive: {int(snaps)} | line moves detected: {n_moves}")

    news = read("SELECT event_type, parsed_status, player_name, raw_text FROM news_events "
                "ORDER BY captured_at DESC LIMIT 8")
    if not news.empty:
        print("\nRecent parsed news events:")
        print(news.to_string(index=False, max_colwidth=70))

    metrics = diff_engine.latency_metrics()
    print("\nlatency metrics (anchor -> soft-book lag):", metrics)
    if int(snaps) < 2:
        print("\nNeed >=2 snapshots over time for lag measurement. Run the scheduler")
        print("(feeds/scheduler.py) across a lineup-news window to accumulate the series.")


if __name__ == "__main__":
    main()
