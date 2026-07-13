"""Source health snapshot -> source_health table (read by the dashboard tab)."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from feeds.storage import ENGINE, init_storage
from src.utils import get_logger

log = get_logger(__name__)


def _timed(fn):
    t0 = time.time()
    try:
        out = fn()
        return True, out, round(time.time() - t0, 1)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:80], round(time.time() - t0, 1)


def snapshot() -> pd.DataFrame:
    from feeds import espn_client, odds_api_client, rotowire_scraper
    from feeds.normalizers import PlayerResolver
    from feeds.prizepicks_client import PrizePicksClient
    from feeds.sgp_scraper import viability
    from feeds.x_client import XClient

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []

    ok, res, el = _timed(lambda: XClient().fetch_recent_posts(max_results=10))
    rows.append({"source": "X / Twitter", "tier": "official API", "viable": ok,
                 "rows": len(res) if ok else 0, "latency_s": el, "note": "news monitoring"})

    ok, res, el = _timed(lambda: odds_api_client.snapshot_props())
    rows.append({"source": "The Odds API", "tier": "official API", "viable": ok,
                 "rows": len(res) if ok else 0, "latency_s": el,
                 "note": "sportsbook props + Pinnacle anchor"})

    ok, res, el = _timed(lambda: PrizePicksClient().fetch_wnba(include_alt=True))
    rows.append({"source": "PrizePicks", "tier": "public JSON (curl_cffi)", "viable": ok,
                 "rows": (len(res) if hasattr(res, "__len__") else 0) if ok else 0,
                 "latency_s": el, "note": "DFS lines (execution venue)"})

    ok, res, el = _timed(lambda: espn_client.confirmed_lineups(PlayerResolver()))
    rows.append({"source": "ESPN", "tier": "official API", "viable": ok,
                 "rows": len(res) if ok else 0, "latency_s": el,
                 "note": "confirmed starters / injuries"})

    probe = rotowire_scraper.RotowireScraper().probe()
    rows.append({"source": "Rotowire", "tier": "HTML scrape", "viable": bool(probe.get("viable")),
                 "rows": 0, "latency_s": None,
                 "note": probe.get("recommendation", "ok")[:60]})

    rows.append({"source": "SGP builders", "tier": "browser", "viable": viability()["viable_automated"],
                 "rows": 0, "latency_s": None, "note": "research-only (anti-bot)"})

    df = pd.DataFrame(rows)
    df["checked_at"] = ts
    init_storage()
    df.to_sql("source_health", ENGINE, if_exists="replace", index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    print(snapshot().to_string(index=False))
