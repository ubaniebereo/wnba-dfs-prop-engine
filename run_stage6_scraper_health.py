#!/usr/bin/env python3
"""Stage 6 — scraper/feed health (Sec 9D). Honest viability per source.

  python run_stage6_scraper_health.py
"""

from __future__ import annotations

import time

import pandas as pd

from feeds import espn_client, odds_api_client, rotowire_scraper
from feeds.normalizers import PlayerResolver
from feeds.x_client import XClient

pd.set_option("display.width", 160)


def _timed(fn):
    t0 = time.time()
    try:
        out = fn()
        return {"ok": True, "result": out, "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:120], "elapsed_s": round(time.time() - t0, 1)}


def main():
    resolver = PlayerResolver()
    rows = []

    x = XClient()
    r = _timed(lambda: x.fetch_recent_posts(max_results=10))
    rows.append({"source": "x_api", "viable": x.viable, "ok": r["ok"],
                 "rows": len(r.get("result", []) or []), "elapsed_s": r["elapsed_s"],
                 "note": "official API; recent-search 450/15min"})

    r = _timed(lambda: odds_api_client.snapshot_props())
    rows.append({"source": "the_odds_api", "viable": True, "ok": r["ok"],
                 "rows": len(r.get("result", []) or []), "elapsed_s": r["elapsed_s"],
                 "note": "official API; live + Pinnacle"})

    r = _timed(lambda: espn_client.confirmed_lineups(resolver))
    rows.append({"source": "espn", "viable": True, "ok": r["ok"],
                 "rows": len(r.get("result", []) or []), "elapsed_s": r["elapsed_s"],
                 "note": "official API; confirmed starters"})

    rw = rotowire_scraper.RotowireScraper()
    probe = rw.probe()
    rows.append({"source": "rotowire", "viable": bool(probe.get("viable")), "ok": True,
                 "rows": 0, "elapsed_s": None,
                 "note": probe.get("recommendation", probe.get("url", "ok"))[:60]})

    from feeds.sgp_scraper import viability
    v = viability()
    rows.append({"source": "sgp_builder", "viable": v["viable_automated"], "ok": True,
                 "rows": 0, "elapsed_s": None, "note": v["reason"][:60]})

    df = pd.DataFrame(rows)
    print("=" * 92)
    print("STAGE 6 SCRAPER / FEED HEALTH")
    print("=" * 92)
    print(df.to_string(index=False))
    print(f"\nunresolved player-name mappings this run: {len(resolver.unresolved)}")
    print("Viable continuous stack: X API + The Odds API + ESPN (all official). "
          "Rotowire/SGP = research-only.")


if __name__ == "__main__":
    main()
