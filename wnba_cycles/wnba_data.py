"""Free, keyless WNBA box-score source: data.wnba.com.

The league's `stats.wnba.com` API is Akamai bot-protected and hangs for plain
HTTP clients. But the older `data.wnba.com` JSON feed is open and returns REAL
integer box scores. This module pulls the schedule + per-game detail and
normalizes each player's line into the dict shape `analysis.build_series`
expects, i.e. keys: player{id}, game{date}, min, pts, fgm, fga, ftm, fta,
oreb, dreb, reb, ast, stl, blk, turnover, pf  (+ name).

Endpoints:
  schedule    : .../wnba/{year}/league/10_full_schedule.json
  game detail : .../wnba/{year}/scores/gamedetail/{gid}_gamedetail.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "https://data.wnba.com/data/v2015/json/mobile_teams/wnba"
UA = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/124.0 Safari/537.36")}
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "wnba"
CACHE.mkdir(parents=True, exist_ok=True)

# season type code in gid: digit 3 of gid -> '1' preseason, '2' regular, '4' playoffs
# (gid like '1022500041' -> the '2' after leading '10' marks regular season)


def _get_json(url: str, cache_name: str, *, refresh: bool = False,
              retries: int = 3) -> dict | None:
    cp = CACHE / cache_name
    if cp.exists() and not refresh:
        try:
            return json.loads(cp.read_text())
        except json.JSONDecodeError:
            pass
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200 and r.content[:1] in (b"{", b"["):
                cp.write_text(r.text)
                return r.json()
            return None
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_schedule(year: int, *, refresh: bool = False) -> list[dict]:
    url = f"{BASE}/{year}/league/10_full_schedule.json"
    data = _get_json(url, f"schedule_{year}.json", refresh=refresh)
    if not data:
        return []
    games = []
    for mon in data.get("lscd", []):
        for g in mon.get("mscd", {}).get("g", []):
            games.append(g)
    return games


def _is_regular_season(gid: str) -> bool:
    # gid: '10' + season-type digit + 2-digit year + sequence. type '2' = regular
    return len(gid) >= 3 and gid[2] == "2"


def fetch_game_players(year: int, gid: str, gdate: str,
                       *, refresh: bool = False) -> list[dict]:
    url = f"{BASE}/{year}/scores/gamedetail/{gid}_gamedetail.json"
    data = _get_json(url, f"gd_{gid}.json", refresh=refresh)
    if not data:
        return []
    g = data.get("g", {})
    date = g.get("gdte") or gdate
    rows = []
    for side in ("vls", "hls"):
        team = g.get(side, {}) or {}
        tcode = team.get("ta", "")
        for p in team.get("pstsg", []) or []:
            rows.append(_normalize(p, date, tcode, gid))
    return rows


def _normalize(p: dict, date: str, team: str, gid: str) -> dict:
    def n(*keys):
        for k in keys:
            v = p.get(k)
            if v not in (None, ""):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0
    pid = p.get("pid")
    name = f"{p.get('fn','').strip()} {p.get('ln','').strip()}".strip()
    return {
        "player": {"id": pid},
        "player_name": name,
        "team": team,
        "game": {"id": gid, "date": date},
        "min": n("min"),
        "pts": n("pts"), "fgm": n("fgm"), "fga": n("fga"),
        "ftm": n("ftm"), "fta": n("fta"),
        "oreb": n("oreb"), "dreb": n("dreb"), "reb": n("reb"),
        "ast": n("ast"), "stl": n("stl"), "blk": n("blk"),
        "turnover": n("tov"), "pf": n("pf"),
    }


def collect_season(year: int, *, refresh: bool = False, regular_only: bool = True,
                   progress=None) -> tuple[list[dict], dict[int, str]]:
    """Return (all player-game rows, {player_id: name}) for a season."""
    sched = fetch_schedule(year, refresh=refresh)
    games = [g for g in sched if g.get("stt") == "Final"
             and (not regular_only or _is_regular_season(g.get("gid", "")))]
    rows: list[dict] = []
    names: dict[int, str] = {}
    for i, g in enumerate(games):
        gid, gdate = g.get("gid"), g.get("gdte", "")
        for r in fetch_game_players(year, gid, gdate, refresh=refresh):
            rows.append(r)
            if r["player"]["id"] is not None:
                names[int(r["player"]["id"])] = r["player_name"]
        if progress and (i % 25 == 0 or i == len(games) - 1):
            progress(i + 1, len(games), len(rows))
    return rows, names
