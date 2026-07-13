"""Thin client for the balldontlie WNBA API.

All endpoints require an API key passed via the `Authorization` header.
Set it in the BALLDONTLIE_API_KEY environment variable.

Docs / spec: https://www.balldontlie.io/openapi/wnba.yml
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Any, Iterator

import requests

BASE_URL = "https://api.balldontlie.io/wnba/v1"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


class BDLError(RuntimeError):
    pass


class BDLClient:
    def __init__(self, api_key: str | None = None, *, use_cache: bool = True):
        self.api_key = api_key or os.environ.get("BALLDONTLIE_API_KEY", "")
        if not self.api_key:
            raise BDLError(
                "No API key. Set BALLDONTLIE_API_KEY env var "
                "(get one at https://www.balldontlie.io)."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": self.api_key})
        self.use_cache = use_cache
        CACHE_DIR.mkdir(exist_ok=True)

    # ---- low level -------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        params = params or {}
        for attempt in range(6):
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:  # rate limited
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 30))
                continue
            if resp.status_code == 401:
                raise BDLError("401 Unauthorized — bad API key.")
            if resp.status_code == 403:
                raise BDLError(
                    f"403 Forbidden on {path} — your tier likely doesn't "
                    "include this endpoint (odds/props need ALL-STAR or GOAT)."
                )
            if resp.status_code >= 400:
                raise BDLError(f"{resp.status_code} on {path}: {resp.text[:200]}")
            return resp.json()
        raise BDLError(f"Repeatedly rate-limited on {path}")

    def paginate(self, path: str, params: dict[str, Any] | None = None,
                 *, per_page: int = 100, max_pages: int | None = None) -> Iterator[dict]:
        """Yield every record across cursor-paginated pages."""
        params = dict(params or {})
        params["per_page"] = per_page
        cursor = None
        pages = 0
        while True:
            if cursor is not None:
                params["cursor"] = cursor
            payload = self._get(path, params)
            for row in payload.get("data", []):
                yield row
            meta = payload.get("meta", {}) or {}
            cursor = meta.get("next_cursor")
            pages += 1
            if not cursor or (max_pages and pages >= max_pages):
                break

    # ---- cached bulk pulls ----------------------------------------------
    def _cache_path(self, name: str) -> Path:
        return CACHE_DIR / f"{name}.json"

    def cached_collect(self, name: str, path: str,
                       params: dict[str, Any] | None = None,
                       *, refresh: bool = False, **kw) -> list[dict]:
        cp = self._cache_path(name)
        if self.use_cache and cp.exists() and not refresh:
            return json.loads(cp.read_text())
        rows = list(self.paginate(path, params, **kw))
        cp.write_text(json.dumps(rows))
        return rows

    # ---- domain helpers --------------------------------------------------
    def active_players(self, refresh: bool = False) -> list[dict]:
        return self.cached_collect("players_active", "players/active", refresh=refresh)

    def player_stats(self, *, seasons: list[int], player_ids: list[int] | None = None,
                     refresh: bool = False) -> list[dict]:
        params: dict[str, Any] = {"seasons[]": seasons}
        if player_ids:
            params["player_ids[]"] = player_ids
        tag = "stats_" + "_".join(map(str, seasons))
        if player_ids:
            tag += f"_p{len(player_ids)}"
        return self.cached_collect(tag, "player_stats", params, refresh=refresh)

    def games(self, *, seasons: list[int], refresh: bool = False) -> list[dict]:
        params = {"seasons[]": seasons}
        tag = "games_" + "_".join(map(str, seasons))
        return self.cached_collect(tag, "games", params, refresh=refresh)

    def player_props(self, game_id: int) -> list[dict]:
        # props are returned in a single response, not paginated
        payload = self._get("odds/player_props", {"game_id": game_id})
        return payload.get("data", payload) if isinstance(payload, dict) else payload

    def upcoming_games(self, *, start_date: str, end_date: str,
                       seasons: list[int]) -> list[dict]:
        params = {"start_date": start_date, "end_date": end_date, "seasons[]": seasons}
        return list(self.paginate("games", params))
