"""BallDontLie WNBA API client (Module 0).

Auth via raw `Authorization: <key>` header. Cursor pagination, rate-limit aware
(600 req/min observed; sleeps on 429 and when the remaining budget is low).
Data spans 2008-current. Spec: https://www.balldontlie.io/openapi/wnba.yml
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterator

import requests

from src.config import PROJECT_ROOT  # ensures .env is loaded
from src.utils import get_logger

log = get_logger(__name__)
BASE = "https://api.balldontlie.io/wnba/v1"


class BallDontLieError(RuntimeError):
    pass


class BallDontLieClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("BALLDONTLIE_API_KEY", "")
        if not self.api_key:
            raise BallDontLieError("Set BALLDONTLIE_API_KEY (env or .env).")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": self.api_key})

    # ---- low level ------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{BASE}/{path.lstrip('/')}"
        for attempt in range(6):
            r = self.session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                log.warning("429 rate-limited; sleeping %.1fs", wait)
                time.sleep(min(wait, 30))
                continue
            if r.status_code == 401:
                raise BallDontLieError("401 Unauthorized — bad BallDontLie key.")
            if r.status_code >= 400:
                raise BallDontLieError(f"{r.status_code} {path}: {r.text[:200]}")
            # be polite when the per-minute budget runs low
            rem = r.headers.get("x-ratelimit-remaining")
            if rem is not None and int(rem) < 20:
                time.sleep(2.0)
            return r.json()
        raise BallDontLieError(f"Repeatedly rate-limited on {path}")

    def paginate(self, path: str, params: dict[str, Any] | None = None,
                 per_page: int = 100, max_pages: int | None = None) -> Iterator[dict]:
        params = dict(params or {})
        params["per_page"] = per_page
        cursor, pages = None, 0
        while True:
            if cursor is not None:
                params["cursor"] = cursor
            payload = self._get(path, params)
            yield from payload.get("data", [])
            meta = payload.get("meta", {}) or {}
            cursor = meta.get("next_cursor")
            pages += 1
            if not cursor or (max_pages and pages >= max_pages):
                break

    # ---- domain endpoints ----------------------------------------------
    def get_wnba_teams(self) -> list[dict]:
        return self._get("teams").get("data", [])

    def get_wnba_players(self, per_page: int = 100) -> list[dict]:
        return list(self.paginate("players", per_page=per_page))

    def get_wnba_games(self, season: int, per_page: int = 100) -> list[dict]:
        return list(self.paginate("games", {"seasons[]": season}, per_page))

    def get_wnba_player_stats(self, season: int, game_ids: list[int] | None = None,
                              per_page: int = 100) -> Iterator[dict]:
        params: dict[str, Any] = {"seasons[]": season}
        if game_ids:
            params["game_ids[]"] = game_ids
        yield from self.paginate("player_stats", params, per_page)
