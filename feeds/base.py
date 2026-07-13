"""Base feed client: retry, timeout, rate-limit, jitter, honest viability flags.

Every source client subclasses BaseClient so scheduling, backoff and logging are
uniform. Sources that can't be reached reliably are marked NOT_VIABLE rather than
retried forever.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import httpx

from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class FetchResult:
    ok: bool
    status: int
    data: object = None
    error: str | None = None
    elapsed: float = 0.0


class BaseClient:
    name = "base"
    default_headers: dict = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    }

    def __init__(self, min_interval=0.0, timeout=30.0, retries=3):
        self.min_interval = min_interval        # per-source rate limit (s)
        self.timeout = timeout
        self.retries = retries
        self._last = 0.0
        self.viable: bool | None = None         # set by health checks

    def _throttle(self):
        gap = time.time() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap + random.uniform(0, 0.3))  # jitter
        self._last = time.time()

    def get(self, url, *, headers=None, params=None, as_json=True) -> FetchResult:
        h = {**self.default_headers, **(headers or {})}
        for attempt in range(1, self.retries + 1):
            self._throttle()
            t0 = time.time()
            try:
                r = httpx.get(url, headers=h, params=params, timeout=self.timeout,
                              follow_redirects=True)
                el = time.time() - t0
                if r.status_code == 429:
                    wait = float(r.headers.get("retry-after", 2 ** attempt))
                    log.warning("%s 429; backoff %.1fs", self.name, wait)
                    time.sleep(min(wait, 30)); continue
                if r.status_code >= 400:
                    return FetchResult(False, r.status_code, error=r.text[:160], elapsed=el)
                return FetchResult(True, r.status_code,
                                   data=(r.json() if as_json else r.text), elapsed=el)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s fetch error (try %d): %s", self.name, attempt, exc)
                time.sleep(1.0 * attempt)
        return FetchResult(False, 0, error="exhausted retries")
