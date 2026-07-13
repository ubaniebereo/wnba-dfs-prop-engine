"""Shared helpers: logging, resilient HTTP, and box-score value parsing."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Iterator

import requests

from . import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a module logger configured once at the project log level."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(config.LOG_LEVEL)
        logger.propagate = False
    return logger


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
_session = requests.Session()
_session.headers.update(config.HTTP_HEADERS)


def http_get_json(url: str, params: dict[str, Any] | None = None) -> dict | None:
    """GET a URL and return parsed JSON, with retries and backoff.

    Returns None on persistent failure rather than raising, so a single bad
    game does not abort a long ingestion run.
    """
    last_err: Exception | None = None
    for attempt in range(1, config.HTTP_RETRIES + 1):
        try:
            resp = _session.get(url, params=params, timeout=config.HTTP_TIMEOUT)
            if resp.status_code == 200:
                time.sleep(config.REQUEST_PAUSE)
                return resp.json()
            log.warning("GET %s -> HTTP %s (attempt %d)", url, resp.status_code, attempt)
        except (requests.RequestException, ValueError) as exc:  # ValueError = bad JSON
            last_err = exc
            log.warning("GET %s failed: %s (attempt %d)", url, exc, attempt)
        time.sleep(config.HTTP_BACKOFF * attempt)
    if last_err:
        log.error("Giving up on %s: %s", url, last_err)
    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def to_int(value: Any, default: int | None = 0) -> int | None:
    """Best-effort int parse (handles '', None, '12', '12.0')."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def split_made_attempted(text: Any) -> tuple[int, int]:
    """Parse an ESPN 'made-attempted' string like '5-12' -> (5, 12)."""
    if not text or not isinstance(text, str) or "-" not in text:
        return 0, 0
    made, _, att = text.partition("-")
    return to_int(made) or 0, to_int(att) or 0


def parse_espn_date(value: str | None) -> str | None:
    """Convert an ESPN ISO timestamp ('2025-06-01T19:00Z') to 'YYYY-MM-DD'."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value[:10] if len(value) >= 10 else None


def daterange(start: date, end: date) -> Iterator[date]:
    """Yield each calendar date from start to end inclusive."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")
