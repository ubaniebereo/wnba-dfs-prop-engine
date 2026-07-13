"""Minute-cap detection from injury/news text (Stage 3, Section 2).

Pure keyword/phrase matching over injury blurbs -> minutes_cap_flag (+ rough
bucket). No network here; injury_notes_etl.py supplies the text.
"""

from __future__ import annotations

import re

CAP_PHRASES = [
    "minutes restriction", "minute restriction", "limited minutes", "pitch count",
    "on a minutes", "play limited", "minutes limit", "minutes cap", "ease back",
    "ramp up", "load management", "rest", "managed minutes",
]
BUCKET_RX = re.compile(r"(\d{1,2})\s*[-to]+\s*(\d{1,2})\s*minute", re.I)


def detect_minutes_cap(text: str) -> dict:
    t = (text or "").lower()
    flag = int(any(p in t for p in CAP_PHRASES))
    bucket = None
    m = BUCKET_RX.search(t)
    if m:
        bucket = (int(m.group(1)) + int(m.group(2))) / 2.0
    elif flag:
        bucket = 22.0                            # default soft cap when phrased vaguely
    return {"minutes_cap_flag": flag, "minutes_cap_bucket": bucket}
