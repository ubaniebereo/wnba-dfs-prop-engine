"""X / Twitter news feed (Stage 6, Sec 5A) — VERIFIED working token.

Uses the v2 recent-search endpoint (rate limit ~450/15min) to pull beat-writer /
team posts and a regex parser to extract status changes (OUT, questionable,
starting, minutes restriction). Outputs normalized news_events rows.
"""

from __future__ import annotations

import os
import re

from .base import BaseClient
from .normalizers import PlayerResolver, norm_name
from .storage import now_iso
from src.utils import get_logger

log = get_logger(__name__)
SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

# Default beat-writer / aggregator handles (extend per team).
DEFAULT_HANDLES = ["Lynx", "nyliberty", "LVAces", "chicagosky", "IndianaFever",
                   "UnderdogWNBA", "WNBAInjuryRep", "Rotowire"]

STATUS_RX = [
    (r"\b(out|ruled out|will not play|inactive)\b", "scratch", "out", 0.9),
    (r"\b(doubtful)\b", "status", "doubtful", 0.8),
    (r"\b(questionable|game[- ]time decision|gtd)\b", "status", "questionable", 0.75),
    (r"\b(probable|available|active|cleared)\b", "status", "available", 0.7),
    (r"\b(start(ing)?|in the starting (lineup|five))\b", "starter_change", "starting", 0.7),
    (r"\b(minutes restriction|limited minutes|minutes limit|workload)\b",
     "minutes_cap", "minutes_cap", 0.8),
]


class XClient(BaseClient):
    name = "x"

    def __init__(self):
        super().__init__(min_interval=2.0, timeout=25.0)
        self.token = os.environ.get("X_BEARER_TOKEN", "")
        self.viable = bool(self.token)

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def fetch_recent_posts(self, handles=None, keywords=None, max_results=25,
                           since_id=None) -> list[dict]:
        handles = handles or DEFAULT_HANDLES
        kw = keywords or ["out", "questionable", "doubtful", "starting",
                          "minutes", "active", "inactive"]
        frm = " OR ".join(f"from:{h}" for h in handles)
        query = f"({frm}) ({' OR '.join(kw)}) -is:retweet"
        params = {"query": query, "max_results": min(max(max_results, 10), 100),
                  "tweet.fields": "created_at,author_id"}
        if since_id:
            params["since_id"] = since_id
        res = self.get(SEARCH_URL, headers=self._headers(), params=params)
        if not res.ok:
            log.warning("x search failed: %s %s", res.status, res.error)
            return []
        return (res.data or {}).get("data", []) or []

    @staticmethod
    def parse_news_post(text: str) -> dict | None:
        t = text.lower()
        for rx, etype, status, conf in STATUS_RX:
            if re.search(rx, t):
                return {"event_type": etype, "parsed_status": status, "confidence": conf}
        return None

    def to_news_events(self, posts: list[dict], resolver: PlayerResolver) -> list[dict]:
        rows = []
        for p in posts:
            parsed = self.parse_news_post(p.get("text", ""))
            if not parsed:
                continue
            # naive player extraction: match any known name appearing in the text
            pid = pname = None
            tn = norm_name(p.get("text", ""))
            for key in resolver.keys:
                if key and key in tn and len(key.split()) >= 2:
                    r = resolver.by_key.loc[key]
                    pid, pname = r["player_id"], r["player_name"]; break
            rows.append({"source": "x", "captured_at": p.get("created_at", now_iso()),
                         "event_type": parsed["event_type"], "player_id": pid,
                         "player_name": pname, "raw_text": p.get("text", "")[:280],
                         "parsed_status": parsed["parsed_status"],
                         "confidence": parsed["confidence"],
                         "metadata_json": {"tweet_id": p.get("id"),
                                           "author_id": p.get("author_id")}})
        return rows
