"""News latency & fast feeds (Stage 5, Section 3).

A NewsEvent bus that triggers re-pricing of affected props in the latency window
before books fully move. ESPN confirmed rosters (verifiable, JSON API) are the
reliable source; Rotowire HTML and X/Twitter are best-effort, rate-limited
extensions (require their own access/tokens) and fail soft.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import espn_client
from minutes_caps_nlp import detect_minutes_cap
from src.utils import get_logger, http_get_json

log = get_logger(__name__)
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries"


@dataclass
class NewsEvent:
    player_name: str
    team: str | None
    type: str               # "scratch" | "minutes_cap" | "starter_change" | "status"
    status: str
    source: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class NewsBus:
    def __init__(self):
        self.events: list[NewsEvent] = []
        self._subs = []

    def subscribe(self, fn):
        self._subs.append(fn)

    def publish(self, ev: NewsEvent):
        self.events.append(ev)
        log.info("NEWS %s | %s | %s (%s)", ev.type, ev.player_name, ev.status, ev.source)
        for fn in self._subs:
            try:
                fn(ev)
            except Exception as exc:
                log.warning("subscriber error: %s", exc)


def poll_espn_injuries(bus: NewsBus) -> int:
    """Confirmed-status source: emit NewsEvents for OUT / minute-cap players."""
    payload = http_get_json(ESPN_INJURIES) or {}
    n = 0
    for team in payload.get("injuries", []) or []:
        for inj in team.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            status = (inj.get("status") or "").strip()
            txt = " ".join(filter(None, [inj.get("shortComment"), inj.get("longComment")]))
            cap = detect_minutes_cap(txt)
            etype = ("scratch" if status.lower() in ("out", "injured reserve")
                     else "minutes_cap" if cap["minutes_cap_flag"] else "status")
            if etype in ("scratch", "minutes_cap"):
                bus.publish(NewsEvent(ath.get("displayName"), None, etype, status, "espn"))
                n += 1
    return n


def confirmed_starters(event_id: str) -> list[dict]:
    """Confirmation near tip via ESPN game roster (is_starter / is_active)."""
    return espn_client.get_game_roster(event_id)


# --- best-effort extensions (require their own access; fail soft) ----------
def poll_rotowire(bus: NewsBus) -> int:
    """Placeholder: parse rotowire.com/wnba lineups+injuries (HTML). Rate-limit
    and respect ToS. Left unimplemented to avoid brittle scraping in core."""
    log.info("rotowire poll skipped (enable with HTML parser + low rate)")
    return 0


def poll_twitter(bus: NewsBus, handles: list[str]) -> int:
    """Placeholder: X API for beat-writer scratch/minute-cap tweets. Needs token."""
    log.info("twitter poll skipped (needs X API token)")
    return 0


def evaluate_news(bus: NewsBus) -> dict:
    return {"events": len(bus.events),
            "by_type": {t: sum(e.type == t for e in bus.events)
                        for t in {e.type for e in bus.events}}}
