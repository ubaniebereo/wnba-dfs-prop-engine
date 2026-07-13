"""SGP / pair price capture (Stage 6, Sec 5D) — honest viability adapter.

Goal: capture real same-game-parlay prices to test whether books price near
independence. Reality: US books gate SGP builders behind authenticated, JS-heavy,
anti-bot-protected UIs. We do NOT build reckless evasion. This module defines the
adapter interface and classifies SGP scraping as research-only / manual, with a
Playwright skeleton the user can enable deliberately for low-frequency capture.

Recommended viable alternatives (cheaper + ToS-clean):
  * a provider that exposes SGP/parlay odds via API (e.g. OddsJam/Unabated), OR
  * MANUAL entry of a handful of SGP quotes for the pair-validation study.
"""

from __future__ import annotations

from dataclasses import dataclass

from .storage import now_iso
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class Leg:
    player: str
    market: str          # player_points / player_rebounds / player_assists
    selection: str       # over / under
    line: float


def viability() -> dict:
    return {"viable_automated": False,
            "reason": "US SGP builders require auth + JS + anti-bot; automated scraping "
                      "is brittle and ToS-risky.",
            "recommended": ["SGP odds via a provider API (OddsJam/Unabated)",
                            "manual quote entry for the low-frequency pair study"]}


def get_sgp_quote(bookmaker: str, event_id: str, legs: list[Leg],
                  parlay_odds_american: float | None = None) -> dict | None:
    """Adapter. If a provider/manual price is supplied, normalize+return a quote.

    parlay_odds_american: pass a known/manually-read SGP price to record it.
    Automated browser capture is intentionally NOT performed here.
    """
    if parlay_odds_american is None:
        log.info("SGP capture is manual/provider-only: %s", viability()["reason"])
        return None
    from edge.odds_espn import american_to_decimal
    from propmodel.devig import implied
    return {"captured_at": now_iso(), "bookmaker": bookmaker, "event_id": event_id,
            "legs_json": [leg.__dict__ for leg in legs],
            "parlay_odds_american": float(parlay_odds_american),
            "parlay_odds_decimal": american_to_decimal(parlay_odds_american),
            "implied_prob_raw": round(implied(parlay_odds_american), 4),
            "metadata_json": {"source": "manual_or_provider"}}


def record_sgp_quote(quote: dict | None) -> int:
    if not quote:
        return 0
    from .storage import init_storage, insert
    init_storage()
    return insert("sgp_quotes", [quote])
