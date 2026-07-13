"""Injury-note ETL (Stage 3, Section 2) — ToS-safe via ESPN's JSON API.

Rather than scraping HTML, we read the injury comments ESPN already exposes as
JSON (longComment/shortComment), run minute-cap NLP, and store flags. Rotowire
HTML parsing is intentionally left as an optional, rate-limited extension.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from minutes_caps_nlp import detect_minutes_cap
from src.database import get_engine
from src.utils import get_logger, http_get_json

log = get_logger(__name__)
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries"


def fetch_injury_notes() -> pd.DataFrame:
    payload = http_get_json(ESPN_INJURIES) or {}
    rows = []
    for team in payload.get("injuries", []) or []:
        for inj in team.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            txt = " ".join(filter(None, [inj.get("shortComment"), inj.get("longComment")]))
            cap = detect_minutes_cap(txt)
            rows.append({
                "player_id_espn": str(ath.get("id")),
                "player_name": ath.get("displayName"),
                "status": inj.get("status"), "text": txt[:300],
                "minutes_cap_flag": cap["minutes_cap_flag"],
                "minutes_cap_bucket": cap["minutes_cap_bucket"],
            })
    return pd.DataFrame(rows)


def store_injury_notes(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    eng = get_engine()
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS injury_notes"))
    df.to_sql("injury_notes", eng, if_exists="replace", index=False)
    return len(df)


def evaluate_injury_notes() -> dict:
    df = fetch_injury_notes()
    return {"notes": len(df),
            "minutes_cap_flagged": int(df["minutes_cap_flag"].sum()) if not df.empty else 0,
            "examples": df.loc[df.minutes_cap_flag == 1, "player_name"].head(5).tolist()
            if not df.empty else []}
