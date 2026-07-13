"""Prop discovery + taxonomy (Sections 3/4/14).

Discovers the stat types actually present in the PrizePicks/DFS feed, normalizes
source labels to canonical internal names, and logs unsupported types. We do NOT
assume coverage — `discover_prizepicks()` enumerates what the live feed really has.
"""

from __future__ import annotations

import pandas as pd

# canonical name -> set of known source labels (extend as feeds reveal more)
CANON = {
    "player_points": {"Points"},
    "player_rebounds": {"Rebounds"},
    "player_assists": {"Assists"},
    "player_threes_made": {"3-PT Made", "3 Point FG", "Threes"},
    "player_steals": {"Steals"},
    "player_blocks": {"Blocked Shots", "Blocks"},
    "player_turnovers": {"Turnovers"},
    "player_pra": {"Pts+Rebs+Asts"},
    "player_pr": {"Pts+Rebs"},
    "player_pa": {"Pts+Asts"},
    "player_ra": {"Rebs+Asts"},
    "player_fantasy_score": {"Fantasy Score"},
    "player_fg_made": {"FG Made"},
    "player_ft_made": {"Free Throws Made"},
    "player_3pa": {"3-PT Attempted"},
    "player_fga": {"FG Attempted"},
    "player_blks_stls": {"Blks+Stls", "Blocks+Steals"},
    "player_double_double_yesno": {"Double Double", "Dunks"},
}
# which canonical props the current model/sim can price (others are display-only)
MODELED = {"player_points", "player_rebounds", "player_assists"}
SIMULATABLE_COMBOS = {"player_pra", "player_pr", "player_pa", "player_ra",
                      "player_fantasy_score"}

_LABEL2CANON = {lbl: c for c, lbls in CANON.items() for lbl in lbls}


def normalize_label(source_label: str) -> str | None:
    return _LABEL2CANON.get((source_label or "").strip())


def discover_prizepicks(leagues=("3",)) -> pd.DataFrame:
    """Enumerate real stat types in the live PrizePicks WNBA feed with counts."""
    from feeds.prizepicks_client import PrizePicksClient
    cl = PrizePicksClient()
    rows = []
    for lid in leagues:
        d = cl._fetch_json(int(lid))
        if not d:
            continue
        for p in d.get("data", []):
            a = p.get("attributes", {})
            rows.append({"league_id": lid, "source_label": a.get("stat_type"),
                         "odds_type": a.get("odds_type", "standard")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    tax = (df.groupby("source_label")
           .agg(count=("source_label", "size"),
                variants=("odds_type", lambda s: ",".join(sorted(set(s)))))
           .reset_index())
    tax["canonical"] = tax["source_label"].map(normalize_label)
    tax["status"] = tax["canonical"].apply(
        lambda c: "MODELED" if c in MODELED else
        "SIM-COMBO" if c in SIMULATABLE_COMBOS else
        "display-only" if c else "UNMAPPED")
    return tax.sort_values("count", ascending=False).reset_index(drop=True)
