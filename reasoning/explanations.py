"""Layer 3b — plain-English renderer (Section 2/11).

Turns structured evidence into analyst-style text. Only mentions factors that are
actually present in the evidence (which is itself grounded in real feature
values). No generic 'model agrees'; no invented causes.
"""

from __future__ import annotations


def _join(phrases: list[str]) -> str:
    phrases = [p for p in phrases if p]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def why_it_stands_out(ev: dict, side: str) -> list[str]:
    pos = [i["phrase"] for b in ("opportunity", "context") for i in ev[b]
           if i["polarity"] > 0]
    return pos[:4]


def why_confidence_is_limited(ev: dict) -> list[str]:
    lim = [i["phrase"] for i in ev["reliability"]]
    lim += [i["phrase"] for i in ev["context"] if i["polarity"] == 0]  # e.g. role unstable
    return lim[:4]


def game_context_summary(row: dict, ev: dict) -> str:
    team, opp = row.get("team") or "?", row.get("opponent")
    head = f"{team} vs {opp}" if (opp and not str(opp).isdigit()) else f"{team}"
    bits = []
    pace = row.get("opp_pace")
    if pace:
        bits.append(f"pace ~{float(pace):.0f}")
    matchup = next((i["phrase"] for i in ev["context"] if i["key"] == "matchup"), None)
    if matchup:
        bits.append(matchup)
    return head + (": " + "; ".join(bits) if bits else "")


def verdict_summary(row: dict, ev: dict, side: str, final_score: float,
                    decision_conf: float) -> str:
    pros = why_it_stands_out(ev, side)
    sidetxt = "OVER" if side == "over" else "UNDER"
    mkt = row.get("market_type", "this prop")
    player = row.get("player_name", "player")
    line = row.get("line_value")
    if not pros:
        body = "no strong basketball or market factors lean either way right now"
        head = f"{player} {mkt} {line}: no clear edge"
    else:
        body = _join(pros)
        head = f"{player} {mkt} {sidetxt} {line}"
    summary = f"{head} — this {sidetxt.lower()} stands out because {body}." if pros \
        else f"{head} — {body}."
    lims = why_confidence_is_limited(ev)
    if lims:
        summary += f" Confidence is capped because {_join(lims)}."
    return summary


def reason_tags(ev: dict) -> list[str]:
    tags = []
    keymap = {"stale_vs_anchor": "STALE", "news": "NEWS", "line_move": "MOVE",
              "minutes_trend": "MINUTES", "matchup": "MATCHUP", "pace": "PACE",
              "role": "STARTER", "pair": "PAIR"}
    for b in ("opportunity", "context", "entry"):
        for i in ev[b]:
            t = keymap.get(i["key"])
            if t and i["polarity"] >= 0 and t not in tags:
                tags.append(t)
    return tags


def risk_tags(ev: dict, row: dict) -> list[str]:
    risks = []
    for i in ev["reliability"]:
        if i["key"] == "volatility":
            risks.append("HIGH-VAR")
        if i["key"] == "wide_range":
            risks.append("WIDE-RANGE")
        if i["key"] == "feature_gaps":
            risks.append("THIN-DATA")
    if any(i["key"] == "role_unstable" for i in ev["context"]):
        risks.append("ROLE-RISK")
    return risks
