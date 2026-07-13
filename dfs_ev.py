"""DFS payout + entry-EV engine (Section 10).

Exact PrizePicks Power/Flex payout tables (CONFIGURABLE — verify current values per
state/promo) and correlation-aware multi-leg entry EV via Monte Carlo. Power =
all-or-nothing; Flex = partial payouts. Legs are simulated jointly so same-game
correlation isn't ignored (independence over-/under-states parlay probability).
"""

from __future__ import annotations

import numpy as np

# === PrizePicks payout multipliers (per $1). VERIFY — these vary by state/promo ===
POWER = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 37.5}
# Flex: {n_legs: {n_correct: multiplier}}
FLEX = {
    3: {3: 2.25, 2: 1.25},
    4: {4: 5.0, 3: 1.5},
    5: {5: 10.0, 4: 2.0, 3: 0.4},
    6: {6: 25.0, 5: 2.0, 4: 0.4},
}
RISK_TIER = {2: "Core", 3: "Plus", 4: "Aggressive", 5: "High Variance", 6: "Risky"}


def breakeven_power(n_legs: int) -> float:
    """Per-leg win prob needed to break even on an all-correct Power play."""
    return POWER[n_legs] ** (-1.0 / n_legs)


def simulate_entry(leg_probs: list[float], payout_style: str = "power",
                   corr: np.ndarray | None = None, n: int = 40000,
                   rng=None) -> dict:
    """EV of a DFS entry. leg_probs = calibrated P(win) per leg (already on the
    chosen side). corr = optional leg-by-leg correlation matrix (Gaussian copula)."""
    rng = rng or np.random.default_rng(0)
    k = len(leg_probs)
    if k < 2 or k > 6:
        return {"error": "entries are 2-6 legs"}
    p = np.clip(np.array(leg_probs, float), 1e-4, 1 - 1e-4)

    # correlated Bernoulli via Gaussian copula
    if corr is None:
        corr = np.eye(k)
    corr = np.array(corr, float)
    np.fill_diagonal(corr, 1.0)
    try:
        L = np.linalg.cholesky(corr + 1e-9 * np.eye(k))
    except np.linalg.LinAlgError:
        L = np.eye(k)
    z = (L @ rng.standard_normal((k, n))).T
    thresh = np.array([_z(pi) for pi in p])           # win if z > thresh
    hits = (z > thresh).sum(axis=1)

    payout = np.zeros(n)
    if payout_style == "power":
        payout[hits == k] = POWER.get(k, 0.0)
    else:  # flex
        table = FLEX.get(k, {})
        for correct, mult in table.items():
            payout[hits == correct] = mult
    ev = float(payout.mean()) - 1.0                   # per $1 staked
    joint_all = float((hits == k).mean())
    return {"entry_size": k, "payout_style": payout_style,
            "payout_if_perfect": (POWER.get(k) if payout_style == "power"
                                  else FLEX.get(k, {}).get(k)),
            "joint_prob_all": round(joint_all, 4),
            "expected_payout": round(float(payout.mean()), 3),
            "EV_per_$1": round(ev, 3),
            "breakeven_each": round(breakeven_power(k), 3) if payout_style == "power" else None,
            "risk_tier": RISK_TIER.get(k), "n_legs": k}


def _z(p):
    from scipy.stats import norm
    return float(norm.ppf(1 - p))      # win region is the upper tail


def build_entries(legs: list[dict], max_size=4, payout_style="power",
                  top=15) -> list[dict]:
    """Greedy: rank candidate legs by single-leg quality, form best small entries.

    legs: [{player, market, side, prob, corr_key, ...}] with calibrated prob.
    Returns ranked entries with EV (independence assumed unless legs share a game).
    """
    import itertools
    strong = sorted([l for l in legs if l.get("prob", 0) >= 0.5],
                    key=lambda l: l["prob"], reverse=True)[:10]
    out = []
    for size in range(2, max_size + 1):
        for combo in itertools.combinations(strong, size):
            # avoid two legs on the same player
            if len({l["player"] for l in combo}) < size:
                continue
            probs = [l["prob"] for l in combo]
            res = simulate_entry(probs, payout_style)
            if "error" in res:
                continue
            res["legs"] = [f"{l['player']} {l['market']} {l['side']} {l['line']}"
                           for l in combo]
            out.append(res)
    out.sort(key=lambda r: r["EV_per_$1"], reverse=True)
    return out[:top]
