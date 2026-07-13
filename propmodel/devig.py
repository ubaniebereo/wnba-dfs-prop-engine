"""Two-way devig methods (Stage 3, Section 7).

Multiplicative (simple), Shin (accounts for insider/longshot bias), and
logit/probit (symmetric two-way). For ~-110/-115 prop markets Shin and logit
are typically closer to realized frequencies than multiplicative.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def implied(american: float) -> float:
    o = float(american)
    return 100.0 / (o + 100.0) if o > 0 else (-o) / (-o + 100.0)


def devig_multiplicative(q_a: float, q_b: float) -> tuple[float, float]:
    s = q_a + q_b
    return q_a / s, q_b / s


def devig_shin(q_a: float, q_b: float) -> tuple[float, float]:
    """Shin (1992): solve for insider fraction z, back out fair probs."""
    qs = np.array([q_a, q_b], float)
    booksum = qs.sum()

    def fair(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * qs ** 2 / booksum) - z) / (2 * (1 - z))

    def constraint(z):
        return fair(z).sum() - 1.0

    try:
        z = brentq(constraint, 1e-6, 0.2)
    except ValueError:
        return devig_multiplicative(q_a, q_b)
    p = fair(z)
    return float(p[0]), float(p[1])


def devig_logit(q_a: float, q_b: float) -> tuple[float, float]:
    """Probit/logit-style: shift both implied probs in z-space to sum to 1."""
    za, zb = norm.ppf(np.clip([q_a, q_b], 1e-6, 1 - 1e-6))

    def total(shift):
        return norm.cdf(za - shift) + norm.cdf(zb - shift) - 1.0

    try:
        s = brentq(total, -3, 3)
    except ValueError:
        return devig_multiplicative(q_a, q_b)
    return float(norm.cdf(za - s)), float(norm.cdf(zb - s))


METHODS = {"multiplicative": devig_multiplicative, "shin": devig_shin,
           "logit": devig_logit}


def devig(american_a: float, american_b: float, method="shin") -> tuple[float, float]:
    return METHODS[method](implied(american_a), implied(american_b))
