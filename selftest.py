"""Validate the cycle detector on synthetic players (no API key needed).

Builds two kinds of fake players on a realistic WNBA-like cadence (~40 games
over ~120 days, unevenly spaced):
  - CYCLIC: efficiency has a true 28-day dip + noise
  - NOISE : efficiency is pure noise, no cycle

A trustworthy detector should flag most CYCLIC players and almost no NOISE
players. We print the separation so you can judge the false-positive rate.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from wnba_cycles.analysis import build_series, cycle_test, PlayerSeries


def _fake_dates(rng, n=40, span=120):
    days = np.sort(rng.choice(np.arange(span), size=n, replace=False)).astype(float)
    start = datetime(2025, 5, 16)
    return days, [start + timedelta(days=int(d)) for d in days]


def _series_from(days, dates, gmsc, pid, name):
    mu, sd = gmsc.mean(), gmsc.std()
    z = (gmsc - mu) / sd
    pts = np.clip(gmsc * 0.8 + 8, 0, None)
    return PlayerSeries(pid, name, days, dates, gmsc, pts, z, np.full_like(gmsc, 28.0))


def make_cyclic(rng, pid, period=28.0, amp=6.0, noise=4.0):
    days, dates = _fake_dates(rng)
    phase = rng.uniform(0, period)
    base = 14.0
    sig = -amp * np.cos(2 * np.pi * (days - phase) / period)  # dip once per cycle
    gmsc = base + sig + rng.normal(0, noise, size=len(days))
    return _series_from(days, dates, gmsc, pid, f"cyclic_{pid}")


def make_noise(rng, pid, noise=4.0):
    days, dates = _fake_dates(rng)
    gmsc = 14.0 + rng.normal(0, noise, size=len(days))
    return _series_from(days, dates, gmsc, pid, f"noise_{pid}")


def main():
    rng = np.random.default_rng(7)
    N = 40
    cyc_flagged = noise_flagged = 0
    cyc_p, noise_p = [], []
    for i in range(N):
        s = make_cyclic(rng, i)
        r = cycle_test(s, n_perm=800, rng=rng)
        cyc_p.append(r.p_value)
        cyc_flagged += r.significant
    for i in range(N):
        s = make_noise(rng, 1000 + i)
        r = cycle_test(s, n_perm=800, rng=rng)
        noise_p.append(r.p_value)
        noise_flagged += r.significant

    print("=== Cycle detector self-test ===")
    print(f"CYCLIC players (true 28d dip): flagged {cyc_flagged}/{N} "
          f"(sensitivity {cyc_flagged/N:.0%}), median p={np.median(cyc_p):.3f}")
    print(f"NOISE  players (no cycle)    : flagged {noise_flagged}/{N} "
          f"(false-positive {noise_flagged/N:.0%}), median p={np.median(noise_p):.3f}")
    print()
    ok = cyc_flagged >= 0.5 * N and noise_flagged <= 0.15 * N
    print("RESULT:", "PASS — detector separates signal from noise"
          if ok else "WEAK — tune amp/noise/thresholds")


if __name__ == "__main__":
    main()
