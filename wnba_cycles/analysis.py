"""Efficiency, drop detection, and cycle (periodicity) testing.

The pipeline per player:
  1. game_score()    -> one efficiency number per game (Hollinger Game Score)
  2. build_series()  -> dates + z-scored efficiency residuals vs the player's
                        own baseline (so we test the player against themselves)
  3. cycle_test()    -> Lomb-Scargle periodogram over uneven game dates, with a
                        permutation test for significance, restricted to the
                        21-35 day band the user cares about.
  4. flag_player()   -> combine into a single red-flag verdict + score.

Why Lomb-Scargle: WNBA games are NOT evenly spaced in time, so a plain FFT is
invalid. Lomb-Scargle is the standard tool for periodicity in unevenly-sampled
series. Why a permutation test: it gives an honest p-value under THIS player's
exact sampling cadence, which kills most of the false-pattern risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy.signal import lombscargle

# Period band the user is interested in (days).
BAND_LO, BAND_HI = 21.0, 35.0

# Tunables
MIN_GAMES = 15          # need enough games for any periodicity power
DROP_Z = -0.8           # a game is a "drop" if its z-score is below this
SIG_P = 0.05            # permutation p-value to call a cycle real
N_PERM = 2000           # permutations for the significance test


# --------------------------------------------------------------------------
# 1. per-game efficiency
# --------------------------------------------------------------------------
def _f(stat: dict, *keys: str, default: float = 0.0) -> float:
    """Tolerant numeric field getter (API key naming varies)."""
    for k in keys:
        if k in stat and stat[k] is not None:
            try:
                return float(stat[k])
            except (TypeError, ValueError):
                pass
    return default


def _minutes(stat: dict) -> float:
    m = stat.get("min", stat.get("minutes"))
    if m is None:
        return 0.0
    if isinstance(m, str):  # "MM:SS" or "MM"
        if ":" in m:
            mm, ss = m.split(":")[:2]
            return float(mm) + float(ss) / 60.0
        return float(m or 0)
    return float(m)


def game_score(stat: dict) -> float:
    """Hollinger Game Score — a single-number per-game efficiency rating."""
    pts = _f(stat, "pts", "points")
    fgm = _f(stat, "fgm", "field_goals_made")
    fga = _f(stat, "fga", "field_goals_attempted")
    ftm = _f(stat, "ftm", "free_throws_made")
    fta = _f(stat, "fta", "free_throws_attempted")
    orb = _f(stat, "oreb", "offensive_rebounds")
    drb = _f(stat, "dreb", "defensive_rebounds")
    reb = _f(stat, "reb", "rebounds")
    if orb == 0 and drb == 0 and reb:  # only total available -> split ~30/70
        orb, drb = 0.3 * reb, 0.7 * reb
    ast = _f(stat, "ast", "assists")
    stl = _f(stat, "stl", "steals")
    blk = _f(stat, "blk", "blocks")
    tov = _f(stat, "turnover", "turnovers", "to")
    pf = _f(stat, "pf", "personal_fouls", "fouls")
    return (pts + 0.4 * fgm - 0.7 * fga - 0.4 * (fta - ftm)
            + 0.7 * orb + 0.3 * drb + stl + 0.7 * ast + 0.7 * blk
            - 0.4 * pf - tov)


def _game_date(stat: dict) -> datetime | None:
    g = stat.get("game") or {}
    ds = g.get("date") or stat.get("date")
    if not ds:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ds[:len(fmt) + 6], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ds.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# 2. build the per-player time series
# --------------------------------------------------------------------------
@dataclass
class PlayerSeries:
    player_id: int
    name: str
    days: np.ndarray          # days since first game
    dates: list[datetime]
    gmsc: np.ndarray          # raw game score per game
    pts: np.ndarray           # points per game (for prop comparison)
    z: np.ndarray             # z-scored efficiency residual vs own baseline
    minutes: np.ndarray

    @property
    def n(self) -> int:
        return len(self.days)


def build_series(player_id: int, name: str, stats: list[dict]) -> PlayerSeries | None:
    rows = []
    for s in stats:
        d = _game_date(s)
        if d is None:
            continue
        mins = _minutes(s)
        if mins <= 0:                      # DNP / did-not-play: skip, not a "drop"
            continue
        rows.append((d, game_score(s), _f(s, "pts", "points"), mins))
    if len(rows) < MIN_GAMES:
        return None
    rows.sort(key=lambda r: r[0])
    dates = [r[0] for r in rows]
    days = np.array([(d - dates[0]).days for d in dates], dtype=float)
    gmsc = np.array([r[1] for r in rows], dtype=float)
    pts = np.array([r[2] for r in rows], dtype=float)
    mins = np.array([r[3] for r in rows], dtype=float)
    mu, sd = gmsc.mean(), gmsc.std()
    z = (gmsc - mu) / sd if sd > 1e-9 else np.zeros_like(gmsc)
    return PlayerSeries(player_id, name, days, dates, gmsc, pts, z, mins)


def drop_games(s: PlayerSeries) -> np.ndarray:
    """Boolean mask of games that are efficiency drops vs the player's baseline."""
    return s.z <= DROP_Z


# --------------------------------------------------------------------------
# 3. periodicity test
# --------------------------------------------------------------------------
@dataclass
class CycleResult:
    player_id: int
    name: str
    n_games: int
    best_period: float        # days, peak within the 21-35 band
    peak_power: float
    p_value: float            # permutation p-value (lower = more real)
    in_band: bool             # peak period within 21-35d
    significant: bool
    drop_mean_gmsc: float     # avg efficiency on predicted-dip games
    base_mean_gmsc: float     # avg efficiency on all other games
    drop_mean_pts: float
    base_mean_pts: float
    dip_fraction: float       # fraction of variance the cycle explains-ish
    next_dip_dates: list[str] = field(default_factory=list)
    red_flag_score: float = 0.0


def _periodogram(days: np.ndarray, vals: np.ndarray, periods: np.ndarray) -> np.ndarray:
    ang = 2.0 * np.pi / periods                      # angular frequencies
    v = vals - vals.mean()
    return lombscargle(days, v, ang, normalize=True)


def cycle_test(s: PlayerSeries, *, n_perm: int = N_PERM,
               rng: np.random.Generator | None = None) -> CycleResult:
    rng = rng or np.random.default_rng(12345)
    periods = np.linspace(10.0, 45.0, 350)           # scan a wide band...
    power = _periodogram(s.days, s.z, periods)
    band = (periods >= BAND_LO) & (periods <= BAND_HI)   # ...score only 21-35d
    band_power = power[band]
    best_idx = np.argmax(band_power)
    best_period = float(periods[band][best_idx])
    peak = float(band_power[best_idx])

    # Permutation null: shuffle efficiency values across the SAME game dates.
    # This preserves the player's exact (uneven) cadence -> honest p-value.
    null_max = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(s.z)
        p = _periodogram(s.days, perm, periods)
        null_max[i] = p[band].max()
    p_value = float((np.sum(null_max >= peak) + 1) / (n_perm + 1))

    # Phase-fold on the best period to label predicted "dip" games:
    # the dip is the half-cycle centered on the periodogram's trough phase.
    phase = (s.days % best_period) / best_period
    # estimate dip phase = phase where z is lowest on average (cosine fit)
    c = np.cos(2 * np.pi * phase)
    sn = np.sin(2 * np.pi * phase)
    # least squares: z ~ a*cos + b*sin ; trough at angle of (a,b)+pi
    A = np.vstack([c, sn, np.ones_like(c)]).T
    coef, *_ = np.linalg.lstsq(A, s.z, rcond=None)
    a, b = coef[0], coef[1]
    amp = float(np.hypot(a, b))
    trough_angle = (np.arctan2(b, a) + np.pi) % (2 * np.pi)
    ang_games = 2 * np.pi * phase
    dist = np.abs((ang_games - trough_angle + np.pi) % (2 * np.pi) - np.pi)
    dip_mask = dist <= (np.pi / 2)                   # within quarter-cycle of trough
    base_mask = ~dip_mask

    def _m(arr, mask):
        return float(arr[mask].mean()) if mask.any() else float("nan")

    in_band = True  # peak is by construction inside the band
    significant = (p_value < SIG_P) and (s.n >= MIN_GAMES)
    res = CycleResult(
        player_id=s.player_id, name=s.name, n_games=s.n,
        best_period=best_period, peak_power=peak, p_value=p_value,
        in_band=in_band, significant=significant,
        drop_mean_gmsc=_m(s.gmsc, dip_mask), base_mean_gmsc=_m(s.gmsc, base_mask),
        drop_mean_pts=_m(s.pts, dip_mask), base_mean_pts=_m(s.pts, base_mask),
        dip_fraction=amp,
    )

    # Predict upcoming dip dates: project trough phase forward from last game.
    last = s.dates[-1]
    base_day = s.days[-1]
    upcoming = []
    for k in range(0, 4):
        # next day where phase hits trough_angle
        target = trough_angle / (2 * np.pi)
        cur_phase = (base_day % best_period) / best_period
        delta = (target - cur_phase) % 1.0
        day_ahead = base_day + delta * best_period + k * best_period
        upcoming.append((last + _td(day_ahead - base_day)).strftime("%Y-%m-%d"))
    res.next_dip_dates = upcoming

    # Red-flag score: significance x dip depth x sample size, only if it bites.
    depth = res.base_mean_gmsc - res.drop_mean_gmsc
    res.red_flag_score = max(0.0, -np.log10(p_value)) * max(0.0, depth) \
        * np.sqrt(s.n) if significant else 0.0
    return res


def _td(days: float):
    from datetime import timedelta
    return timedelta(days=float(days))


def flag_player(player_id: int, name: str, stats: list[dict],
                **kw) -> CycleResult | None:
    s = build_series(player_id, name, stats)
    if s is None:
        return None
    return cycle_test(s, **kw)
