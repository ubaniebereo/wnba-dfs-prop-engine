"""Synthetic player-game generator with KNOWN ground-truth cycle effects.

Why synthetic: real menstrual-cycle phase/symptom data does not exist in any
public basketball feed and must not be inferred about real athletes. To validate
a *small-effect detector* we instead plant effects we control, sized to match the
research (trivial-to-small: ~1-3% efficiency, <1 rebound), in a MINORITY of
players, and bury them under realistic basketball + fatigue + random variance.
The pipeline is then judged on whether it recovers the planted truth.

Ground-truth columns (prefix `gt_`) are emitted for evaluation ONLY and must not
be used as model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = ["menstruation", "follicular", "luteal"]
POSITIONS = ["PG", "SG", "SF", "PF", "C"]


def _phase_from_day(day_in_cycle: float) -> str:
    """Map a day within the cycle to one of three coarse phases."""
    d = day_in_cycle
    if d < 5:
        return "menstruation"
    if d < 14:
        return "follicular"
    return "luteal"          # 14..cycle_len (incl. ovulation->late luteal)


def generate(n_players: int = 40, n_games: int = 38, seed: int = 7,
             sensitive_fraction: float = 0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n_sensitive = max(1, int(round(n_players * sensitive_fraction)))
    sensitive_ids = set(rng.choice(n_players, n_sensitive, replace=False).tolist())

    for pid in range(n_players):
        pos = POSITIONS[pid % len(POSITIONS)]
        is_big = pos in ("PF", "C")
        # stable player profile
        base_min = rng.uniform(22, 33)
        base_ts = rng.normal(0.55, 0.03)
        base_usage = rng.uniform(0.18, 0.30)
        base_pts = base_min * rng.uniform(0.35, 0.62)
        base_reb = base_min * (rng.uniform(0.18, 0.30) if is_big else rng.uniform(0.07, 0.14))
        base_pir = base_pts * 0.7 + base_reb * 0.8 + rng.uniform(2, 6)
        med_use = int(rng.random() < 0.35)            # hormonal contraception/NSAID
        cycle_len = float(np.clip(rng.normal(28, 1.5), 24, 32))
        phase0 = rng.uniform(0, cycle_len)            # phase at season start

        # planted ground-truth cycle effect (only for sensitive players, small)
        sensitive = pid in sensitive_ids
        amp = rng.uniform(0.6, 1.0) if sensitive else 0.0
        if med_use:
            amp *= 0.5                                # meds blunt the effect
        gt_e_ts = 0.022 * amp                         # ~2% TS swing at most
        gt_e_reb = 0.7 * amp
        gt_e_pir = 1.6 * amp
        gt_e_pts = 1.1 * amp

        # schedule: walk dates with realistic rest pattern
        date = pd.Timestamp("2025-05-16") + pd.Timedelta(days=int(rng.integers(0, 6)))
        recent_min: list[float] = []
        week_anchor = date
        games_this_week = 0

        for g in range(n_games):
            gap = int(rng.choice([1, 2, 2, 3, 3, 4], p=[.18, .26, .0, .28, .0, .28])
                      if False else rng.choice([1, 2, 3, 4], p=[.20, .34, .28, .18]))
            date = date + pd.Timedelta(days=gap)
            days_rest = gap
            b2b = int(days_rest == 1)
            if (date - week_anchor).days >= 7:
                week_anchor = date
                games_this_week = 0
            game_in_week = games_this_week
            games_this_week += 1
            cum_min_3 = float(np.sum(recent_min[-3:]))

            # cycle phase for this date (independent of fatigue by construction)
            day_in_cycle = (phase0 + (date - pd.Timestamp("2025-05-16")).days) % cycle_len
            phase = _phase_from_day(day_in_cycle)

            # symptoms: higher in menstruation & late luteal, blunted by meds
            sym_base = {"menstruation": 6.0, "follicular": 2.0, "luteal": 4.3}[phase]
            symptom_score = float(np.clip(
                sym_base + rng.normal(0, 1.6) - 2.2 * med_use, 0, 10))
            perceived_recovery = float(np.clip(
                8 - 0.6 * symptom_score + 0.4 * (days_rest - 2) + rng.normal(0, 1.0),
                1, 10))

            # context
            opp_strength = float(rng.normal(0, 1))    # z-scored opponent quality
            home = int(rng.random() < 0.5)
            minutes = float(np.clip(base_min + rng.normal(0, 3)
                                    - 1.2 * b2b - 0.05 * (cum_min_3 - base_min * 3) / 5,
                                    8, 40))
            recent_min.append(minutes)

            # ---- performance deltas: BASKETBALL + FATIGUE dominate, cycle is faint
            fatigue = (-0.012 * b2b - 0.004 * max(0, 3 - days_rest)
                       - 0.0008 * max(0, cum_min_3 - base_min * 3))
            context = -0.018 * opp_strength + 0.006 * home
            # cycle effect direction: menstruation worse, late-follicular better
            phase_sign = {"menstruation": -1.0, "follicular": 0.6, "luteal": -0.35}[phase]
            symptoms_high = int(symptom_score >= 6)
            cyc_ts = gt_e_ts * phase_sign - 0.004 * amp * symptoms_high

            ts = base_ts + fatigue + context + cyc_ts + rng.normal(0, 0.075)
            efg = ts - rng.uniform(0.0, 0.02)
            reb = (base_reb + gt_e_reb * phase_sign + 2.5 * fatigue * 30
                   - 0.5 * opp_strength + rng.normal(0, base_reb * 0.30))
            pir = (base_pir + gt_e_pir * phase_sign + 40 * fatigue
                   - 1.2 * opp_strength + 0.6 * home + rng.normal(0, 4.0))
            pts = (base_pts + gt_e_pts * phase_sign + 30 * (fatigue + context)
                   + rng.normal(0, base_pts * 0.28))

            rows.append({
                "player_id": f"P{pid:02d}", "player_name": f"Player_{pid:02d}",
                "team_id": f"T{pid % 12:02d}", "position": pos,
                "date": date.strftime("%Y-%m-%d"), "game_id": f"G{pid:02d}_{g:02d}",
                "minutes": round(minutes, 1),
                "points": round(max(0, pts), 1), "REB": round(max(0, reb), 1),
                "TS": round(float(np.clip(ts, 0.2, 0.9)), 4),
                "eFG": round(float(np.clip(efg, 0.15, 0.9)), 4),
                "PIR": round(pir, 1),
                "usage_rate": round(base_usage + rng.normal(0, 0.02), 4),
                "cycle_phase": phase, "symptom_score": round(symptom_score, 1),
                "perceived_recovery": round(perceived_recovery, 1), "med_use": med_use,
                "days_rest": days_rest, "b2b_flag": b2b, "game_in_week": game_in_week,
                "cumulative_minutes_last_3": round(cum_min_3, 1),
                "opponent_strength": round(opp_strength, 3), "home_away": "home" if home else "away",
                # ---- ground truth (evaluation only; never a model input) ----
                "gt_cycle_sensitive": int(sensitive), "gt_amp": round(amp, 3),
                "gt_e_ts": round(gt_e_ts, 4), "gt_med_use": med_use,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/games_synth.csv", index=False)
    print(f"Generated {len(df)} player-game rows, {df.player_id.nunique()} players, "
          f"{df.gt_cycle_sensitive.sum() // df.groupby('player_id').size().mean():.0f} "
          f"sensitive players (ground truth).")
    print(df.groupby("cycle_phase")[["TS", "REB", "PIR", "symptom_score"]].mean().round(3))
