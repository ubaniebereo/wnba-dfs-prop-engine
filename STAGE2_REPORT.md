# WNBA Pipeline — Stage 2 Progress + Next-Steps Brief (for Perplexity)

**Read with** `RESEARCH_BRIEF.md` (Stage 1). This updates what changed after wiring in
the BallDontLie WNBA API, and asks the specific, sourced research needed to close the
remaining gap before the next build stage.

**How to help:** prefer **free/low-cost, programmatic** answers with concrete endpoints,
package names, dataset links, and citations. For modeling questions, give a recommended
recipe + 1–2 authoritative references. Flag licensing/ToS limits.

---

## PART A — What we improved and did (Stage 2)

**Data depth (the Stage-1 bottleneck) is largely solved:**
- Verified the BallDontLie WNBA key against the live API: **full history 2008–2026, no
  season restriction, 600 req/min**, real box scores + player positions.
- Built `balldontlie_client.py` + `balldontlie_etl.py`; re-sourced games + player_game_stats
  to **consistent BDL IDs** (no cross-source ID mapping), filtered All-Star/exhibition squads,
  derived opponent / is_home / a starter proxy (top-5 minutes per team-game).
- **Backfilled 2021–2026 → 32,158 player-game rows** (was ~8,300 / ~1.1 seasons → ~4×).
- Ingested **real positions** (858 players) → 3-cluster (guard/wing/big); multi-season
  baselines pass the basketball sanity check (bigs 0.246 reb/min > guards 0.112; guards
  0.107 ast/min > bigs 0.057).

**Measured effect on the diagnostics (shallow → deep):**
| Metric | Before (1.1 seasons) | After (6 seasons) |
|---|---|---|
| Star points bias (root cause of under-skew) | old RF **−0.92** | **+0.32** (≈0) |
| Live edge over/under split | ~85% unders | **~75% unders** |
| Share of edges ≤5% | 9% | **17%** |
| Top "plausible" edges | 10–15% | **4–7% (believable)** |
| Expansion-team Elo | ok | **stable on full history** |
| Prop calibration Brier (backtest) | ~0.180 | ~0.180 |

**Conclusion:** more data fixed the **star regression-to-mean** that was manufacturing fake
"under" edges. The remaining gap is no longer raw sample size.

**Still NOT solved (honest):** median flagged edge still ~10% with a model-error tail
(max ~0.37); unders still ~75% (target 50–60%); calibration only marginally improved.

---

## PART B — What we need to do next (and the research to support it)

The residual under-lean + fat tail now point at three things we have not closed:
**(1) confirmed minutes/lineups, (2) hierarchical pooling + market-weight tuning,
(3) honest validation against historical closing prop lines.**

### Highest-priority research questions

**1. Confirmed starting lineups, inactives, and minutes restrictions (free/programmatic).**
   Our `starter` is still a top-5-minutes proxy; the biggest remaining under-skew driver is
   minutes misestimation. We need:
   - The exact free feed + path for WNBA **confirmed starting lineups** pre-tip, and **how
     early** they post. Evaluate the ESPN core API
     (`sports.core.api.espn.com/v2/sports/basketball/wnba/...` rosters/depth charts),
     Rotowire, lineups.com, RotoGrinders. Which is usable via HTTP/JSON without scraping?
   - Free source for **minutes restrictions / load-management / return-from-injury minute
     caps**.

**2. Minutes & usage modeling methodology.**
   - How do sharp shops model **expected minutes** for basketball props (blowout/garbage-time
     shrinkage, foul-trouble, rotation size)? Cite references.
   - **Teammate-out redistribution**: with a starter OUT, how to estimate the minutes AND
     usage a backup gains (on/off or "with-or-without-you" methods) given limited samples?

**3. Validation against the market — the single biggest missing piece.**
   We can compute live edges but **cannot yet backtest Closing Line Value (CLV)** because we
   have no history of closing prop lines.
   - Where can we get **historical WNBA closing player-prop lines** (points/reb/ast),
     free or cheap? Evaluate The Odds API historical endpoints, OddsJam/OddsBlaze historical,
     and any open dataset. This gates honest "do we beat the market" measurement.
   - What **CLV / hit-rate-vs-implied** is considered good for niche markets like WNBA props?

**4. Distributional / variance modeling.**
   - Confirm the **law-of-total-variance** form for minutes × rate:
     `Var(stat) = E[min]²·Var(rate) + E[rate]²·Var(min) + Var(min)·Var(rate)`, and whether a
     **compound Poisson-Gamma / Negative Binomial** is preferred for reb/ast counts.
   - Any published **overdispersion (NB r)** values for WNBA (or NBA) rebound/assist props.

**5. Hierarchical partial pooling at our scale.**
   - Practical guidance: **PyMC vs statsmodels MixedLM** for per-minute-rate partial pooling
     (player → position → league) with ~6 seasons / ~30k rows. Runtime, and how to feed
     posterior means as priors/overrides for low-sample players.

**6. Advanced context we may still be missing.**
   - Does BallDontLie (or another free source) expose **usage rate, true off/def rating, or
     opponent shooting by zone / defense-vs-position** directly, or must we keep computing
     proxies from box scores?
   - Is any **WNBA shot-location / player-tracking** data public?

**7. WNBA prop market microstructure.**
   - Which WNBA prop markets/books are **softest**, typical **limits**, and how fast lines
     move (informs which edges are real vs stale).
   - Best **de-vig method for two-way props** (multiplicative vs Shin vs power).

### What we already have (so you don't re-research it)
- Multi-season box scores + positions (BallDontLie 2008–2026). ✓
- Real multi-book game + prop odds (The Odds API, live only — NOT historical). ✓
- ESPN injuries feed (statuses) and live schedule. ✓
- Dead ends: stats.wnba.com (Akamai-blocked to curl), SportsData.io trial (fake stats).

### Definition of "ready for next stage"
Confirmed-lineup minutes integrated, hierarchical pooling live, AND a **historical-closing-line
backtest** showing edges median ≤~5%, over/under ~50–60%, calibrated probabilities, and
**non-negative CLV**.
