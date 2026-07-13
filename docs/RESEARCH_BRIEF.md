# WNBA Prediction Pipeline — Research Brief for Perplexity

**Purpose:** We've built a working WNBA prediction + betting-edge pipeline on free data.
It runs end-to-end but is **not yet sharp enough to beat the market**. This brief tells you
(a) exactly what exists, (b) what we already ruled out so you don't repeat it, and
(c) the specific, sourced research we need to improve it before the next build stage.

**How to help most:** For each question, prefer **free or low-cost, programmatically
accessible** answers. Give **concrete endpoints / package names / dataset links**, cite
sources, and where it's a modeling question, give a **recommended recipe**, not just theory.

---

## 1. What we have built (current state)

**Data (free, no paid key):**
- ESPN public endpoints → SQLite: `scoreboard?dates=YYYYMMDD` (schedule/odds) and
  `summary?event=ID` (player + team box scores).
- `data.wnba.com` keyless JSON as a real fallback (schedule + game-detail box scores).
- Current DB: ~430 completed games, ~8,300 player-game rows (2025 + partial 2026 season).

**Odds (real):**
- The Odds API (free tier, 500 req/mo): multi-book moneyline/spread/total **and** player
  props (points, rebounds, assists) from FanDuel, DraftKings, BetOnline. Working.

**Models:**
- Game: logistic win prob (test acc **0.685**, ROC-AUC **0.694**, beats an Elo baseline),
  ridge/GBM margin, GBM total. **But margin R²≈0.10 and total R²≈0.03 vs the sample mean**
  — i.e., the regressions barely beat "predict the average."
- Props: RandomForest for points/rebounds/assists (MAE 3.5 / 1.5 / 1.1), leakage-free
  rolling features, residual-SD → Normal → P(over/under).
- Edge engine: de-vig (two-way normalization), best-line shopping across books, EV, and a
  "plausibility" tag that flags edges too large to be real.

**Honest result so far:** the engine flags far too many "edges" (e.g., 88 unique prop
edges, most >10%), which against sharp books means **model miscalibration, not value**. Our
models systematically under-project players vs market lines (especially stars — a RF
regresses them toward the mean). The market line is currently a *better* estimate than our model.

---

## 2. Diagnosed weaknesses (root causes to fix)

1. **Sample too small** (~1 season). Models regress everyone to the mean.
2. **No injury / inactive / confirmed-lineup / minutes data** → props guess who even plays.
3. **Crude opponent context** — we use points-allowed proxies; no true pace/possessions,
   no off/def efficiency, no opponent-defense-vs-position.
4. **No role/usage modeling** — starter vs bench, usage rate, foul trouble, blowout/garbage time.
5. **Naive prop variance** — single Gaussian residual SD, not a count-appropriate distribution.
6. **New 2026 expansion teams** (Golden State Valkyries, Toronto Tempo, Portland Fire) have
   little/no history → unstable ratings.

---

## 3. Already ruled out — DO NOT re-research these

- **stats.wnba.com / stats.nba.com**: Akamai bot-protection; raw `curl`/`requests` hang
  (HTTP 000). Need a wrapper or browser. (Question 1c asks which wrapper works.)
- **SportsData.io free trial**: returns **scrambled/fake** stat values (decimals, >40 min,
  FGM>FGA). Useless for real modeling.
- **balldontlie**: requires a paid key for stats depth + props.
- **Menstrual-cycle 21–35 day performance hypothesis**: we tested it on two real seasons —
  flags = chance level, **0 survive FDR correction, 0 cross-season replication**. No usable
  signal found, and no consented label data exists. (See Question 5 — confirm/refute only.)

---

## 4. Research questions we need answered (prioritized)

### A. Free/cheap WNBA data depth  ← highest priority
1. **Historical box scores back to ~2010**: What is the most reliable **free, programmatic**
   source? Specifically evaluate: `wehoop` (R) / `sportsdataverse` (Python), Kaggle WNBA
   datasets, Basketball-Reference WNBA (scraping terms?), and older-season availability/format
   of `data.wnba.com`. Give exact package + function or URL patterns.
2. **Injuries / inactives / availability**: Free WNBA feeds for injury status and *confirmed*
   (not projected) starting lineups, with how far before tip they post. Evaluate ESPN's
   injuries endpoint reliability, Rotowire, lineups.com, and any public API.
3. **Advanced stats** (pace/possessions, offensive/defensive rating, usage rate,
   opponent points/reb/ast allowed **by position**): which free source exposes these for the
   WNBA, and via what endpoint/wrapper? Does `nba_api` or `wehoop` cover WNBA advanced/tracking?
4. **Minutes projections**: any free source for projected minutes, or the best method to
   estimate expected minutes from recent usage + injuries.
5. **Player tracking / shot location** for WNBA — does it exist publicly at all, and where?

### B. Prop & game modeling methodology
6. **Distributional prop models**: best-practice distribution for per-game WNBA **points**
   (Gaussian vs skewed?), **rebounds/assists** (Poisson vs Negative Binomial?). Is the
   **minutes × per-minute-rate** decomposition (project minutes, then rate, then combine)
   the recommended approach? Cite NBA/WNBA prop-modeling references.
7. **Small-sample remedies**: hierarchical / Bayesian **partial pooling** for player and team
   effects with ~1 season of data — recommended frameworks (e.g., PyMC, mixed models),
   shrinkage toward position/league priors, and how many games are needed for stable rolling features.
8. **Calibrate to the market**: accepted methods for using the **de-vigged consensus line as a
   prior** (the market is sharper than a weak model). Blending model + market, and isotonic/Platt
   calibration of probability outputs. What proper-scoring benchmarks (Brier/log-loss) are
   considered "good" for WNBA props?
9. **Context handling**: how do sharp models handle **blowout/garbage-time**, **foul trouble**,
   **back-to-backs/rest**, and **starter vs bench** for player props?
10. **New-team / expansion** ratings: how to seed Elo / power ratings for the 2026 expansion
    franchises with little history (priors, regression to mean, borrowing from roster talent).

### C. Betting market reality
11. **WNBA market efficiency**: which WNBA markets are softest (player props vs sides vs
    totals), typical **hold/vig**, and realistic **achievable edge** sizes. Is **Closing Line
    Value (CLV)** the right success metric, and what CLV is considered good?
12. **De-vig method accuracy**: multiplicative vs additive vs **Shin / power** methods —
    which produces the most accurate fair probabilities, especially for longshots/props?
13. **Bankroll & sizing**: fractional **Kelly** norms for props, and how to handle correlated
    same-game markets.

### D. WNBA-specific priors
14. League scoring/pace norms and **typical total points distribution** (mean/SD) to use as a
    Bayesian prior for our totals model. WNBA games are 40 minutes — note any modeling
    differences vs the NBA (pace, possessions, scoring variance).
15. Roster/availability volatility specific to the WNBA (overseas commitments, load
    management norms) that affect minutes/availability modeling.

### E. Cycle question — confirm and shelve (ethical framing)
16. Current **peer-reviewed consensus** on menstrual-cycle phase effects on **objective**
    team-sport / basketball performance: is there ANY validated, game-level usable signal, or
    is the literature settled at "trivial/inconsistent"? We will only ever use **consented**
    athlete data and will **not infer private health info**; we just need to know whether to
    permanently shelve this or revisit it with proper data. Cite the strongest meta-analyses.

---

## 5. Answer format we want
- Lead with the **free/programmatic** option for each data question (package, endpoint, URL).
- For modeling questions, give a **recommended approach + 1–2 authoritative references**.
- Flag anything that requires paid access and give a rough cost.
- Note licensing/Terms-of-Service constraints for scraping where relevant.

## 6. Definition of "ready for the next stage"
We consider the pipeline ready to advance when:
- ≥3–4 seasons of history are ingested, and
- injuries + expected minutes are integrated, and
- prop models are **distribution-based and market-calibrated**, such that flagged edges are
  small (≤~5%), roughly balanced over/under, and our probabilities are **calibrated**
  (Brier/log-loss competitive) and ideally show **positive CLV** in a forward test.
