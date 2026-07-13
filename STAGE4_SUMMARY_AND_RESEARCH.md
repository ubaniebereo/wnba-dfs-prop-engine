# WNBA Prop Engine — Project Summary + Research Brief for Perplexity

**Goal we're building toward:** a prediction engine that watches the live WNBA prop
market, prices each prop with our own model, finds **value props and correlated pairs**,
and pushes the best signals as **notifications (Claude Cowork / dashboard)**.

**Why this brief:** the data + modeling infrastructure is essentially complete, but our
honest validation says the model is **roughly at market efficiency — not yet beating the
close.** We need research on how to actually gain an edge and turn this into a
profitable, alert-driven engine. Please answer with **concrete sources, methods, and
citations**, and prefer free/low-cost where possible.

---

## PART A — What we have built and proven (Stages 1–4)

**Data (all real, multi-season):**
- BallDontLie WNBA API: full history **2008–2026**, ~32k+ player-game rows, real
  positions. ESPN: box scores + **confirmed starters** + injuries. The Odds API:
  live multi-book game + prop odds, AND a **historical plan with closing prop lines**.

**Modeling:**
- Minutes × per-minute-rate prop model (GBM), empirical-Bayes + MixedLM hierarchical
  pooling, advanced context (pace, off/def rating, defense-vs-position from box scores).
- **Negative Binomial** for rebounds/assists (beats Normal: Brier 0.220→0.212, 0.201→0.189),
  Normal+law-of-total-variance for points.
- Confirmed ESPN starters replaced the minutes proxy (minutes MAE 4.82→4.69).
- WOWY teammate-out redistribution via **modal-5 per-season starters** (minutes rise
  sensibly as starters sit).
- Edge engine: **Shin/logit devig**, best-line shopping, market-centering + calibration.

**The honest validation (the key result):** a real **CLV backtest** vs historical closing
lines. On a 46-bet sample: **mean CLV +0.43%, 52% beat the close, 57% realized hit rate.**
→ Essentially **break-even / market-efficient**, NOT a demonstrated edge (and underpowered).

**What this proved:** our model's large "edges" (~13% median) are mostly **model error**, not
value — if they were real, CLV would be strongly positive. Better distributions (NB) even
*raised* apparent edges by making imperfect means more confident. **The remaining gap is not
plumbing — it's genuine predictive sharpness and/or market-mechanics edge.**

---

## PART B — Research questions (how do we actually gain an edge + build the engine?)

### 1. Where does edge in WNBA props REALLY come from? (most important)
- For a niche market like WNBA props, is the realistic edge primarily **market mechanics**
  (line shopping across books, reacting to news faster than books, exploiting **stale lines**
  on role/injury changes, low-limit soft books) rather than a sharper box-score model?
- Is the proven sharp method **"anchor fair value to Pinnacle (or the sharpest book), de-vig
  it, and bet other books' stale numbers"** — and does that work for WNBA props specifically?
- Which **books are softest** for WNBA props, what are typical **limits**, and how fast do
  WNBA prop lines move after lineup/injury news?

### 2. What information are we missing that sharp prop bettors use?
- **Real-time injury/lineup news latency**: what's the fastest free/cheap WNBA source to
  detect a starter scratch BEFORE books fully adjust? (Beat-writer Twitter/X lists, team
  PR, RotoWire/Underdog alerts — which has the lowest latency and an API?)
- **Sharp/consensus odds**: how to access **Pinnacle** WNBA prop odds (directly or via
  aggregator) to use as the fair-value anchor.
- **Matchup granularity**: defender-level matchup data, on/off splits, pace-up/pace-down
  spots, blowout scripts, referee tendencies, B2B/travel — which of these measurably move
  WNBA prop outcomes, and where is the data?

### 3. Increasing ROI — staking, line shopping, and the metric that matters
- Realistic **ROI / yield** for WNBA props for a disciplined bettor, and the **CLV threshold**
  (e.g., % beating close) that statistically confirms a real edge. How many bets to be sure?
- **Kelly / fractional-Kelly** sizing for props with ~$500–1000 limits; bankroll mechanics.
- Best **two-way devig** for props (we use Shin) — is there evidence Shin/logit/power beats
  multiplicative on realized prop frequencies?

### 4. "Pairs" — correlated props and same-game value (the user explicitly wants this)
- How to find and price **correlated prop pairs / same-game parlays** in basketball:
  - Within-team correlations (two scorers, pace-driven dual overs, usage substitution when
    a star sits), negative correlations (two bigs splitting rebounds).
  - How to **estimate the correlation structure** from box-score history and detect when a
    book's **parlay/SGP price misprices correlation** (the classic +EV pair angle).
- Is correlated-pair mispricing a more reliable WNBA edge than single props? References.

### 5. The prediction → notification engine (architecture)
- For a value-scanning engine that alerts on props + pairs, what should the **ranking signal**
  be — model edge, **CLV-expectation**, edge-vs-Pinnacle, or a blend — and what **thresholds**
  make an alert "send-worthy" (avoid alert spam)?
- **Latency**: how fresh must odds be to act before lines move? Recommended **polling cadence**
  vs The Odds API credit budget; how to detect **line movement** as its own signal.
- Patterns for pushing alerts (we'd send to Claude Cowork or a dashboard): what fields make a
  signal actionable (player, market, line, book, fair prob, edge, CLV estimate, reason tag)?

### 6. Model sharpness — is there headroom left, or is the market just efficient?
- Given box-score-only models plateau near market efficiency, what features have the **highest
  documented lift** for player props (shot quality / location, tracking, role/usage change
  detection, minutes-projection accuracy)? Is any of it available for WNBA free/cheap?
- Or is the honest conclusion that **single-prop modeling can't beat WNBA closes**, and edge
  must come from mechanics (Sections 1, 4)?

### What we already have / dead ends (don't re-research)
- Have: BDL 2008–2026 box scores + positions; ESPN confirmed starters + injuries; The Odds
  API live + **historical** props; NB distributions; devig; real CLV backtest harness.
- Dead ends: stats.wnba.com (Akamai-blocked), SportsData.io trial (fake), no free WNBA
  shot-tracking. Single-prop box-score model is ~at market efficiency.

### Definition of "ready to be a real engine"
A scan that ranks props + pairs by a CLV-predictive signal, alerts only above a threshold,
and shows — over a **200+ bet** forward sample — **positive CLV (>~55–60% beating close)** and
calibrated probabilities. Until CLV is convincingly positive, treat outputs as research, not value.
