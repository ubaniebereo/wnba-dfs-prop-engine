# WNBA Prop Engine — Dashboard Build Report + Research Brief for Perplexity

**Ask:** I have a working WNBA prop pricing + edge engine and a live-intelligence feed layer.
I now want to build an **interactive dashboard that scans the live prop market and surfaces the
props that stand out across ALL features** (model fair value, sharp-anchor value, edge, CLV
history, news flags, correlated-pair flags, line movement). This brief says what exists and asks
the specific questions I need answered to build it well. Prefer concrete frameworks, code
patterns, architectures, and citations; free/low-cost where possible.

---

## PART A — What I've built (Stages 1–6)

**Data (all real):** BallDontLie 2008–2026 box scores + positions; ESPN box scores, confirmed
starters, injuries; The Odds API live + **historical closing** props (incl **Pinnacle** in EU
region); **X/Twitter API** (recent search, verified). Stored in SQLite (`wnba.sqlite`,
`feeds.sqlite`).

**Models / engine:** minutes × per-minute-rate (GBM), MixedLM pooling, **Negative Binomial**
for reb/ast, Shin/logit devig, **sharp-anchor fair value** (Pinnacle/consensus), **pair engine**
(empirical teammate correlations + Gaussian copula), CLV-predictive ranking, alert payloads.

**Live-intelligence layer (Stage 6):** modular `feeds/` (X news, ESPN lineups, Odds snapshots,
diff engine for line moves, scheduler), `odds_snapshots / news_events / lineup_events /
sgp_quotes / line_move_events` tables.

**Validation:** real **CLV backtest** vs historical closing lines.

**Honest state:** single-prop box-score modeling is ~at market efficiency (CLV ~break-even);
in a steady-state snapshot **0 of 98 book prices were stale vs the anchor**. The remaining edge
candidates are **news-latency stale lines** and **correlated-pair mispricing**.

## Features I want the dashboard to scan/display PER PROP
(player, market, line) with: best book odds across books · model fair prob/line · **anchor
(Pinnacle/consensus) fair prob/line** · model edge · **anchor edge** · calibrated confidence ·
CLV history/expectation · **news flag** (out/questionable/minutes-cap) · **pair/correlation flag**
· **line-movement** (open→now, sparkline) · de-vig method · staleness vs anchor.

A "standout" = high composite score across these, not just one big model edge (we proved big
model edges are mostly noise).

---

## PART B — What I need to BUILD the dashboard (questions for Perplexity)

### 1. Framework choice (most important)
- For a **live, interactive sports-betting dashboard** (auto-refreshing odds tables, filters,
  highlighting, sparklines), what's the best tradeoff: **Streamlit vs Dash/Plotly vs FastAPI +
  React (Next.js)**? Which gives the fastest path to a *good* live board for a solo builder,
  and which scales if it becomes real-time?
- Concrete patterns for **auto-refreshing data** in each (Streamlit `st_autorefresh`/fragments,
  Dash `dcc.Interval`, React SWR/websockets) — and which handles a 30–60s odds refresh cleanly
  without full-page reloads or flicker.

### 2. Live data / refresh architecture
- Recommended **backend service** to run the scanner continuously and serve the dashboard
  (FastAPI + a background scheduler/APScheduler? a separate worker writing to SQLite/Postgres
  that the UI reads?). How to decouple **scan loop** from **UI** so odds polling doesn't block.
- **Polling cadence vs The Odds API credit budget**: how to poll often enough to catch line
  moves (esp. in news windows) without burning credits — caching, conditional fetches,
  per-event focus. Any way to detect "something moved" cheaply before a full pull?
- Storing **time-series odds** for line-movement sparklines — schema + query patterns for
  "last N snapshots per (player, market, book)" that stay fast.

### 3. The "standout" ranking + visualization (the core)
- How to combine many per-prop features into **one CLV-predictive "standout" score** (weighting
  anchor edge vs model edge vs news vs pair vs movement) — and how to **learn/validate** those
  weights against CLV rather than guessing. Is a small logistic/GBM "does-this-beat-close"
  meta-model the right approach?
- Best **visual encodings** to let a human eye catch standouts fast: heat-coloring edge,
  badges for news/pair/movement, confidence shading, sortable/filterable tables. Examples of
  good betting/odds dashboards to model the UX on.
- How to show a prop's **full feature stack** in a compact row + an expandable detail view
  (model vs anchor vs each book, movement sparkline, the reason tags).

### 4. Line-movement & news integration in the UI
- Patterns for showing **open→current line movement** and flagging **fresh moves** (e.g., last
  5 min) so news-latency opportunities surface immediately.
- How to surface a **news event → affected props** link in the UI (when a player is ruled out,
  highlight their teammates' props that should move).

### 5. Correlated-pair view
- How to present **same-game pair candidates** (dual-over overpriced, two-bigs rebound split)
  alongside singles — a separate tab, or inline? How to show copula-joint vs independence and
  the implied pair edge so it's actionable.

### 6. Backend / storage / hosting
- **SQLite vs Postgres** for a continuously-writing odds time series + a reading dashboard —
  when does SQLite's single-writer become a problem, and what's the cheapest upgrade path?
- **Hosting** a small live dashboard cheaply (Streamlit Community Cloud, Render, Fly.io, a VPS)
  — tradeoffs for a background poller + UI; auth options for a private dashboard.

### 7. Notifications (Claude Cowork / push)
- Best way to push **top standout alerts** out of the dashboard to a notification channel
  (Claude Cowork, webhook, Telegram/Discord, email) with sensible **thresholds + dedup** so it
  doesn't spam, and so an alert fires fast in a news window.

### What we already have / dead ends (don't re-research)
- Have: the full engine + feeds + SQLite stores + Pinnacle anchor + X news + CLV harness.
- Dead ends: stats.wnba.com (blocked), SportsData trial (fake), Rotowire/SGP scraping
  (JS/anti-bot, research-only), single-prop box-score model is ~efficient.

### Definition of "dashboard is doing its job"
It refreshes live, ranks props by a CLV-predictive standout score, makes news-latency and
correlated-pair opportunities visually obvious, shows line movement, and pushes only the
highest-quality alerts — so a human can act inside the stale-line window.
