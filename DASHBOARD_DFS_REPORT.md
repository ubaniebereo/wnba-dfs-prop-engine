# WNBA Prop Engine — Dashboard + DFS Lines Report for Perplexity

**New, important context:** the user is in **Georgia**, where traditional online sportsbooks
aren't usable. They bet on **DFS pick'em apps — PrizePicks, Underdog, Sleeper, DraftKings
Pick6** — NOT sportsbooks. This changes the data-acquisition and pricing problem. I need help
reliably pulling DFS lines and computing correct DFS EV. Please answer with concrete endpoints,
auth requirements, libraries, payout math, and citations; free/low-cost preferred.

---

## PART A — What I built this stage

- **Interactive dashboard (Dash + Plotly):** decoupled scanner → SQLite → UI. Tabs for the
  standout-props board, correlated pairs, news, and now **PrizePicks vs Sharp**. KPI cards,
  filters, heat-colored standout score, detail panel, line-movement plumbing.
- **One-command launcher (`start.py` + double-click `.command`):** runs a first scan, keeps
  scanning every 2 min in a background thread (APScheduler), starts the dashboard, and opens
  the browser automatically. (Earlier the scanner and UI were two separate commands, which
  confused the user — now it's one.)
- **Performance fix:** swapped GradientBoosting → **HistGradientBoosting** (training ~13s → ~2s);
  added a model cache so repeated scans don't retrain.
- **Standout score (NOT model-edge-dominated):** weighted composite — anchor-relative staleness
  (0.35) + news (0.20) + fresh line move (0.15) + pair (0.10) + stale-vs-anchor (0.10) +
  model edge (only 0.10, since we proved big model edges are noise). TODO hook for a learned
  beat-close meta-model once a labeled CLV sample accrues.
- **PrizePicks (DFS) integration started:** found PrizePicks' public JSON
  (`api.prizepicks.com/projections?league_id=3` = WNBA), normalized lines, and a comparison
  that scores each PrizePicks line vs (a) the **sharp sportsbook consensus line** and (b) the
  model — surfacing where PrizePicks is soft. Sportsbooks (incl Pinnacle) are used purely as the
  fair-value **benchmark**, not a betting venue.

## PART B — What's working vs not

**Working:**
- Dashboard runs locally, one command, auto-refresh, PrizePicks tab wired.
- PrizePicks public API returns real WNBA lines (verified: 205 projections when a slate is up).
- Engine logic to compare a DFS line to the sharp anchor is sound (line_diff + model prob).

**Not working / open:**
- **Cloudflare blocks httpx/requests** on PrizePicks (403/429); **`curl_cffi` with Chrome TLS
  impersonation works (verified 200)**. Need to confirm this is a stable, ToS-acceptable long-term
  approach vs a paid aggregator.
- **Underdog & Sleeper**: not yet integrated — they appear to require auth/device tokens.
- **DFS payout math not yet modeled**: PrizePicks power/flex plays and goblin/demon lines have
  specific payout structures; our EV/"value score" currently uses a naive ~0.5 breakeven.
- DFS requires **2+ legs**, so correlation matters MORE than for single sportsbook props — our
  pair engine is relevant but not yet adapted to DFS entry construction.

---

## PART C — Research questions (DFS-focused — the user's real need)

### 1. Reliably pulling DFS pick'em lines (highest priority)
- **PrizePicks:** is the public `api.prizepicks.com/projections` endpoint stable, and is
  **curl_cffi (Chrome impersonation)** the right long-term fetch method, or will Cloudflare
  escalate? Any rate/ToS limits to respect?
- **Underdog Fantasy:** what's the actual endpoint(s) and **auth flow** for over/under lines
  (`api.underdogfantasy.com/...`)? Does it need a logged-in bearer/device token, and is
  programmatic access feasible without violating ToS?
- **Sleeper:** does Sleeper expose pick'em/props lines via API, and under what auth?
- **DraftKings Pick6** (the user says DK works for them in GA): is there a public/JSON endpoint
  for Pick6 lines?
- **Paid one-stop:** which aggregator (OddsJam, OddsBlaze, PropsCash, BettingPros) covers ALL
  these DFS apps via API, at what cost — is that more reliable than scraping each?

### 2. DFS payout structure & correct EV (so our "value" is real)
- Exact **payout multipliers** for PrizePicks **Power** vs **Flex** plays (2/3/4/5/6-pick) and
  the implied **per-leg breakeven win probability** for each.
- How **goblin** (lower line/lower payout) and **demon** (higher line/higher payout) lines change
  breakeven — are they ever +EV, and how to compute it?
- Underdog/Sleeper payout tables and their breakevens.
- Given those, what **per-leg win probability** (vs the sharp-implied prob) is needed to be +EV
  on a typical 2–6 pick entry?

### 3. Is "DFS line vs sharp sportsbook line" the right edge?
- Is comparing a DFS line to the **de-vigged sharp sportsbook consensus / Pinnacle** the proven
  method for finding soft DFS lines? What **line gap** (e.g., ≥1.0, ≥1.5 pts) is actually
  profitable after DFS payout structure?
- How fast do DFS apps move lines vs sportsbooks after news (is the news-latency edge bigger on
  DFS than on sportsbooks)?

### 4. DFS correlation / entry construction (since 2+ legs are required)
- Because DFS requires multiple legs, **correlation across legs** matters a lot. How do sharp DFS
  players build **+EV correlated entries** (same-game stacks, game-environment correlation), and
  how to price an entry's joint probability vs the flat payout?
- Does our negative teammate-scoring correlation finding imply specific DFS constructions to
  prefer/avoid (e.g., don't stack two same-team scorers' overs)?

### What we already have / dead ends (don't re-research)
- Have: BDL 2008–2026, ESPN starters/injuries, The Odds API live+historical sportsbook props
  (incl Pinnacle), X API news, NB distributions, sharp anchor, pair/copula engine, CLV harness,
  working Dash dashboard, **PrizePicks public feed via curl_cffi**.
- Dead ends: stats.wnba.com (blocked), SportsData trial (fake), httpx/requests on PrizePicks
  (Cloudflare-blocked → use curl_cffi), single-prop box-score model ~at market efficiency.

### Definition of "DFS engine is doing its job"
The dashboard pulls **PrizePicks (+ Underdog/Sleeper if feasible)** lines reliably, computes
**correct DFS EV** using real payout tables, ranks soft lines vs the sharp benchmark, accounts
for **multi-leg correlation**, and surfaces the entries that are genuinely +EV after payout
structure — with CLV-style validation against where the sharp line closes.
