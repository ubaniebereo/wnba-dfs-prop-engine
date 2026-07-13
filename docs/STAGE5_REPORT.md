# WNBA Prop Engine — Stage 5 Report + Research Brief for Perplexity

**What this is:** an honest status of a WNBA prop-pricing + edge engine, the ideas I want to
test next, and the specific information/research I need. Please answer with **concrete sources,
methods, and citations**, free/low-cost where possible.

**One-line state:** the engine is built and honest, but it is **at market efficiency** — CLV is
break-even. The likely real edge is **correlated pairs + news-latency stale lines**, NOT a
sharper box-score model. I need help verifying that and getting the missing data.

---

## PART A — What I've built (Stages 1–5)

- **Data:** BallDontLie 2008–2026 (~32k player-games + positions); ESPN box scores, confirmed
  starters, injuries; The Odds API live multi-book + **historical closing** prop odds (incl Pinnacle in EU region).
- **Model:** minutes × per-minute-rate (GBM), hierarchical pooling (MixedLM), pace/def-vs-position
  context, **Negative Binomial** for reb/ast, confirmed-starter minutes, modal-5 WOWY.
- **Edge engine:** Shin/logit devig, **Pinnacle/consensus sharp anchor**, best-line shopping,
  **correlated-pair engine** (empirical teammate correlations + Gaussian copula vs independence),
  CLV-predictive ranking, alert payloads, news-event bus.
- **Validation:** real **CLV backtest** vs historical closing lines.

## PART B — What's working vs not working

**Working:**
- Multi-season depth fixed star regression-to-mean (star bias −0.92 → ~0).
- NB distributions calibrate counts better than Normal (Brier 0.220→0.212, 0.201→0.189).
- The **sharp anchor** gives believable edges: vs Pinnacle, only ~1 of 308 props is mispriced
  (+2.6%) — small and real, unlike the model's ~13% (which CLV proved is mostly error).
- **Pair engine** finds data-grounded correlation mispricing (teammate points mean ρ≈−0.04,
  50% negative → dual-overs overpriced vs independence). Most defensible signal we have.

**Not working / unresolved:**
- **CLV is break-even** (~52% beat close, n=23–46, underpowered). No proven edge yet.
- Single-prop **model edges don't translate to CLV** — the box-score model is ~efficient.
- Pair flags are **candidates, not confirmed** — we don't have actual SGP/parlay prices, so we
  assume books price near independence.
- No **news-latency capture** yet (we detect news but don't measure/act on the line-move lag).
- Calibration degrades on tiny live samples (overconfident in 0.6–0.7 bucket).

---

## PART C — Ideas / hypotheses I want to VERIFY (ranked)

1. **Correlation mispricing is the real edge.** Hypothesis: books price 2-leg same-game prop
   combos near independence, but teammate scoring is meaningfully negatively correlated, so
   **fading dual-overs / betting over-under splits** is +EV. *Need:* actual SGP prices to confirm.
2. **News latency is exploitable.** Hypothesis: after a confirmed starter scratch, soft books lag
   the sharp/Pinnacle move by minutes, leaving stale backup-minutes props. *Need:* measured lag.
3. **Pinnacle-anchored line shopping beats modeling.** Hypothesis: de-vig Pinnacle = fair value;
   bet soft books that haven't matched it. *Need:* does this clear vig + limits over 200+ bets?
4. **Minutes is the dominant prop driver.** Hypothesis: most prop edge is really a **minutes
   projection** edge (confirmed lineups + blowout/foul-out), not a scoring-rate edge.
5. **Totals/pace correlation for dual-overs.** Hypothesis: in high-total, fast-pace games,
   positive game-environment correlation can flip teammate pairs positive — worth a conditional copula.

## PART D — Research questions / information I need

**On confirming an edge:**
- How do I get **actual WNBA SGP / same-game-parlay prices** (book API, OddsJam/Unabated, manual)
  to confirm pair mispricing rather than assuming independence?
- What **CLV sample size + threshold** statistically confirms a real edge (n, % beat-close, CI)?
- Is **CLV vs Pinnacle's close** the right benchmark for props, vs CLV vs the book you bet?

**On news latency (idea #2):**
- Lowest-latency **free/cheap WNBA injury+lineup feed** (RotoWire, Underdog, team PR, beat-writer X
  lists) and which has an API. Typical **lag between a scratch and soft-book prop adjustment**?

**On the sharp anchor (idea #3):**
- Is Pinnacle the right WNBA anchor, or should I use a **multi-sharp consensus** (Pinnacle + Circa +
  BetOnline)? Any evidence on which book is sharpest for **WNBA props specifically**?
- Realistic **limits and hold** on WNBA props at soft books — do edges survive after vig + limits?

**On model headroom (ideas #1, #4):**
- Highest-documented-lift features for basketball player props (minutes accuracy, shot quality,
  usage-change detection) — and which exist for WNBA free/cheap?
- Any public research on **teammate stat correlation / SGP correlation modeling** in basketball
  (copula choice, conditioning on pace/total) I can borrow?

**On the engine / notifications:**
- Best practice for a **CLV-predictive ranking signal** (weighting anchor edge vs model edge vs
  CLV history) and **alert thresholds** that avoid spam while catching the rare real edge.
- Recommended **polling cadence** vs odds-API cost, and detecting **line movement as its own signal**.

## PART E — What I already have / dead ends (don't re-research)
- Have: BDL 2008–2026, ESPN starters/injuries, Odds API live+historical props (incl Pinnacle EU),
  NB distributions, Shin devig, real CLV harness, copula pair engine, news bus.
- Dead ends: stats.wnba.com (Akamai-blocked), SportsData.io trial (fake), menstrual-cycle hypothesis
  (tested negative), single-prop box-score model is ~at market efficiency.

## Definition of "real edge confirmed"
Over **200+ forward bets**: ≥55–60% beat the close, non-negative mean CLV, Brier ≤0.18–0.20,
edges mostly 1–5%, balanced over/under — with pair edges confirmed against **actual SGP prices**.
