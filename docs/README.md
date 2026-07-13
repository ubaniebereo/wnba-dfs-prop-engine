# Development log & research notes

These documents are the **working engineering log** for this project — the research briefs,
honest status reports, and open questions written at each stage of the build. They're kept
here deliberately: they show the *process*, not just the finished product.

A recurring theme runs through all of them — **verify before you trust, and report results
honestly**, even when the honest answer is "this doesn't beat the market." Highlights:

- A menstrual-cycle performance hypothesis was tested rigorously and **reported as negative**
  (no cross-season replication, indistinguishable from noise) instead of being forced to work.
- Multiple data sources were **verified and rejected** (SportsData.io returned scrambled fake
  stats; `stats.wnba.com` is bot-blocked) before settling on ones that actually work.
- A CLV backtest against real historical closing lines came back **~break-even**, and that is
  stated plainly rather than dressed up as an edge.

| File | What it covers |
|------|----------------|
| `RESEARCH_BRIEF.md` | Stage 1 — data sources, model diagnosis, open questions |
| `STAGE2_REPORT.md` | Multi-season data depth (BallDontLie), fixing star regression-to-mean |
| `STAGE4_SUMMARY_AND_RESEARCH.md` | CLV backtest, honest market-efficiency finding |
| `STAGE5_REPORT.md` | Sharp anchor, correlated pairs, news-latency edge hypotheses |
| `DASHBOARD_BRIEF.md` | Requirements for the value-scanning dashboard |
| `DASHBOARD_DFS_REPORT.md` | Pivot to DFS pick'em (PrizePicks/Underdog) + payout modeling |

For how to **run** the project, see the [top-level README](../README.md).
