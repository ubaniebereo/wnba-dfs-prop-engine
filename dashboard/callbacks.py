"""Dash callbacks (Section 8/9/12): refresh, filters, detail panel, tabs."""

from __future__ import annotations

import json

import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html, no_update

from . import data_access as da
from .components import badge_pills, badges, distribution_figure, kpi_card, metric, pill

DARK = {"paper_bgcolor": "#161b22", "plot_bgcolor": "#161b22",
        "font": {"color": "#8b949e", "size": 10},
        "margin": {"l": 30, "r": 10, "t": 20, "b": 20}, "height": 140}


def register(app):
    # ---- refresh KPIs + table + last-scan ----
    @app.callback(
        Output("kpi-row", "children"), Output("scan-table", "data"),
        Output("last-scan", "children"),
        Input("refresh-interval", "n_intervals"), Input("refresh-btn", "n_clicks"),
        Input("f-team", "value"), Input("f-market", "value"),
        Input("f-standout", "value"), Input("f-conf", "value"),
        Input("f-toggles", "value"), State("autorefresh", "value"),
        Input("refresh-interval", "id"))
    def _refresh(_n, _c, team, market, min_so, min_cf, toggles, auto, _trig):
        k = da.get_kpis()
        if not k.get("props"):
            return [kpi_card("Status", "no scan yet", "run scanner")], [], "—"
        kpis = [
            kpi_card("Props", k["props"]),
            kpi_card("Standouts", k["standouts"], ">= 20"),
            kpi_card("Mean Standout", k["mean_standout"]),
            kpi_card("Mean Anchor Edge", f'{k["mean_anchor_edge"]}%'),
            kpi_card("News-Flagged", k["news_flagged"]),
            kpi_card("Stale vs Anchor", k["stale_vs_anchor"]),
            kpi_card("Pairs", k["pair_flagged"]),
        ]
        df = da.filter_scan(
            da.get_scan(), team=team, market=market, min_standout=min_so or 0,
            min_conf=min_cf or 0, news_only="news" in toggles, stale_only="stale" in toggles,
            pair_only="pair" in toggles, movers_only="movers" in toggles)
        if not df.empty:
            df = df.copy()
            df["flags"] = df.apply(lambda r: badges(r.to_dict()), axis=1)
        rows = df.to_dict("records") if not df.empty else []
        ts = (k.get("last_scan") or "")[:19].replace("T", " ")
        return kpis, rows, f"last scan {ts} UTC"

    # ---- detail panel on row select ----
    @app.callback(Output("detail-panel", "children"),
                  Input("scan-table", "selected_rows"), State("scan-table", "data"))
    def _detail(sel, data):
        if not sel or not data:
            return html.Div("Select a prop row to inspect its full feature stack.",
                            style={"color": "#6e7681"})
        r = data[sel[0]]
        spark = da.get_sparkline(r["player_name"], r["market_type"], r["side"])
        books = da.get_all_books(r["player_name"], r["market_type"])

        fig = go.Figure()
        if not spark.empty and len(spark) > 1:
            fig.add_trace(go.Scatter(x=list(range(len(spark))), y=spark["odds_american"],
                          mode="lines+markers", line={"color": "#58a6ff"}))
        fig.update_layout(title="odds history", **DARK)

        book_rows = [html.Tr([html.Td(b["bookmaker"]), html.Td(b["side"]),
                              html.Td(b["line_value"]), html.Td(int(b["odds_american"]))],
                             style={"color": "#3fb950" if b.get("is_anchor_book") else "#e6edf3"})
                     for _, b in books.iterrows()] if not books.empty else []

        def bullets(text):
            items = [b.strip() for b in str(text or "").split("•") if b.strip() and b.strip() != "—"]
            return html.Ul([html.Li(b, style={"fontSize": "12px", "margin": "2px 0"})
                            for b in items], style={"margin": "4px 0", "paddingLeft": "18px"}) \
                if items else html.Div("—", className="note")

        drivers = {}
        try:
            drivers = json.loads(r.get("feature_drivers_json") or "{}")
        except Exception:
            pass
        chips = [pill(d["phrase"][:42], "stale") for d in drivers.get("positive", [])[:5]]
        chips += [pill(d["phrase"][:42], "demon") for d in drivers.get("negative", [])[:3]]

        verdict = r.get("final_verdict", "—")
        vcolor = {"LEAN OVER": "#3fd17a", "LEAN UNDER": "#3fd17a",
                  "WATCH": "#f5b945", "PASS": "#8b97a7"}.get(verdict, "#e6edf3")

        return html.Div([
            html.Div(f'{r["player_name"]} · {r["team"]} vs {r.get("opponent","?")}',
                     style={"fontSize": "15px", "fontWeight": 700}),
            html.Div(f'{r["market_type"]} {r["side"]} {r["line_value"]} @ {r["best_book"]} '
                     f'({r["best_odds_american"]})', style={"color": "#5b9dff", "marginBottom": "8px"}),
            # ---- reasoning verdict card ----
            html.Div(verdict, style={"display": "inline-block", "fontWeight": 800,
                     "color": vcolor, "fontSize": "16px", "marginBottom": "4px"}),
            html.Div(r.get("verdict_summary", ""), style={"fontSize": "12.5px",
                     "lineHeight": "1.5", "margin": "4px 0 10px", "color": "#e6edf3"}),
            # ---- raw vs reasoning vs validation comparison ----
            html.Div(className="panel", style={"padding": "8px 12px", "marginBottom": "10px"},
                     children=[
                metric("Raw standout", r.get("standout_score")),
                metric("Reasoning final score", r.get("final_standout_score")),
                metric("Raw confidence", r.get("confidence_score")),
                metric("Decision confidence", r.get("decision_confidence")),
                metric("Trust adjustment", r.get("trust_adjustment")),
                metric("Validation status", r.get("validation_status")),
                metric("Model fair p / Anchor fair p",
                       f'{r.get("model_fair_prob")} / {r.get("anchor_fair_prob")}'),
            ]),
            html.H3("Why it stands out", className="section"), bullets(r.get("why_it_stands_out")),
            html.H3("Why confidence is limited", className="section"),
            bullets(r.get("why_confidence_is_limited")),
            html.H3("Game context", className="section"),
            html.Div(r.get("game_context_summary") or "—", className="note"),
            html.H3("Top drivers", className="section"),
            html.Div(chips or [html.Span("—", className="note")], style={"margin": "4px 0"}),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Div("Books:", style={"color": "#8b97a7", "fontSize": "11px", "marginTop": "6px"}),
            html.Table(book_rows, style={"fontSize": "11px", "fontFamily": "monospace"}),
        ])

    # ---- tab switching: pairs / news / singles ----
    @app.callback(Output("tab-content", "children"), Output("singles-view", "style"),
                  Input("tabs", "value"), Input("f-market", "value"),
                  Input("f-team", "value"))
    def _tab(tab, f_market=None, f_team=None):
        if tab == "singles":
            return None, {"display": "block"}
        if tab == "prizepicks":
            pp = da.get_prizepicks()
            if not pp.empty and f_market and f_market != "ALL" and "stat_type" in pp.columns:
                pp = pp[pp["stat_type"] == f_market]          # family filter (combos/threes live here)
            if not pp.empty and f_team and f_team != "ALL" and "team" in pp.columns:
                pp = pp[pp["team"] == f_team]
            cols = [c for c in ["player_name", "stat_type", "pp_line", "book_line",
                                "line_diff", "model_prob_over", "coverage",
                                "calib_method", "recommend", "pp_value_score",
                                "odds_type", "why"]
                    if not pp.empty and c in pp.columns]
            tbl = dash_table.DataTable(
                data=pp.to_dict("records") if not pp.empty else [],
                columns=[{"name": c, "id": c} for c in cols],
                sort_action="native", page_size=40,
                style_header={"backgroundColor": "#0d1117", "color": "#8b949e"},
                style_cell={"backgroundColor": "#161b22", "color": "#e6edf3",
                            "fontSize": "12px", "fontFamily": "monospace", "padding": "4px 8px"},
                style_data_conditional=[
                    {"if": {"filter_query": "{recommend} contains 'MORE'", "column_id": "recommend"},
                     "color": "#3fb950", "fontWeight": "bold"},
                    {"if": {"filter_query": "{recommend} contains 'LESS'", "column_id": "recommend"},
                     "color": "#f85149", "fontWeight": "bold"},
                    {"if": {"filter_query": "{pp_value_score} >= 40", "column_id": "pp_value_score"},
                     "backgroundColor": "#1f5132", "color": "#fff"}])
            note = html.Div("PrizePicks line vs the SHARP sportsbook line (book_line) and the "
                            "model. line_diff = sportsbook − PrizePicks; positive (PP lower) "
                            "favors MORE. You're on PrizePicks; sportsbooks are the benchmark, "
                            "not a place to bet. Combos aren't modeled (shown w/o book_line).",
                            style={"color": "#8b949e", "fontSize": "12px", "margin": "8px 0"})
            return html.Div([note, tbl]), {"display": "none"}
        _HDR = {"backgroundColor": "#0d1117", "color": "#8b949e"}
        _CELL = {"backgroundColor": "#161b22", "color": "#e6edf3", "fontSize": "12px",
                 "fontFamily": "monospace", "padding": "4px 8px", "textAlign": "left",
                 "whiteSpace": "normal", "height": "auto", "maxWidth": "480px"}

        if tab == "entries":
            e = da.get_entries()
            cols = [c for c in ["entry_size", "risk_tier", "payout_multiple",
                                "joint_prob_all", "entry_ev", "entry_verdict", "legs_json"]
                    if not e.empty and c in e.columns]
            tbl = dash_table.DataTable(
                data=e.to_dict("records") if not e.empty else [],
                columns=[{"name": c, "id": c} for c in cols],
                sort_action="native", page_size=25, style_header=_HDR, style_cell=_CELL,
                style_data_conditional=[
                    {"if": {"filter_query": "{entry_verdict} = '+EV'", "column_id": "entry_verdict"},
                     "color": "#3fb950", "fontWeight": "bold"},
                    {"if": {"filter_query": "{entry_verdict} = '-EV'", "column_id": "entry_verdict"},
                     "color": "#f85149"}])
            note = html.Div("2–6 leg PrizePicks entries ranked by EV (Power payouts, "
                            "correlation-aware). Leg probabilities are market-shrunk "
                            "(calibration placeholder), so EVs are honest: near break-even "
                            "is expected — real +EV needs genuinely soft lines.",
                            style={"color": "#8b949e", "fontSize": "12px", "margin": "8px 0"})
            return html.Div([note, tbl]), {"display": "none"}

        if tab == "health":
            fam = da.get_family_metrics()
            recs = da.get_recommendations()
            fi = da.get_feature_importance()
            fcols = [c for c in ["family", "distribution", "MAE", "calib_method",
                                 "brier_raw", "brier_chosen"] if not fam.empty and c in fam.columns]
            fam_tbl = dash_table.DataTable(
                data=fam.to_dict("records") if not fam.empty else [],
                columns=[{"name": c, "id": c} for c in fcols],
                sort_action="native", style_header=_HDR, style_cell=_CELL,
                style_data_conditional=[
                    {"if": {"filter_query": "{calib_method} != 'raw'", "column_id": "calib_method"},
                     "color": "#58a6ff"}])
            rec_tbl = dash_table.DataTable(
                data=recs.to_dict("records") if not recs.empty else [],
                columns=[{"name": c, "id": c} for c in ["priority", "family", "recommendation"]
                         if not recs.empty and c in recs.columns],
                style_header=_HDR, style_cell=_CELL,
                style_data_conditional=[
                    {"if": {"filter_query": "{priority} = 'high'", "column_id": "priority"},
                     "color": "#f85149", "fontWeight": "bold"},
                    {"if": {"filter_query": "{priority} = 'medium'", "column_id": "priority"},
                     "color": "#d29922"}])
            empty = html.Div("No model-health run yet. Run:  ./.venv/bin/python model_health.py",
                             style={"color": "#8b949e", "margin": "8px 0"})
            children = [html.H3("Per-family models + calibration", style={"color": "#e6edf3"})]
            children += [fam_tbl if not fam.empty else empty]
            children += [html.H3("Improvement recommendations",
                                 style={"color": "#e6edf3", "marginTop": "16px"}),
                         rec_tbl if not recs.empty else html.Div()]
            if not fi.empty:
                imp = fi.groupby("family")["importance"].max().reset_index()
                children += [html.H3("Top feature importance (per family)",
                                     style={"color": "#e6edf3", "marginTop": "16px"}),
                             dash_table.DataTable(
                                 data=fi.to_dict("records"),
                                 columns=[{"name": c, "id": c} for c in ["family", "feature", "importance"]],
                                 sort_action="native", page_size=20,
                                 style_header=_HDR, style_cell=_CELL)]
            return html.Div(children), {"display": "none"}

        if tab == "player":
            scan = da.get_scan()
            cards = []
            sub = scan[scan["proj_mean"].notna()] if ("proj_mean" in scan.columns and not scan.empty) else scan.iloc[0:0]
            for r in sub.head(6).to_dict("records"):
                book_imp = None
                try:
                    book_imp = round(1 / r["best_odds_decimal"], 3) if r.get("best_odds_decimal") else None
                except Exception:
                    pass
                header = html.Div([
                    html.Span(f"{r['player_name']}", style={"fontWeight": 700, "fontSize": "15px"}),
                    html.Span(f"  {r['market_type']} {r['side']} {r['line_value']}",
                              style={"color": "#8b97a7", "marginLeft": "6px"}),
                    html.Span(badge_pills(r), style={"marginLeft": "10px"})])
                chart = distribution_figure(r["proj_mean"], r.get("proj_sd") or 5.0,
                                            r["line_value"],
                                            f"simulated {r['market_type']} distribution")
                mets = html.Div([
                    metric("model P(over)", r.get("model_fair_prob")),
                    metric("anchor (sharp) P(over)", r.get("anchor_fair_prob")),
                    metric("best book", f"{r.get('best_book')} {r.get('best_odds_american')} (imp {book_imp})"),
                    metric("anchor edge", r.get("anchor_edge")),
                    metric("standout / confidence", f"{r.get('standout_score')} / {r.get('confidence_score')}"),
                    metric("why", r.get("reason_tags_json"))])
                cards.append(html.Div(className="panel", style={"marginBottom": "12px"},
                                      children=[header, html.Div(style={"display": "flex",
                                      "gap": "16px", "alignItems": "center"},
                                      children=[html.Div(chart, style={"flex": "1.3"}),
                                                html.Div(mets, style={"flex": "1"})])]))
            body = cards or [html.Div("No projectable props in the current scan.", className="note")]
            return html.Div([html.H3("Player / Prop Detail — top standouts", className="section"),
                             html.Div("Simulated outcome distribution (line dashed, model mean "
                                      "solid) with model vs sharp-anchor vs book. Click rows on "
                                      "Soft Lines for any prop.", className="note"), *body]), {"display": "none"}

        if tab == "validation":
            fam = da.get_family_metrics()
            rows = []
            for r in (fam.to_dict("records") if not fam.empty else []):
                b = r.get("brier_chosen", 1)
                calibd = r.get("calib_method", "raw") != "raw"
                status = ("Validated" if b < 0.18 else "Promising" if b < 0.21
                          else "Noisy" if b < 0.24 else "Model-only")
                kind = {"Validated": "val", "Promising": "stale", "Noisy": "noisy",
                        "Model-only": "noisy"}.get(status, "noisy")
                rows.append({**r, "validation": status, "_kind": kind})
            tbl = dash_table.DataTable(
                data=[{k: v for k, v in r.items() if k != "_kind"} for r in rows],
                columns=[{"name": c, "id": c} for c in
                         ["family", "distribution", "MAE", "calib_method", "brier_chosen", "validation"]],
                sort_action="native", style_header=_HDR, style_cell=_CELL,
                style_data_conditional=[
                    {"if": {"filter_query": "{validation} = 'Validated'", "column_id": "validation"},
                     "color": "#3fd17a", "fontWeight": "bold"},
                    {"if": {"filter_query": "{validation} = 'Model-only'", "column_id": "validation"},
                     "color": "#ff5d6c"}])
            note = html.Div("Per-family validation from out-of-time calibration (Brier). "
                            "Overall CLV backtest vs historical closing lines was ~break-even "
                            "(+0.4%, 52% beat close, small sample) — single-prop edges are not "
                            "yet proven; treat 'Validated' as well-calibrated, not profitable.",
                            className="note")
            return html.Div([html.H3("Validation", className="section"), note,
                             tbl if rows else html.Div("Run model_health.py first.", className="note")]), {"display": "none"}

        if tab == "sources":
            sh = da.get_source_health()
            if sh.empty:
                return html.Div([html.H3("Source Health", className="section"),
                                 html.Div("No snapshot yet. Run:  ./.venv/bin/python -m "
                                          "feeds.health_snapshot", className="note")]), {"display": "none"}
            tbl = dash_table.DataTable(
                data=sh.to_dict("records"),
                columns=[{"name": c, "id": c} for c in
                         ["source", "tier", "viable", "rows", "latency_s", "note"] if c in sh.columns],
                style_header=_HDR, style_cell=_CELL,
                style_data_conditional=[
                    {"if": {"filter_query": "{viable} = 1", "column_id": "viable"}, "color": "#3fd17a"},
                    {"if": {"filter_query": "{viable} = 0", "column_id": "viable"}, "color": "#ff5d6c"}])
            return html.Div([html.H3("Source Health", className="section"),
                             html.Div("Viable continuous stack = X + Odds API + ESPN + PrizePicks "
                                      "(official/public). Rotowire + SGP are research-only.",
                                      className="note"), tbl]), {"display": "none"}

        if tab == "pairs":
            p = da.get_pairs()
            cols = [c for c in ["team", "player_a", "player_b", "rho", "p_independent",
                                "p_joint", "mispricing", "direction"] if not p.empty and c in p.columns]
            tbl = dash_table.DataTable(
                data=p.to_dict("records") if not p.empty else [],
                columns=[{"name": c, "id": c} for c in cols],
                style_header={"backgroundColor": "#0d1117", "color": "#8b949e"},
                style_cell={"backgroundColor": "#161b22", "color": "#e6edf3",
                            "fontSize": "12px", "fontFamily": "monospace"})
            note = html.Div("Dual-over OVERPRICED = book pricing near independence misses "
                            "negative teammate correlation (fade / split candidate).",
                            style={"color": "#8b949e", "fontSize": "12px", "margin": "8px 0"})
            return html.Div([note, tbl]), {"display": "none"}
        # news
        n = da.get_recent_news()
        items = [html.Div([html.Span(f'[{row["parsed_status"]}] ',
                 style={"color": "#d29922", "fontWeight": 700}),
                 html.Span(row["player_name"] or "?", style={"color": "#58a6ff"}),
                 html.Span(f'  {row["raw_text"][:120]}', style={"color": "#8b949e"})],
                 style={"fontSize": "12px", "marginBottom": "6px"})
                 for _, row in n.iterrows()] if not n.empty else [html.Div("no recent news")]
        return html.Div(items), {"display": "none"}
