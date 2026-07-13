"""Single-page layout (Section 8): topbar, KPIs, filters, tabs, detail panel."""

from __future__ import annotations

from dash import dash_table, dcc, html

from . import data_access as da
from .components import TABLE_COLUMNS, TABLE_STYLE, kpi_card

PAGE = {"backgroundColor": "#0d1117", "minHeight": "100vh", "padding": "16px 20px",
        "fontFamily": "Inter, system-ui, sans-serif"}
ROW = {"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"}
CTRL = {"background": "#161b22", "color": "#e6edf3", "border": "1px solid #30363d",
        "borderRadius": "6px", "padding": "6px"}


def _dropdown(id_, options, value):
    return dcc.Dropdown(id=id_, options=options, value=value, clearable=False,
                        style={"width": "150px"}, className="dark-dd")


def serve_layout():
    teams = da.teams_list()
    markets = da.markets_list()
    return html.Div(style=PAGE, children=[
        dcc.Interval(id="refresh-interval", interval=45_000, n_intervals=0),
        dcc.Store(id="scan-store"),

        # ---- top bar ----
        html.Div(className="topbar", children=[
            html.Div([html.Span(className="brand-dot"),
                      html.Span("WNBA Prop Engine", className="brand"),
                      html.Span(id="last-scan", style={"marginLeft": "14px",
                                "color": "#8b97a7", "fontSize": "12px"})]),
            html.Div([
                html.Button("↻ Refresh", id="refresh-btn", n_clicks=0, className="btn"),
                dcc.Checklist(id="autorefresh", options=[{"label": " auto", "value": "on"}],
                              value=["on"], style={"display": "inline-block",
                              "marginLeft": "10px", "color": "#8b97a7"}),
            ]),
        ]),

        # ---- KPI cards ----
        html.Div(id="kpi-row", style=ROW),

        # ---- filters ----
        html.Div(style=ROW, children=[
            _dropdown("f-team", [{"label": t, "value": t} for t in teams], "ALL"),
            _dropdown("f-market", [{"label": m, "value": m} for m in markets], "ALL"),
            html.Div(["min standout ",
                      dcc.Input(id="f-standout", type="number", value=15, min=0, max=100,
                                step=5, style={**CTRL, "width": "70px"})],
                     style={"color": "#8b949e", "fontSize": "12px"}),
            html.Div(["min conf ",
                      dcc.Input(id="f-conf", type="number", value=0, min=0, max=100,
                                step=5, style={**CTRL, "width": "70px"})],
                     style={"color": "#8b949e", "fontSize": "12px"}),
            dcc.Checklist(id="f-toggles", options=[
                {"label": " news", "value": "news"}, {"label": " stale", "value": "stale"},
                {"label": " pair", "value": "pair"}, {"label": " movers", "value": "movers"}],
                value=[], inline=True, style={"color": "#8b949e", "fontSize": "12px"},
                inputStyle={"marginLeft": "10px", "marginRight": "3px"}),
        ]),

        # ---- tabs ----
        dcc.Tabs(id="tabs", value="singles", className="dash-tabs", children=[
            dcc.Tab(label="Soft Lines", value="singles", className="dash-tab",
                    selected_className="dash-tab--selected"),
            dcc.Tab(label="DFS Entries", value="entries", className="dash-tab",
                    selected_className="dash-tab--selected"),
            dcc.Tab(label="Player Detail", value="player", className="dash-tab",
                    selected_className="dash-tab--selected"),
            dcc.Tab(label="PrizePicks vs Sharp", value="prizepicks", className="dash-tab",
                    selected_className="dash-tab--selected"),
            dcc.Tab(label="Validation", value="validation", className="dash-tab",
                    selected_className="dash-tab--selected"),
            dcc.Tab(label="Model Health", value="health", className="dash-tab",
                    selected_className="dash-tab--selected"),
            dcc.Tab(label="Source Health", value="sources", className="dash-tab",
                    selected_className="dash-tab--selected"),
            dcc.Tab(label="Correlated Pairs", value="pairs", className="dash-tab",
                    selected_className="dash-tab--selected"),
            dcc.Tab(label="Recent News", value="news", className="dash-tab",
                    selected_className="dash-tab--selected"),
        ], style={"marginBottom": "12px"}),

        html.Div(id="tab-content"),

        # main singles table + detail panel (shown on the singles tab)
        html.Div(id="singles-view", children=[
            html.Div(style={"display": "flex", "gap": "12px"}, children=[
                html.Div(style={"flex": "3"}, children=[
                    dash_table.DataTable(
                        id="scan-table", columns=TABLE_COLUMNS, data=[],
                        sort_action="native", row_selectable="single",
                        page_size=40, **TABLE_STYLE),
                ]),
                html.Div(id="detail-panel", style={"flex": "1.4", "background": "#161b22",
                         "border": "1px solid #30363d", "borderRadius": "8px",
                         "padding": "12px", "minHeight": "300px", "color": "#e6edf3"}),
            ]),
        ]),
    ])
