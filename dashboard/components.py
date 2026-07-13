"""Reusable UI pieces + table styling for the Dash board (Section 8)."""

from __future__ import annotations

from dash import dcc, html

ACCENT = "#5b9dff"


def kpi_card(label: str, value, sub: str = "", tone: str = "") -> html.Div:
    """Premium KPI card; tone in {'', 'good', 'warn', 'bad'} colors the value."""
    return html.Div(className=f"kpi {tone}".strip(), children=[
        html.Div(label, className="lbl"),
        html.Div(str(value), className="val"),
        html.Div(sub, className="sub"),
    ])


def pill(text: str, kind: str) -> html.Span:
    return html.Span(text, className=f"badge badge-{kind}")


def badge_pills(row: dict) -> list:
    out = []
    if row.get("news_flag"):
        out.append(pill("NEWS", "news"))
    if row.get("stale_vs_anchor"):
        out.append(pill("STALE", "stale"))
    if row.get("line_move_flag"):
        out.append(pill("MOVE", "move"))
    if row.get("pair_flag"):
        out.append(pill("PAIR", "pair"))
    return out


def metric(k: str, v) -> html.Div:
    return html.Div(className="metric-row", children=[
        html.Span(k, className="k"), html.Span(str(v), className="v")])


def distribution_figure(mean: float, sd: float, line: float, title: str):
    """Clean normal-density chart: line marked, over/under shaded."""
    import numpy as np
    import plotly.graph_objects as go
    xs = np.linspace(max(0, mean - 3.2 * sd), mean + 3.2 * sd, 200)
    ys = np.exp(-0.5 * ((xs - mean) / max(sd, 1e-6)) ** 2)
    over = xs >= line
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs[~over], y=ys[~over], fill="tozeroy", mode="none",
                             fillcolor="rgba(255,93,108,.25)"))
    fig.add_trace(go.Scatter(x=xs[over], y=ys[over], fill="tozeroy", mode="none",
                             fillcolor="rgba(63,209,122,.30)"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#9db7ff", width=2)))
    fig.add_vline(x=line, line=dict(color="#f5b945", width=2, dash="dash"))
    fig.add_vline(x=mean, line=dict(color="#5b9dff", width=1))
    fig.update_layout(template="plotly_dark", height=210,
                      title=dict(text=title, font=dict(size=12)),
                      margin=dict(l=10, r=10, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      yaxis=dict(visible=False), xaxis=dict(gridcolor="#262d3a"))
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def badges(row: dict) -> str:
    """Compact flag string for the main row."""
    b = []
    if row.get("news_flag"):
        b.append("NEWS")
    if row.get("stale_vs_anchor"):
        b.append("STALE")
    if row.get("line_move_flag"):
        b.append("MOVE")
    if row.get("pair_flag"):
        b.append("PAIR")
    return " ".join(b)


# Main DataTable columns (compact). 'flags' + 'standout_score' are the focus.
TABLE_COLUMNS = [
    {"name": "Player", "id": "player_name"},
    {"name": "Tm", "id": "team"},
    {"name": "Mkt", "id": "market_type"},
    {"name": "Side", "id": "side"},
    {"name": "Line", "id": "line_value"},
    {"name": "Book", "id": "best_book"},
    {"name": "Odds", "id": "best_odds_american"},
    {"name": "ModEdge", "id": "model_edge"},
    {"name": "AnchEdge", "id": "anchor_edge"},
    {"name": "Raw", "id": "standout_score"},
    {"name": "Final", "id": "final_standout_score"},
    {"name": "Verdict", "id": "final_verdict"},
    {"name": "Conf", "id": "decision_confidence"},
    {"name": "Flags", "id": "flags"},
]


def standout_heat_styles() -> list[dict]:
    """Green heat ramp on the Standout column."""
    ramp = [(15, "#1b3a2b"), (25, "#1f5132"), (35, "#2ea043"), (50, "#3fb950"),
            (70, "#56d364")]
    styles = [{"if": {"column_id": "standout_score", "filter_query": f"{{standout_score}} >= {lo}"},
               "backgroundColor": col, "color": "#0d1117", "fontWeight": "bold"}
              for lo, col in ramp]
    # low-confidence visual warning
    styles.append({"if": {"column_id": "confidence_score",
                          "filter_query": "{confidence_score} < 35"},
                   "color": "#f85149"})
    styles.append({"if": {"column_id": "anchor_edge", "filter_query": "{anchor_edge} >= 0.02"},
                   "color": "#3fb950", "fontWeight": "bold"})
    return styles


TABLE_STYLE = {
    "style_table": {"overflowX": "auto", "maxHeight": "62vh", "overflowY": "auto"},
    "style_header": {"backgroundColor": "#0d1117", "color": "#8b949e",
                     "fontWeight": "bold", "border": "1px solid #30363d", "fontSize": "12px"},
    "style_cell": {"backgroundColor": "#161b22", "color": "#e6edf3",
                   "border": "1px solid #21262d", "fontSize": "12px",
                   "padding": "4px 8px", "fontFamily": "ui-monospace, monospace"},
    "style_data_conditional": standout_heat_styles(),
}
