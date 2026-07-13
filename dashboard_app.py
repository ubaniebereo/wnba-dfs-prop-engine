"""WNBA Prop Engine — interactive Dash dashboard (entry point).

Reads precomputed scan results from feeds.sqlite (written by scanner/run_live_scan).
The UI never trains models or polls odds — it only renders. Run the scanner
separately (run_daily_scan.py) to keep the board fresh.

  python run_dashboard.py      # serves http://127.0.0.1:8050
"""

from __future__ import annotations

import dash

from dashboard import callbacks
from dashboard.layout import serve_layout

app = dash.Dash(__name__, title="WNBA Prop Engine", suppress_callback_exceptions=True)
app.layout = serve_layout          # callable -> re-evaluated on each page load
callbacks.register(app)
server = app.server                # for gunicorn/hosting


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
