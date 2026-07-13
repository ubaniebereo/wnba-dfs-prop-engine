#!/usr/bin/env python3
"""Launch the WNBA Prop Engine dashboard (http://127.0.0.1:8050)."""

from dashboard_app import app

if __name__ == "__main__":
    print("WNBA Prop Engine dashboard -> http://127.0.0.1:8050  (Ctrl-C to stop)")
    app.run(host="127.0.0.1", port=8050, debug=False)
