"""Central configuration for the WNBA points-prediction pipeline.

All tunables live here so the rest of the code stays declarative. Values can be
overridden via a `.env` file (loaded with python-dotenv) without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
for _d in (DATA_DIR, MODELS_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Load environment overrides (optional .env at project root)
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = Path(os.getenv("WNBA_DB_PATH", DATA_DIR / "wnba.sqlite"))
DB_URL = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# ESPN public (unofficial) endpoints
# ---------------------------------------------------------------------------
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
ESPN_SCOREBOARD = ESPN_BASE + "/scoreboard"      # ?dates=YYYYMMDD
ESPN_SUMMARY = ESPN_BASE + "/summary"            # ?event=<eventId>

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
HTTP_TIMEOUT = int(os.getenv("WNBA_HTTP_TIMEOUT", "30"))
HTTP_RETRIES = int(os.getenv("WNBA_HTTP_RETRIES", "3"))
HTTP_BACKOFF = float(os.getenv("WNBA_HTTP_BACKOFF", "1.5"))
# Polite delay between ESPN requests (seconds) to avoid hammering the endpoint.
REQUEST_PAUSE = float(os.getenv("WNBA_REQUEST_PAUSE", "0.4"))

# ---------------------------------------------------------------------------
# Season / ingestion window
# ---------------------------------------------------------------------------
# WNBA regular seasons roughly run mid-May through September.
DEFAULT_SEASONS = [int(s) for s in os.getenv("WNBA_SEASONS", "2024,2025").split(",")]
SEASON_START_MMDD = (5, 1)    # inclusive scan start each season (month, day)
SEASON_END_MMDD = (10, 15)    # inclusive scan end each season

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
TARGET = "points"
ROLL_WINDOWS = (3, 5)         # rolling windows used for points/minutes
MIN_HISTORY_GAMES = 3         # rows with fewer prior games are dropped for training
TEAM_PROXY_WINDOW = 10        # rolling window for opponent proxies
MAX_DAYS_REST = 14            # cap days_rest to avoid out-of-distribution blowup

# Columns the model trains on (must exist in model_features table).
FEATURE_COLUMNS = [
    "last_3_points",
    "last_5_points",
    "last_3_minutes",
    "last_5_minutes",
    "season_avg_points",
    "season_avg_minutes",
    "home_away_flag",
    "days_rest",
    "opponent_points_allowed_proxy",
    "opponent_pace_proxy",
]

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
# Fraction of the date range (chronological) used for the test split.
TEST_FRACTION = float(os.getenv("WNBA_TEST_FRACTION", "0.2"))
RANDOM_STATE = 42
RF_PARAMS = dict(n_estimators=300, max_depth=12, min_samples_leaf=5,
                 random_state=RANDOM_STATE, n_jobs=-1)
RIDGE_PARAMS = dict(alpha=1.0)
MODEL_PATH = MODELS_DIR / "points_model.joblib"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("WNBA_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
