"""Train and evaluate the points model (Ridge baseline + RandomForest).

  * Drops rows with insufficient prior history (config.MIN_HISTORY_GAMES).
  * Splits train/test chronologically by game_date (never random).
  * Median-imputes features and persists the medians so prediction matches.
  * Evaluates MAE + RMSE; saves the best model (lowest test RMSE) via joblib.

CLI:
  python -m src.train
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, database
from .utils import get_logger

log = get_logger(__name__)


def load_training_frame(engine) -> pd.DataFrame:
    """Completed games with enough history to have meaningful rolling features."""
    df = database.read_sql(engine, "SELECT * FROM model_features")
    if df.empty:
        return df
    df = df[df["points"].notna()]
    df = df[df["history_games"] >= config.MIN_HISTORY_GAMES]
    return df.sort_values("game_date").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_fraction: float):
    """Split by date so the test set is strictly the most recent games."""
    dates = np.sort(df["game_date"].unique())
    cut_idx = int(len(dates) * (1 - test_fraction))
    cut_date = dates[max(cut_idx - 1, 0)]
    train = df[df["game_date"] <= cut_date]
    test = df[df["game_date"] > cut_date]
    if test.empty:  # tiny datasets: fall back to a row-based tail split
        n = max(1, int(len(df) * test_fraction))
        train, test = df.iloc[:-n], df.iloc[-n:]
        cut_date = train["game_date"].max()
    return train, test, cut_date


def _evaluate(model, X, y) -> dict[str, float]:
    pred = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {"MAE": float(mean_absolute_error(y, pred)), "RMSE": rmse}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train WNBA points model.")
    parser.add_argument("--test-fraction", type=float, default=config.TEST_FRACTION)
    args = parser.parse_args(argv)

    engine = database.init_db()
    df = load_training_frame(engine)
    if len(df) < 50:
        log.error("Only %d usable rows. Ingest more history before training.", len(df))
        return

    feats = config.FEATURE_COLUMNS
    medians = df[feats].median(numeric_only=True)
    X_all = df[feats].fillna(medians)
    y_all = df[config.TARGET].astype(float)

    train, test, cut_date = chronological_split(df, args.test_fraction)
    Xtr, ytr = X_all.loc[train.index], y_all.loc[train.index]
    Xte, yte = X_all.loc[test.index], y_all.loc[test.index]
    log.info("Train=%d (<= %s)  Test=%d (> %s)", len(train), cut_date, len(test), cut_date)

    candidates = {
        "ridge_baseline": Ridge(**config.RIDGE_PARAMS),
        "random_forest": RandomForestRegressor(**config.RF_PARAMS),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        metrics = _evaluate(model, Xte, yte)
        results[name] = metrics
        log.info("%-16s  MAE=%.3f  RMSE=%.3f", name, metrics["MAE"], metrics["RMSE"])

    best_name = min(results, key=lambda n: results[n]["RMSE"])
    best_model = candidates[best_name]
    # refit the winner on ALL data so prediction uses every available game
    best_model.fit(X_all, y_all)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "features": feats,
        "medians": medians.to_dict(),
        "target": config.TARGET,
        "metrics": results[best_name],
    }
    joblib.dump(bundle, config.MODEL_PATH)
    log.info("Best=%s (RMSE=%.3f). Saved -> %s",
             best_name, results[best_name]["RMSE"], config.MODEL_PATH)

    pd.DataFrame(results).T.reset_index(names="model").to_csv(
        config.OUTPUT_DIR / "model_metrics.csv", index=False)


if __name__ == "__main__":
    main()
