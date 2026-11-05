from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.freight_model import (
    ModelConfig,
    fit_ensemble,
    mean_feature_importance,
    predict_ensemble,
    regression_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen October 2025 holdout evaluation.")
    parser.add_argument("--training", type=Path, default=Path("data/train_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.training, parse_dates=["date"])
    cutoff = pd.Timestamp("2025-10-01")
    training = frame.loc[frame["date"] < cutoff].copy()
    holdout = frame.loc[frame["date"] >= cutoff].copy()
    if training.empty or holdout.empty:
        raise ValueError("chronological split produced an empty partition")

    config = ModelConfig()
    models, training_features = fit_ensemble(training, config)
    prediction = predict_ensemble(models, holdout)
    model_metrics = regression_metrics(holdout["posted_rate"], prediction)

    baseline_rpm = float(np.median(training["posted_rate"] / training["distance"]))
    baseline_prediction = baseline_rpm * holdout["distance"].to_numpy()
    baseline_metrics = regression_metrics(holdout["posted_rate"], baseline_prediction)

    diagnostic = holdout[["load_id", "date", "equipment", "posted_rate"]].copy()
    diagnostic["prediction"] = prediction
    diagnostic["absolute_error"] = np.abs(diagnostic["posted_rate"] - diagnostic["prediction"])
    error_by_equipment = (
        diagnostic.groupby("equipment", observed=True)
        .agg(rows=("load_id", "size"), mae=("absolute_error", "mean"), median_ae=("absolute_error", "median"))
        .reset_index()
    )

    metrics = {
        "split": {
            "training_period": f"{training['date'].min().date()} to {training['date'].max().date()}",
            "holdout_period": f"{holdout['date'].min().date()} to {holdout['date'].max().date()}",
            "training_rows": int(len(training)),
            "holdout_rows": int(len(holdout)),
        },
        "model": model_metrics,
        "baseline": baseline_metrics,
        "mae_improvement_pct": float((baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"] * 100),
        "configuration": {
            "target": "log1p(posted_rate)",
            "algorithm": "three-seed CatBoost ensemble",
            "iterations": config.iterations,
            "depth": config.depth,
            "learning_rate": config.learning_rate,
            "seeds": list(config.seeds),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    diagnostic.to_csv(args.output_dir / "october_holdout_predictions.csv", index=False)
    error_by_equipment.to_csv(args.output_dir / "error_by_equipment.csv", index=False)
    mean_feature_importance(models, training_features.columns).to_csv(
        args.output_dir / "feature_importance.csv", index=False
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
