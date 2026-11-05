from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.freight_model import (
    ID_COLUMN,
    ModelConfig,
    enrich_sparse_inputs,
    fit_ensemble,
    mean_feature_importance,
    predict_ensemble,
    validate_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final model and create required predictions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--predictions", type=Path, default=Path("validation_predictions.csv"))
    parser.add_argument(
        "--december-predictions",
        type=Path,
        default=Path("data/december_chart_inputs.csv"),
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = pd.read_csv(args.data_dir / "train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(args.data_dir / "validation.csv", parse_dates=["date"])
    template = pd.read_csv(args.data_dir / "validation_predictions_template.csv")
    december = pd.read_csv(args.data_dir / "december_chart_inputs_source.csv", parse_dates=["date"])

    validate_input(training, require_target=True)
    validate_input(validation, require_target=False)
    if list(template.columns) != [ID_COLUMN, "predicted_rate"]:
        raise ValueError("prediction template must contain load_id,predicted_rate in that order")
    if template[ID_COLUMN].duplicated().any() or set(template[ID_COLUMN]) != set(validation[ID_COLUMN]):
        raise ValueError("prediction template IDs do not match validation IDs")

    config = ModelConfig()
    models, training_features = fit_ensemble(training, config)
    validation_prediction = predict_ensemble(models, validation)
    prediction_lookup = pd.Series(validation_prediction, index=validation[ID_COLUMN])
    completed = template[[ID_COLUMN]].copy()
    completed["predicted_rate"] = completed[ID_COLUMN].map(prediction_lookup)
    if completed["predicted_rate"].isna().any() or (completed["predicted_rate"] <= 0).any():
        raise ValueError("generated validation predictions are missing or non-positive")

    december_enriched = enrich_sparse_inputs(december, training)
    december_prediction = predict_ensemble(models, december_enriched)
    december_output = december.copy()
    december_output["date"] = december_output["date"].dt.strftime("%Y-%m-%d")
    december_output["predicted_rate"] = december_prediction

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.december_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    completed.to_csv(args.predictions, index=False, float_format="%.2f")
    december_output.to_csv(args.december_predictions, index=False, float_format="%.2f")
    mean_feature_importance(models, training_features.columns).to_csv(
        args.artifacts_dir / "final_feature_importance.csv", index=False
    )
    run_summary = {
        "training_rows": len(training),
        "validation_rows": len(validation),
        "december_rows": len(december_output),
        "model": "three-seed CatBoost ensemble on log1p(posted_rate)",
        "seeds": list(config.seeds),
    }
    (args.artifacts_dir / "prediction_run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
