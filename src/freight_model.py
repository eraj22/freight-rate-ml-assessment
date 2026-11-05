from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


TARGET = "posted_rate"
ID_COLUMN = "load_id"
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment", "lane"]
BASE_FEATURE_COLUMNS = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]


@dataclass(frozen=True)
class ModelConfig:
    """Frozen hyperparameters selected with a September development fold."""

    iterations: int = 400
    depth: int = 8
    learning_rate: float = 0.07
    l2_leaf_reg: float = 8.0
    seeds: tuple[int, ...] = (42, 137, 2026)


def validate_input(frame: pd.DataFrame, *, require_target: bool) -> None:
    """Fail early on schema or integrity problems that would corrupt training."""

    required = set(BASE_FEATURE_COLUMNS) | ({TARGET} if require_target else set())
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if ID_COLUMN in frame:
        if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
            raise ValueError("load_id must be present and unique")
    if (pd.to_numeric(frame["distance"], errors="coerce") <= 0).any():
        raise ValueError("distance must be positive")
    if require_target:
        target = pd.to_numeric(frame[TARGET], errors="coerce")
        if target.isna().any() or (target <= 0).any():
            raise ValueError("posted_rate must contain finite positive values")


def enrich_sparse_inputs(frame: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """Add fields omitted from the fixed December scenario.

    City coordinates come from the labeled data. Market and quote signals use
    the latest 30-day median, a leakage-free operational assumption for future
    rows where those signals are unavailable.
    """

    result = frame.copy()
    dates = pd.to_datetime(training["date"], errors="raise")
    latest_date = dates.max()
    recent = training.loc[dates >= latest_date - pd.Timedelta(days=29)]

    for column in ["market_index", "quote_signal"]:
        if column not in result:
            result[column] = float(recent[column].median())
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(
                float(recent[column].median())
            )

    coordinate_specs = [
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ]
    for city_column, lat_column, lon_column in coordinate_specs:
        lookup = training.groupby(city_column, observed=True)[[lat_column, lon_column]].median()
        for coordinate in [lat_column, lon_column]:
            mapped = result[city_column].map(lookup[coordinate])
            if coordinate not in result:
                result[coordinate] = mapped
            else:
                result[coordinate] = pd.to_numeric(result[coordinate], errors="coerce").fillna(mapped)
            result[coordinate] = result[coordinate].fillna(float(training[coordinate].median()))

    return result


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-free model features with deterministic column order."""

    result = frame.drop(columns=[ID_COLUMN, TARGET, "predicted_rate"], errors="ignore").copy()
    dates = pd.to_datetime(result.pop("date"), errors="raise")

    result["lane"] = result["pickup"].astype(str) + " -> " + result["delivery"].astype(str)
    result["year"] = dates.dt.year.astype(int)
    result["month"] = dates.dt.month.astype(int)
    result["day"] = dates.dt.day.astype(int)
    result["day_of_week"] = dates.dt.day_of_week.astype(int)
    result["day_of_year"] = dates.dt.day_of_year.astype(int)
    result["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    result["days_since_start"] = (dates - pd.Timestamp("2025-01-01")).dt.days.astype(int)
    result["is_weekend"] = (dates.dt.day_of_week >= 5).astype(int)
    result["annual_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["annual_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)
    result["weekly_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["weekly_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)

    raw_weight = pd.to_numeric(result["weight"], errors="coerce")
    result["weight_missing"] = raw_weight.isna().astype(int)
    result["weight_was_negative"] = (raw_weight < 0).astype(int)
    result["weight"] = raw_weight.abs()
    result["log_weight"] = np.log1p(result["weight"])
    result["market_index_missing"] = result["market_index"].isna().astype(int)

    result["distance_x_quote"] = result["distance"] * result["quote_signal"]
    result["distance_x_market"] = result["distance"] * result["market_index"]
    result["distance_x_quote_x_market"] = (
        result["distance"] * result["quote_signal"] * result["market_index"]
    )
    result["log_distance"] = np.log1p(result["distance"])
    result["lat_delta"] = result["delivery_lat"] - result["pickup_lat"]
    result["lon_delta"] = result["delivery_lon"] - result["pickup_lon"]
    result["abs_lat_delta"] = result["lat_delta"].abs()
    result["abs_lon_delta"] = result["lon_delta"].abs()

    for column in CATEGORICAL_COLUMNS:
        result[column] = result[column].fillna("__MISSING__").astype(str)
    return result


def build_model(config: ModelConfig, seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        l2_leaf_reg=config.l2_leaf_reg,
        loss_function="RMSE",
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def fit_ensemble(
    training: pd.DataFrame,
    config: ModelConfig = ModelConfig(),
) -> tuple[list[CatBoostRegressor], pd.DataFrame]:
    validate_input(training, require_target=True)
    features = make_features(training)
    log_target = np.log1p(training[TARGET].astype(float))
    models = []
    for seed in config.seeds:
        model = build_model(config, seed)
        model.fit(features, log_target, cat_features=CATEGORICAL_COLUMNS)
        models.append(model)
    return models, features


def predict_ensemble(models: Sequence[CatBoostRegressor], frame: pd.DataFrame) -> np.ndarray:
    features = make_features(frame)
    expected = list(models[0].feature_names_)
    missing = sorted(set(expected) - set(features.columns))
    extra = sorted(set(features.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"prediction feature mismatch (missing={missing}, extra={extra})")
    features = features.loc[:, expected]
    predictions = [np.expm1(model.predict(features)) for model in models]
    return np.maximum(np.mean(predictions, axis=0), 1.0)


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    error = actual_array - predicted_array
    absolute_error = np.abs(error)
    ss_residual = float(np.sum(error**2))
    ss_total = float(np.sum((actual_array - actual_array.mean()) ** 2))
    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - ss_residual / ss_total),
        "mape_pct": float(np.mean(absolute_error / actual_array) * 100),
        "median_absolute_error": float(np.median(absolute_error)),
        "p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
    }


def mean_feature_importance(
    models: Sequence[CatBoostRegressor], feature_names: Sequence[str]
) -> pd.DataFrame:
    values = np.mean([model.get_feature_importance() for model in models], axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
