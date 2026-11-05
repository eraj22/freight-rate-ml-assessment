# Model card

## Intended use

Estimate the posted freight rate for loads represented by the supplied schema. The model is designed for batch scoring of the November-December 2025 validation data, not autonomous pricing decisions.

## Model

A three-seed CatBoost regression ensemble predicts `log1p(posted_rate)`. Predictions are transformed back to dollars and averaged. CatBoost was selected because it handles nonlinear effects, mixed numerical and categorical data, missing values, and unseen category combinations without a fragile one-hot encoding pipeline.

## Validation design

The data is ordered over time. September 2025 was used as the development fold. After the approach and hyperparameters were frozen, October 2025 was kept as the final holdout. This mirrors the real task of predicting later months and avoids future-to-past leakage.

## Important features

The pipeline uses distance, equipment, origin, destination, lane, coordinates, weight, market index, quote signal, calendar features, cyclic seasonal features, and physically meaningful interactions such as distance multiplied by quote and market signals.

## Data-quality handling

- Missing weight and market-index values receive explicit missingness indicators. CatBoost handles their remaining null values.
- Negative weights are treated as sign errors: their absolute values are used and a separate flag preserves the anomaly signal.
- Training IDs must be unique, distance and targets must be positive, and output IDs must exactly match the supplied template.
- Extreme target values are retained. The log target limits their influence without silently deleting potentially valid expensive loads.

## Limitations and monitoring

The labeled data ends in October 2025. Market regimes, fuel costs, and new geographies may drift after that point. Monitor MAE and error by equipment and lane once actuals arrive. Retrain when error materially degrades or the feature distribution changes. A human should review unusually large quotes and loads outside the observed operating range.
