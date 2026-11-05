# 2-3 minute Loom walkthrough

Camera on. Share the repository README first, then `src/freight_model.py`, `artifacts/validation_metrics.json`, and the December chart.

## 0:00-0:25 - Goal and data

"This project predicts posted freight rates for 12,000 November and December loads. The labeled data has 48,000 rows from January through October, so I treated this as a forward-looking time problem rather than using a random split."

## 0:25-0:55 - Exploration and data quality

"Distance is the dominant raw driver, but equipment, weight, origin and destination, market index, quote signal, and seasonality also matter. I found no duplicate rows or IDs. There are 300 missing training weights, 374 missing market-index values, and 292 negative weights. I preserve missingness as features, let CatBoost handle nulls, and repair negative weights with an absolute value plus an anomaly flag. The final set also contains eight unseen cities and 736 unseen lanes, so the model cannot depend on lane memorization."

## 0:55-1:30 - Model and features

"I chose CatBoost because it handles categorical variables, nonlinear interactions, missing values, and unseen category combinations well. I predict the log of the posted rate to reduce the influence of rare extreme rates. The final model averages three deterministic seeds. Features include distance, equipment, route, coordinates, cleaned weight, market and quote signals, calendar and cyclic seasonality, and distance-signal interactions."

## 1:30-2:00 - Validation

"September was the development fold. I froze the pipeline and used October as the final holdout, which simulates predicting a future month and prevents temporal leakage. The report shows MAE, RMSE, MAPE, R-squared, and a simple historical rate-per-mile baseline. The ensemble cuts MAE by roughly half versus that baseline."

## 2:00-2:35 - Code and outputs

"`train_predict.py` validates schemas, trains on all labeled rows, aligns predictions to the supplied ID template, and creates the fixed December file. `evaluate.py` reproduces the holdout metrics and feature importance. `score.py` then validates all 12,000 IDs, positivity, the fixed 31-day December scenario, and generates this chart. Unit tests cover the most important cleaning, feature, schema, and metric logic."

## 2:35-2:50 - Close

"The repository includes exact setup commands, the final prediction CSV, report, chart, model card, data-quality review, tests, and CI. The main production caveat is market drift, so I would monitor error by equipment and lane and retrain once new actuals arrive."
