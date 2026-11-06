# Freight Rate Prediction Challenge

[![tests](https://github.com/eraj22/freight-rate-ml-assessment/actions/workflows/ci.yml/badge.svg)](https://github.com/eraj22/freight-rate-ml-assessment/actions/workflows/ci.yml)

Production-style solution for predicting posted freight rates on 12,000 future loads. The repository includes reproducible training and evaluation code, final predictions, the employer-provided scorer output, a polished assessment report, unit tests, CI, a model card, and a camera-on Loom script.

## Results

The model is evaluated on an untouched October 2025 holdout after training on January-September data.

| Metric | CatBoost ensemble | Historical rate-per-mile baseline |
| --- | ---: | ---: |
| MAE | **$123.25** | $253.85 |
| RMSE | **$647.62** | $695.05 |
| MAPE | **5.68%** | 11.81% |
| Median absolute error | **$39.73** | $132.82 |
| R-squared | **0.820** | 0.793 |

The ensemble reduces MAE by **51.45%** relative to the transparent baseline. The supplied scorer also validates all 12,000 final IDs and all 31 fixed December rows.

## Why this approach

- **Chronological validation:** September is the development fold and October is the frozen holdout. A random split would leak future market behavior into past training windows.
- **CatBoost:** Directly supports categorical routes, missing numeric values, nonlinear effects, and unseen category combinations.
- **Log target:** `log1p(posted_rate)` reduces the leverage of rare extreme rates without deleting rows.
- **Three-seed ensemble:** Seeds 42, 137, and 2026 reduce model variance while remaining deterministic.
- **Physical and market features:** Distance, equipment, geography, weight, market index, quote signal, cyclic calendar features, missingness flags, and distance-signal interactions.

## Data-quality findings

- No duplicate rows or duplicate IDs.
- 300 missing training weights and 165 missing final weights.
- 374 missing training market-index values and 249 missing final values.
- 292 negative training weights and 145 negative final weights, treated as sign errors using absolute value plus an anomaly flag.
- Eight unseen cities and 736 unseen lanes in the final data, so the model cannot rely on route memorization.
- Rare rates above $10,000 are retained and handled through the log target.

See [data_quality.md](docs/data_quality.md) and [model_card.md](docs/model_card.md) for details.

## Repository layout

```text
.
|-- data/
|   |-- train_test.csv
|   |-- validation.csv
|   |-- validation_predictions_template.csv
|   |-- december_chart_inputs_source.csv
|   `-- december_chart_inputs.csv
|-- src/freight_model.py
|-- tests/test_pipeline.py
|-- artifacts/
|-- docs/
|-- reports/
|-- scorer_results/candidate_december.png
|-- evaluate.py
|-- train_predict.py
|-- score.py
|-- validation_predictions.csv
`-- requirements.txt
```

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Reproduce the final holdout evaluation:

```bash
python scripts/data_audit.py
python evaluate.py
```

Train on all labeled rows and generate both required prediction files:

```bash
python train_predict.py
```

Run the employer-provided scorer:

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Regenerate the PDF report after evaluation and scoring:

```bash
python scripts/generate_report.py
```

## Outputs

- `validation_predictions.csv`: exact `load_id,predicted_rate` schema with 12,000 rows.
- `data/december_chart_inputs.csv`: completed 31-day fixed December scenario.
- `scorer_results/candidate_december.png`: chart created by the supplied scorer.
- `reports/freight_rate_assessment_report.pdf`: validation, data audit, methodology, results, and December chart.
- `artifacts/validation_metrics.json`: machine-readable holdout metrics.
- `artifacts/feature_importance.csv`: holdout feature importance.
- `artifacts/error_by_equipment.csv`: holdout error slices.

## Fixed December assumptions

The provided December scenario omits coordinates, market index, and quote signal. The pipeline maps Lexington and Fort Wayne to median labeled coordinates and uses the latest 30-day labeled medians for market index and quote signal. These values remain fixed, so only date changes across the chart.

## Limitations

The labeled period ends in October 2025. Freight markets can shift with fuel, capacity, and seasonality. Monitor MAE and error by equipment and lane after actuals arrive, retrain when drift is material, and keep human review for exceptional loads outside the observed operating range.
