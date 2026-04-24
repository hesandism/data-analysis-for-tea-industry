# Data Analysis For Tea Industry

Forecasting and analysis pipeline for Sri Lankan tea auction prices using report extraction, weather enrichment, and time-series machine learning.

## Research Domain
Sri Lanka's tea auction market is segmented by product type and quality band (for example high grown, low grown, off-grade, and dust). Prices move with a combination of:
- weather conditions in tea-growing regions,
- supply and demand pressure in weekly auctions,
- product quality mix (grade and tier),
- and temporal effects such as lagged price momentum.

This repository studies those dynamics and compares pooled forecasting against segment-specific forecasting.

## What This Project Does
1. Extract structured data from weekly PDF market reports.
2. Build normalized interim tables (sales context, offerings, quantities, prices, weather, top prices).
3. Construct master and reduced analytical datasets.
4. Preprocess and engineer time-aware features.
5. Run notebook-based experiments for:
- exploratory/event analysis,
- segment-specific forecasting,
- unified pooled forecasting,
- Granger causality for weather-lag feature selection.

## Repository Workflow

### Ingestion (`src/ingestion`)
- `tea_pipeline_v2.py`: parses weekly PDF reports into multiple structured CSV tables.
- `weather_pipeline.py`: extracts weather text, fetches historical weather API data, and creates region-level weather features with lags.

### Processing (`src/processing`)
- `build_master_table.py`: joins interim sources into `master_tea_prices.csv`.
- `build_reduced_master.py`: removes redundant/sparse columns to produce a compact modeling table.
- `preprocess_tea.py`: fixes anomalies, normalizes fields, derives target helpers, and applies encodings.
- `feature_engineering.py`: adds interactions, rolling statistics, and polynomial terms.

### Notebooks (`notebooks`)
- `visualization_eda.ipynb`: exploratory and event-window analysis.
- `segment-specific.ipynb`: per-catalogue time-series CV benchmarking.
- `unified_pooled.ipynb`: pooled benchmark and unified-vs-segment comparison.
- `granger_causality.ipynb`: ADF stationarity and Granger causality tests for lag feature relevance.

## Key Data Outputs
- `data/Interim/*.csv`: extracted report tables.
- `data/processed/master_tea_prices.csv`: full feature table.
- `data/processed/reduced_master_tea_prices.csv`: reduced table for modeling.
- `data/processed/tea_preprocessed.csv`: cleaned modeling base.
- `data/processed/final_clean_dataset_long.csv`: primary notebook dataset.
- `results/*.csv`: model CV summaries, comparison outputs, and causality results.

## Setup
Use the steps in `Setup.md` or run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Typical Run Order
```bash
python src/ingestion/tea_pipeline_v2.py
python src/processing/build_master_table.py
python src/processing/build_reduced_master.py
python src/processing/preprocess_tea.py
python src/processing/feature_engineering.py
```

Then open notebooks in `notebooks/` to reproduce EDA, forecasting benchmarks, and causality analysis.

## Models and Evaluation
- Models: LightGBM, XGBoost, Random Forest, Gradient Boosting.
- Validation: time-aware splits (`TimeSeriesSplit`).
- Metrics: RMSE, MAE, MAPE, R2.

## Notes
- The project is designed so new weekly sales can be added by updating raw reports and re-running the pipeline.
- Feature relevance can differ by catalogue; both pooled and segment-specific views are kept in the workflow.
