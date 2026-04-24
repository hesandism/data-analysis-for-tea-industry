# DS-Research Project Summary

## Overview
This project studies weekly Sri Lankan tea auction prices and how they change with market structure, weather conditions, and lagged temporal effects. The workflow combines PDF report extraction, weather enrichment, dataset engineering, and time-series aware machine learning.

The core prediction target in modelling notebooks is the next-auction price, built from `price_mid_lkr` per catalogue and grade stream.

## Research Domain (Short)
Sri Lanka tea auctions are segmented (for example: high grown, low grown, off-grade, dust), and each segment can react differently to rainfall, temperature, sunshine, demand, and supply pressure. This research evaluates whether one pooled model can represent all segments well, or whether segment-specific models perform better.

## End-to-End Process
1. Ingestion from weekly PDF market reports (`src/ingestion`).
2. Parse auction tables, commentary sentiment, sales context, and structured price ranges.
3. Enrich with region-level weather features from Open-Meteo and derive lag features.
4. Build a long-form master table (`src/processing/build_master_table.py`).
5. Build a reduced table with column audits to remove sparse/redundant signals (`src/processing/build_reduced_master.py`).
6. Preprocess for modelling: anomaly fixes, structural handling, encodings, derived targets (`src/processing/preprocess_tea.py`).
7. Add interactions, rolling windows, and polynomial features (`src/processing/feature_engineering.py`).
8. Run notebook experiments for EDA, forecasting benchmarks, and Granger causality.

## Main Data Artifacts
- `data/Interim/01-09_*.csv`: extracted normalized tables from raw reports.
- `data/processed/master_tea_prices.csv`: full joined master table.
- `data/processed/reduced_master_tea_prices.csv`: reduced analytical table.
- `data/processed/tea_preprocessed.csv`: modelling base table.
- `data/processed/final_clean_dataset_long.csv`: notebook-ready long dataset.
- `results/*.csv`: cross-validation summaries, unified-vs-segment comparisons, and Granger outputs.

## Notebook Workflow Summary
- `notebooks/visualization_eda.ipynb`: event-window and exploratory visual analysis.
- `notebooks/segment-specific.ipynb`: per-catalogue forecasting benchmarks using time-series CV.
- `notebooks/unified_pooled.ipynb`: pooled model benchmark and unified-vs-segment evaluation.
- `notebooks/granger_causality.ipynb`: ADF stationarity checks and Granger-based lag relevance analysis.

## Modelling Approach
- Time-aware validation with `TimeSeriesSplit`.
- Benchmarked regressors include LightGBM, XGBoost, Random Forest, and Gradient Boosting.
- Metrics include RMSE, MAE, MAPE, and R2.
- Comparative analysis is done both pooled (all catalogues together) and segment-specific.

## Practical Outcome
The project produces a reproducible data-science pipeline for tea auction forecasting, while also giving interpretable evidence on which weather and lag features are useful by segment.
