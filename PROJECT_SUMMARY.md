# Sri Lanka Tea Auction Research Project - Technical Summary

## 1. Project Purpose

This project builds an end-to-end data pipeline for Sri Lanka tea auction reports and prepares a research-ready dataset for modeling tea auction prices.

Core objective:

- Predict and explain variation in auction mid-price (LKR) using market, weather, production, sentiment, and category-level features.

Working target used in analysis:

- `price_mid_lkr` (with `price_mid_lkr_log` used to reduce skew for modeling).

---

## 2. What Has Been Built So Far

### 2.1 Ingestion Layer (PDF -> structured CSVs)

Implemented in:

- [src/ingestion/tea_pipeline_v2.py](src/ingestion/tea_pipeline_v2.py)
- [src/ingestion/weather_pipeline.py](src/ingestion/weather_pipeline.py)

Current capabilities:

1. Parses Forbes & Walker weekly PDF reports.
2. Extracts 8 structured market/price tables.
3. Runs weather pipeline inside tea pipeline (integration completed).
4. Auto-builds sale date mapping from raw PDF filenames + cover page sale number parsing.
5. Generates lagged weather features (1, 2, 3 sale lags).

Generated interim outputs:

- [data/Interim/01_sales_index.csv](data/Interim/01_sales_index.csv)
- [data/Interim/02_auction_offerings.csv](data/Interim/02_auction_offerings.csv)
- [data/Interim/03_quantity_sold.csv](data/Interim/03_quantity_sold.csv)
- [data/Interim/04_high_grown_prices.csv](data/Interim/04_high_grown_prices.csv)
- [data/Interim/05_low_grown_prices.csv](data/Interim/05_low_grown_prices.csv)
- [data/Interim/06_offgrade_dust_prices.csv](data/Interim/06_offgrade_dust_prices.csv)
- [data/Interim/07_top_prices.csv](data/Interim/07_top_prices.csv)
- [data/Interim/08_column_dictionary.csv](data/Interim/08_column_dictionary.csv)
- [data/Interim/09_weather_features.csv](data/Interim/09_weather_features.csv)

### 2.2 Master Table Construction

Implemented in:

- [src/processing/build_master_table.py](src/processing/build_master_table.py)

Key method:

1. Creates a price-observation spine from tables 04/05/06.
2. Adds sale-level context from 01 and 03.
3. Pivots auction offerings (02) wide by category.
4. Pivots weather features (09) wide by region.
5. Joins all components into one analytical table.

Output:

- [data/Processed/master_tea_prices.csv](data/Processed/master_tea_prices.csv)

### 2.3 Reduced Feature Set Construction

Implemented in:

- [src/processing/build_reduced_master.py](src/processing/build_reduced_master.py)

Key method:

1. Rebuilds full master.
2. Applies explicit feature pruning rules (high null, constant, redundant, collinearity-prone columns).
3. Runs dynamic audit for additional null/variance issues.
4. Keeps essential analytical columns.

Output:

- [data/Processed/reduced_master_tea_prices.csv](data/Processed/reduced_master_tea_prices.csv)

### 2.4 Preprocessing for Modeling/EDA

Implemented in:

- [src/processing/preprocess_tea.py](src/processing/preprocess_tea.py)

Key method:

1. Leakage review and exclusion list (lo/hi/range leakage columns).
2. Sanity check for known Sale 3 volume anomaly.
3. Elevation harmonization (`high/low/medium` -> `*_grown` form).
4. Structural-null handling strategy (do not impute `grade` and `tier` globally).
5. Lag imputation using grouped ffill/bfill by table source and sale order.
6. Production missingness handling with indicator feature.
7. Feature encoding for month/tier/elevation/category/table source.
8. Target transform (`log1p`) for skew reduction.

Output:

- [data/Processed/tea_preprocessed.csv](data/Processed/tea_preprocessed.csv)

### 2.5 EDA Notebook and Figures

Implemented in:

- [notebooks/tea_eda.ipynb](notebooks/tea_eda.ipynb)

Generated figures:

- [reports/figures/fig1_price_distribution.png](reports/figures/fig1_price_distribution.png)
- [reports/figures/fig2_price_trends.png](reports/figures/fig2_price_trends.png)
- [reports/figures/fig3_grade_analysis.png](reports/figures/fig3_grade_analysis.png)
- [reports/figures/fig4_correlations.png](reports/figures/fig4_correlations.png)
- [reports/figures/fig5_context.png](reports/figures/fig5_context.png)
- [reports/figures/fig6_tier_range.png](reports/figures/fig6_tier_range.png)
- [notebooks/fig7_weather_grade_impact.png](notebooks/fig7_weather_grade_impact.png)

Secondary notebook (experimental/scratch):

- [notebooks/test.ipynb](notebooks/test.ipynb)

### 2.6 Extended EDA Notebook and Figures

### 2.7 Unified Modeling Pipeline

Implemented in:

- [src/processing/unified_modeling_pipeline.ipynb](src/processing/unified_modeling_pipeline.ipynb)

Key method:

1. Loads preprocessed data from [data/Processed/tea_preprocessed.csv](data/Processed/tea_preprocessed.csv).
2. Filters data by 4 tea segments: High Grown, Low Grown, Off-Grade, Dust.
3. Implements TimeSeriesSplit(k=5) chronological cross-validation.
4. Benchmarks 8 regression models:
   - Linear: Ridge (α=1.0), Lasso (α=0.001), ElasticNet (α=0.001, l1_ratio=0.5)
   - Tree-based: RandomForest (n_estimators=400), GradientBoosting, XGBoost (n_estimators=400, learning_rate=0.05), LightGBM (n_estimators=400)
   - Kernel: SVR (kernel='rbf', C=10.0, epsilon=0.05)
5. Computes metrics (RMSE, MAE, MAPE, R²) in original price scale.
6. Aggregates fold results by segment and model.

Output:

- [data/Processed/unified_model_cv_results.csv](data/Processed/unified_model_cv_results.csv) - Summary cross-validation results (mean metrics per model × segment)

### 2.8 Extended EDA Notebook and Figures

Implemented in:

- [notebooks/tea_eda_extended.ipynb](notebooks/tea_eda_extended.ipynb)

This notebook extends the primary EDA with five additional analyses targeting the
dual-market hypothesis — that High Grown and Low Grown tea prices respond to
fundamentally different drivers (weather vs geopolitical/export demand).

Generated figures:

- [reports/figures/fig8_weather_heatmap.png] — Weekly weather condition and crop intake direction by region across all auction weeks
- [reports/figures/fig9_rainfall_vs_price_by_segment.png] — Rainfall vs mid-price scatter comparing High Grown and Low Grown sensitivity
- [reports/figures/fig10_rainfall_lag_effect.png] — Lag effect analysis identifying optimal forecasting horizon for High Grown prices
- [reports/figures/fig11_lkr_vs_usd_price.png] — LKR vs USD price trend comparison exposing inflationary illusion
- [reports/figures/fig12_top_estates.png] — Top estate consistency analysis revealing brand immunity to market shocks

### 2.9 Segment Modeling Notebook (Baseline Benchmark)

Implemented in:

- [notebooks/model.ipynb](notebooks/model.ipynb)

Current capabilities:

1. Builds segment-specific modeling datasets for High Grown, Low Grown, Off-Grade, and Dust.
2. Engineers segment-aware structural and weather interaction features (for example, region-routed precipitation and quality x weather interactions).
3. Uses a time-aware split by sale rank (last 5 unique sales for test) per segment.
4. Trains two tree-based baselines per segment: Random Forest and Gradient Boosting.
5. Evaluates in original LKR scale using MAE, RMSE, R2, and MAPE.
6. Produces comparable test-set benchmark tables across all four segments.

### 2.10 Granger Causality Testing

Implemented in:
- [notebooks/granger_causality.ipynb](notebooks/granger_causality.ipynb)

**Task (T1.2):** Granger causality tests between lagged weather variables (precipitation, temperature, sunshine) and segment prices, at lag orders 1–4 per segment (High Grown, Low Grown, Off-Grade, Dust). Produces a compact IEEE-formatted table for paper Section V.

Process:
1. Aggregated price observations to one mean price per segment per sale (26-point weekly time series).
2. Applied Augmented Dickey-Fuller (ADF) tests to all price and weather series; first-differenced non-stationary series before testing.
3. Ran `statsmodels.grangercausalitytests` (SSR F-test) for all 48 triplets: 4 segments × 3 weather variables × 4 lag orders.
4. Flagged marginal results (0.10 < p < 0.15) as tentative `(t)` given low power (n ≈ 22 effective observations at lag 4).

Generated figures:
- [reports/figures/figA_granger_pvalue_heatmap.png](reports/figures/figA_granger_pvalue_heatmap.png) — P-value heatmap (-log10 scale) across all 12 segment×variable pairs and 4 lag orders
- [reports/figures/figB_granger_fstat_profiles.png](reports/figures/figB_granger_fstat_profiles.png) — F-statistic profiles per weather variable showing how test strength evolves with lag order
- [reports/figures/figC_granger_timeseries_overlays.png](reports/figures/figC_granger_timeseries_overlays.png) — Dual-axis time-series overlays for all significant pairs (p < 0.05)
- [reports/figures/figD_granger_adf_stationarity.png](reports/figures/figD_granger_adf_stationarity.png) — ADF p-values for all tested price and weather series (hatched = first-differenced)

Generated tables:
- [reports/tables/granger_causality_full.csv](reports/tables/granger_causality_full.csv) — All 48 test results (F-stat, p-value, significance marker)
- [reports/tables/granger_causality_summary.csv](reports/tables/granger_causality_summary.csv) — Pivoted summary in IEEE format
- [reports/tables/granger_causality_table.tex](reports/tables/granger_causality_table.tex) — LaTeX-formatted table for paper Section V

### 2.11 SHAP Explainability Analysis (Segment Models)

Implemented in:

- [notebooks/model.ipynb](notebooks/model.ipynb)

Current capabilities:

1. Selects the best-performing model per segment (by highest test R2).
2. Computes SHAP values for each segment-specific best model.
3. Handles post-imputation feature alignment safely (avoids index mismatch when all-missing columns are dropped).
4. Produces a 4-panel SHAP beeswarm summary figure (top features per segment).
5. Produces SHAP dependence plots for the top weather feature per segment with interaction coloring.
6. Exports publication-ready explainability figures in both PNG and PDF formats.

Generated figures:

- [reports/figures/fig_shap_summary_4panel.png](reports/figures/fig_shap_summary_4panel.png)
- [reports/figures/fig_shap_summary_4panel.pdf](reports/figures/fig_shap_summary_4panel.pdf)
- [reports/figures/fig_shap_dependence_weather.png](reports/figures/fig_shap_dependence_weather.png)
- [reports/figures/fig_shap_dependence_weather.pdf](reports/figures/fig_shap_dependence_weather.pdf)

## 3. Current Data Inventory (Observed)

### 3.1 Interim Layer

- `01_sales_index.csv`: 26 rows x 108 cols
- `02_auction_offerings.csv`: 234 rows x 7 cols
- `03_quantity_sold.csv`: 26 rows x 32 cols
- `04_high_grown_prices.csv`: 516 rows x 6 cols
- `05_low_grown_prices.csv`: 1348 rows x 6 cols
- `06_offgrade_dust_prices.csv`: 1170 rows x 6 cols
- `07_top_prices.csv`: 2364 rows x 6 cols
- `08_column_dictionary.csv`: 232 rows x 5 cols
- `09_weather_features.csv`: 100 rows x 61 cols

### 3.2 Processed Layer

- `master_tea_prices.csv`: 3034 rows x 270 cols
- `reduced_master_tea_prices.csv`: 3034 rows x 173 cols
- `tea_preprocessed.csv`: 3034 rows x 182 cols

---

## 4. End-to-End Technical Methodology

## Step A - Data Collection

Source data:

- Raw weekly PDF auction reports in [data/Raw](data/Raw).

Collection protocol:

1. Add new PDF reports into raw folder.
2. Use naming convention `Sale of DD & DD Month YYYY.pdf` (or timestamp-suffixed variants).
3. Run ingestion pipeline to parse all reports.

Weather enrichment sources:

- Text weather descriptions from report section `CROP AND WEATHER`.
- API weather from Open-Meteo archive endpoint (region coordinates fixed in code).

## Step B - Parsing and Structured Extraction

In [src/ingestion/tea_pipeline_v2.py](src/ingestion/tea_pipeline_v2.py):

1. Extracts sale header metadata (`sale_number`, `sale_year`, `sale_month`, `sale_id`).
2. Parses auction totals and category-level offerings.
3. Builds commentary and sentiment scores from text.
4. Extracts production, quantity sold, FX rates.
5. Parses high-grown, low-grown, off-grade/dust price tables.
6. Parses top prices by estate.
7. Writes table-wise interim CSVs with deduplication keys.

## Step C - Weather Feature Engineering

In [src/ingestion/weather_pipeline.py](src/ingestion/weather_pipeline.py):

1. Builds sale date map automatically from filenames + cover page sale ID.
2. Parses weather sentiment/flags from text per region.
3. Fetches 7-day pre-auction weather windows per region.
4. Aggregates weather variables into totals/means/max/min.
5. Adds lagged weather features (`lag1`, `lag2`, `lag3`) by region and sale order.

## Step D - Analytical Master Construction

In [src/processing/build_master_table.py](src/processing/build_master_table.py):

1. Combines price tables as row-level spine.
2. Joins sale context, offerings pivot, weather pivot.
3. Creates derived variables (`price_mid_lkr`, `price_range_lkr`, volume YoY, avg precipitation).
4. Outputs a wide, model-ready master table.

## Step E - Feature Reduction and Audit

In [src/processing/build_reduced_master.py](src/processing/build_reduced_master.py):

1. Explicitly removes low-value and unstable columns.
2. Removes collinear/duplicate representations (examples: some rain totals, lots/kgs duplicates).
3. Removes many near-constant/very sparse columns.
4. Keeps important structural variables and sale identifiers.

## Step F - Preprocessing and Modeling Readiness

In [src/processing/preprocess_tea.py](src/processing/preprocess_tea.py):

1. Establishes leakage guardrails.
2. Applies encoded categorical representations.
3. Handles lag missingness and production missingness.
4. Produces transformed target `price_mid_lkr_log`.
5. Exports final preprocessed table for EDA/modeling.

## Step G - EDA and Preliminary Interpretation

In [notebooks/tea_eda.ipynb](notebooks/tea_eda.ipynb):

1. Dataset profiling and distribution diagnostics.
2. Temporal trend analysis across auction sales.
3. Elevation/category/grade/tier comparison.
4. Correlation screening by feature group.
5. Weather-context and weather-grade sensitivity analysis.
6. Figure export for paper-ready visuals.

## Step H - Baseline Segment Modeling

In [notebooks/model.ipynb](notebooks/model.ipynb):

1. Constructs per-segment feature sets combining sale context, gross revenue signals, structural attributes, and weather features.
2. Creates log-target (`price_log = log1p(price_mid_lkr)`) for model training.
3. Applies chronological split by sale rank (holdout = latest 5 sales).
4. Trains Random Forest and Gradient Boosting regressors inside imputation pipelines.
5. Converts predictions back to LKR and reports MAE/RMSE/R2/MAPE for segment-level comparison.

## Step I - Granger Causality Testing

In [notebooks/granger_causality.ipynb](notebooks/granger_causality.ipynb):
1. Aggregate row-level price data to one mean price per segment per sale (sale-level time series).
2. ADF stationarity tests on all price and weather series; apply first-differencing where required.
3. Run 48 Granger causality tests (SSR F-test) across all segment × weather variable × lag combinations.
4. Identify significant causal relationships and best-lag per pair.
5. Export IEEE-formatted summary table and LaTeX source for paper Section V.
6. Generate four diagnostic figures (heatmap, F-profile, time-series overlays, ADF summary).

---

## 5. Preliminary Results (Current)

From [data/Processed/tea_preprocessed.csv](data/Processed/tea_preprocessed.csv) and EDA figures:

### 5.1 Target Distribution (Refreshed 2026-03-21)

- Price observations with target: 2,886 (95.1% of rows)
- Mean mid-price: 1,255.79 LKR
- Median mid-price: 1,120.00 LKR
- Range: 290.00 to 5,700.00 LKR
- Skewness before transform: 2.869
- Skewness after `log1p` transform: 0.587

Interpretation:

- Distribution remains right-skewed, but the refreshed coverage and log transform continue to yield a modeling-friendly target.

### 5.2 Price Structure by Elevation

- Low Grown: mean 1,446.8 (n=1,659)
- High Grown: mean 1,057.6 (n=854)
- Medium Grown: mean 859.9 (n=373)

Interpretation:

- Elevation is a major structural determinant of price.

### 5.3 Price Structure by Category Type

- Low Grown: mean 1,578.7 (n=1,348)
- High Grown: mean 1,135.6 (n=516)
- Dust: mean 984.4 (n=523)
- Off Grade: mean 792.2 (n=499)

Interpretation:

- Large spread across category types supports category-aware modeling.

### 5.4 Observed Signal Patterns

- Strongest direct correlations with target are structural/derived variables (including leakage variables if included naively).
- Sentiment and most standalone weather features show weaker direct linear effects.
- Notebook Figure 7 indicates grade-level heterogeneity in weather sensitivity (example shown: FBOPF most weather-sensitive in current analysis output).

Interpretation:

- Interaction effects (grade x weather, category x weather) likely matter more than simple linear weather coefficients.

### 5.5 Extended EDA & Coverage Findings

- Rainfall-price sensitivity (Fig 9): Preliminary correlation analysis shows High Grown prices exhibit a measurably different rainfall response compared to Low Grown, consistent with the dual-market hypothesis.
- Lag effect (Fig 10): Rainfall lagged by 1–2 weeks shows stronger correlation with High Grown prices than current-week rainfall, supporting the 14-day supply lag hypothesis central to the forecasting framework.
- Inflationary illusion (Fig 11): LKR and USD price trajectories diverge across the auction series, indicating that LKR-denominated broker sentiment signals can be misleading when currency depreciation is present.
- Estate consistency (Fig 12): A small number of estates appear repeatedly in weekly top-price lists regardless of broader market conditions, suggesting a brand-immunity effect in the premium segment.
- Volume contraction: 2026 auction volumes are running approximately 5.4% below 2025 levels, providing important supply-side context for price trend interpretation.

### 5.6 Baseline Segment Modeling Results

Test split used in notebook:

- High Grown: train=1,564, test=262
- Low Grown: train=4,332, test=856
- Off-Grade: train=1,572, test=313
- Dust: train=1,648, test=335

Observed benchmark metrics (test set):

- High Grown:
  - Random Forest: RMSE=124.42, MAE=77.42, R2=0.74, MAPE=6.72%
  - Gradient Boosting: RMSE=135.37, MAE=85.69, R2=0.71, MAPE=7.36%
- Low Grown:
  - Random Forest: RMSE=347.94, MAE=215.54, R2=0.73, MAPE=12.70%
  - Gradient Boosting: RMSE=349.99, MAE=213.71, R2=0.73, MAPE=12.56%
- Off-Grade:
  - Random Forest: RMSE=34.88, MAE=27.81, R2=0.95, MAPE=3.83%
  - Gradient Boosting: RMSE=34.99, MAE=28.39, R2=0.95, MAPE=3.94%
- Dust:
  - Random Forest: RMSE=49.78, MAE=38.11, R2=0.93, MAPE=4.06%
  - Gradient Boosting: RMSE=48.86, MAE=35.99, R2=0.94, MAPE=3.81%

Interpretation:

- Segment-wise baselines are strong for Off-Grade and Dust (R2 >= 0.93), moderate for High Grown and Low Grown (R2 around 0.73-0.74).
- High Grown currently favors Random Forest on all reported metrics.
- Dust currently favors Gradient Boosting.
- Low Grown is mixed: Random Forest slightly lower RMSE, while Gradient Boosting slightly lower MAE/MAPE.

### 5.7 Granger Causality Results

**Test scope:** 48 tests — 4 segments × 3 weather variables (Precipitation, Temperature, Sunshine) × 4 lag orders (1–4 weeks). Series range: 23–26 auction sales (S34/25 – S10/26).

**Stationarity decisions (ADF at α=0.05):**

| Series | Result | Treatment |
|--------|--------|-----------|
| All price series (4 segments) | Non-stationary | First-differenced |
| Precipitation (all regions) | Non-stationary | First-differenced |
| Temperature (all regions) | Stationary | Levels |
| Sunshine (all regions) | Stationary | Levels |

**Overall significance summary:**

| Threshold | Count (of 48) |
|-----------|--------------|
| p < 0.01 | 0 |
| p < 0.05 | 3 |
| p < 0.10 | 6 |

**Significant relationships (p < 0.10):**

| Segment | Weather Variable | Lag | F-stat | p-value | Marker |
|---------|-----------------|-----|--------|---------|--------|
| High Grown | Sunshine | 2 | 3.35 | 0.0649 | * |
| High Grown | Sunshine | 3 | 2.97 | 0.0787 | * |
| High Grown | Sunshine | 4 | 6.14 | 0.0147 | ** |
| Off-Grade | Sunshine | 2 | 4.32 | 0.0293 | ** |
| Off-Grade | Temperature | 1 | 7.00 | 0.0160 | ** |
| Off-Grade | Temperature | 2 | 3.55 | 0.0530 | * |

**Key findings by weather variable:**
- **Precipitation:** No significant Granger causality detected (p < 0.10) in any segment. Prior week rainfall does not carry incremental predictive power beyond lagged prices alone.
- **Temperature:** Granger-causes Off-Grade prices at Lag 1 (F=7.00, p=0.016 **). Effect weakens but remains marginal at Lag 2. No significant effect on High Grown, Low Grown, or Dust.
- **Sunshine:** Strongest and most consistent causal signal. Granger-causes High Grown prices with increasing strength from Lag 2 through Lag 4 (best: Lag 4, F=6.14, p=0.015 **), consistent with a multi-week supply response. Also Granger-causes Off-Grade prices at Lag 2 (F=4.32, p=0.029 **).

**Interpretation:**
- The sunshine→High Grown relationship (strengthening from Lag 2 to Lag 4) supports the hypothesis that weather conditions during the growing period affect supply 2–4 auction weeks later.
- Precipitation failing to achieve significance despite its theoretical relevance is likely a combination of small sample power (n ≈ 22–25) and high week-to-week variance in rainfall series.
- Low Grown and Dust segments show no statistically significant weather Granger causality at any lag, suggesting their prices are driven more by demand-side or structural factors.
- Results are treated as preliminary given the small auction window (26 sales). All interpretations in the paper should be qualified by the limited sample size.

### 5.8 SHAP Explainability Outcomes (Latest `notebooks/model.ipynb` Run)

Generated explainability outputs:

- Figure A: 4-panel SHAP beeswarm summary comparing top-8 drivers for High Grown, Low Grown, Off-Grade, and Dust.
- Figure B: SHAP dependence plots for the most influential weather feature in each segment.

Observed interpretation outcomes:

- Feature influence is segment-specific, validating the segment-wise modeling strategy rather than a single pooled model.
- Weather-related effects are present across all segments but with different strength and nonlinear behavior by segment.
- Structural tea attributes and market/supply context remain major contributors, while weather and crop signals provide additional explanatory lift.
- Interaction behavior (for example, quality/segment rank with precipitation) appears materially important, supporting the engineered interaction feature design.
- Off-Grade and Dust explainability profiles are generally more stable/compact, consistent with their stronger predictive fit (higher R2).

Research implication:

- The SHAP analysis strengthens the dual-market interpretation by showing that each segment responds to a different mix of market, structural, and weather drivers, rather than a uniform effect pattern.
---

## 6. Technical Validation Status

### 6.1 Pipeline Validation Completed

1. Weather sale-date mapping now handles variable spacing and one-day filename formats in [src/ingestion/weather_pipeline.py](src/ingestion/weather_pipeline.py).
2. Weather pipeline is integrated into tea ingestion via `run_pipeline_weather` in [src/ingestion/tea_pipeline_v2.py](src/ingestion/tea_pipeline_v2.py).
3. Interim weather output now includes 25 unique sales and 4 regions (100 rows).

### 6.2 Data Quality Controls in Code

1. Deduplication keys per table during ingestion writes.
2. Rule-based extraction sanity constraints for sale IDs and date parsing.
3. Leakage columns explicitly identified in preprocessing.
4. Structural null strategy documented (grade/tier not globally imputed).
5. Reduced table performs explicit + dynamic feature audit.

### 6.3 Freshness Status (2026-03-21)

Observed artifact freshness:

- Interim CSVs (01–09) and processed outputs (master/reduced/preprocessed) were all regenerated on 2026-03-21 from the latest ingestion run.
- [notebooks/tea_eda.ipynb](notebooks/tea_eda.ipynb) was re-executed end-to-end, refreshing Figures 1–7 and exporting the new Figure 13 coverage check.
- [notebooks/model.ipynb](notebooks/model.ipynb) contains the latest baseline segment benchmark outputs for High Grown, Low Grown, Off-Grade, and Dust.

Implication:

- Baseline EDA, processed datasets, and published figures now align with the full 26-sale interim inventory.

Next validation step:

- Mirror the refresh in the extended notebook (`notebooks/tea_eda_extended.ipynb`) so that Figures 8–12 also reference the expanded dataset.

---

## 7. File-by-File Role Summary

### Ingestion

- [src/ingestion/tea_pipeline_v2.py](src/ingestion/tea_pipeline_v2.py): Main PDF parser, table extractors, CSV writer, weather pipeline trigger.
- [src/ingestion/weather_pipeline.py](src/ingestion/weather_pipeline.py): Weather extraction/API fusion, sale-date automation, lag feature generation.

### Processing

- [src/processing/build_master_table.py](src/processing/build_master_table.py): Full-width analytical master table construction.
- [src/processing/build_reduced_master.py](src/processing/build_reduced_master.py): Rule-based + dynamic feature reduction.
- [src/processing/preprocess_tea.py](src/processing/preprocess_tea.py): Leakage control, imputation strategy, encodings, target transform.

### Notebooks

- [notebooks/tea_eda.ipynb](notebooks/tea_eda.ipynb): Main EDA notebook for preliminary results and publication figures.
- [notebooks/eda_extended.ipynb](notebooks/eda_extended.ipynb): Extended EDA targeting dual-market hypothesis (weather vs demand drivers).
- [notebooks/granger_causality.ipynb](notebooks/granger_causality.ipynb): Granger causality tests (weather → price) for all segments at lags 1–4; produces IEEE table and four diagnostic figures.
- [notebooks/test.ipynb](notebooks/test.ipynb): Experimental notebook; not the canonical EDA artifact.

### Outputs

- [data/Interim](data/Interim): Extraction-stage structured tables.
- [data/Processed](data/Processed): Modeling-stage master/reduced/preprocessed datasets.
- [reports/figures](reports/figures): Primary exported figure set for the short paper.
- [reports/tables](reports/tables): Statistical result tables (Granger causality full results, summary pivot, LaTeX source).

---

## 8. Suggested Paper-Ready Outline (Based on Current Work)

1. Introduction and research objective.
2. Data sources and collection protocol.
3. Pipeline architecture and extraction methodology.
4. Feature engineering and preprocessing decisions.
5. Technical validation and quality controls.
6. Preliminary results (distribution, segmentation, trend, weather interactions).
7. Limitations and next-phase plan.

---

## 9. Immediate Next Steps for Team

1. Rebuild processed datasets from the latest interim outputs.
2. Re-run EDA notebook to refresh all figures with current data coverage.
3. Freeze a versioned snapshot of CSVs and figures for the short paper submission.
4. Start model benchmarking using leakage-safe feature matrix.
5. Add an experiment log (metrics + feature set + train/validation split) for reproducible reporting.
