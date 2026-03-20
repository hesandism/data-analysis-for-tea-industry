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
- [reports/figures/fig13_data_refresh_coverage.png](reports/figures/fig13_data_refresh_coverage.png)

Secondary notebook (experimental/scratch):
- [notebooks/test.ipynb](notebooks/test.ipynb)

### 2.6 Extended EDA Notebook and Figures 
Implemented in:

 - [notebooks/eda_extended.ipynb]

This notebook extends the primary EDA with five additional analyses targeting the
dual-market hypothesis — that High Grown and Low Grown tea prices respond to
fundamentally different drivers (weather vs geopolitical/export demand).

Generated figures:

- [reports/figures/fig8_weather_heatmap.png] — Weekly weather condition and crop intake direction by region across all auction weeks
- [reports/figures/fig9_rainfall_vs_price_by_segment.png] — Rainfall vs mid-price scatter comparing High Grown and Low Grown sensitivity
- [reports/figures/fig10_rainfall_lag_effect.png] — Lag effect analysis identifying optimal forecasting horizon for High Grown prices
- [reports/figures/fig11_lkr_vs_usd_price.png] — LKR vs USD price trend comparison exposing inflationary illusion
- [reports/figures/fig12_top_estates.png] — Top estate consistency analysis revealing brand immunity to market shocks


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
- Data refresh coverage (Fig 13): Processed datasets now cover 26 sales (up from 10) with 95.1% of rows retaining a usable mid-price target, ensuring downstream models and figures reflect the full interim inventory.
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
- [notebooks/test.ipynb](notebooks/test.ipynb): Experimental notebook; not the canonical EDA artifact.

### Outputs
- [data/Interim](data/Interim): Extraction-stage structured tables.
- [data/Processed](data/Processed): Modeling-stage master/reduced/preprocessed datasets.
- [reports/figures](reports/figures): Primary exported figure set for the short paper.

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

1. Re-run the extended EDA notebook so advanced figures (8–12) share the refreshed coverage.
2. Freeze a versioned snapshot of refreshed processed CSVs and Figures 1–13 for paper-ready handoff.
3. Start model benchmarking using the leakage-safe `tea_preprocessed.csv` (2,886 targets) and track results in an experiment log.
4. Document any additional diagnostics (e.g., feature drift, incremental weather signals) uncovered while scaling beyond the initial 10-sale subset.
