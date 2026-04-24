# DS-Research: Experimental Results

**Dataset**: `final_clean_dataset_long.csv` — 12,233 rows · 26 features · 2023–2026  
**Task**: Predict next-week tea auction price (`price_mid_lkr_t_plus_1w`) per catalogue  
**CV Protocol**: 5-fold `TimeSeriesSplit` · 80/20 temporal train/test split  
**Models**: XGBoost · LightGBM · Random Forest · Gradient Boosting  

---

## 1. Segment-Specific Modelling

Separate model trained per catalogue. Results are 5-fold CV means on the training set (80% of each catalogue's chronological data).

### 1.1 High Grown — n = 90 sales

| Model | RMSE ± std | MAE | MAPE (%) | R² ± std |
|---|---|---|---|---|
| **LightGBM** ⭐ | **159.97 ± 48.38** | **102.72** | **8.60** | **0.461 ± 0.218** |
| XGBoost | 160.22 ± 47.90 | 98.34 | 8.20 | 0.455 ± 0.224 |
| Gradient Boosting | 163.43 ± 51.37 | 101.56 | 8.45 | 0.430 ± 0.263 |
| Random Forest | 170.22 ± 55.33 | 103.38 | 8.58 | 0.354 ± 0.402 |

> **Best**: LightGBM — RMSE 159.97 LKR · R² 0.461 · MAPE 8.60%

---

### 1.2 Low Grown — n = 104 sales

| Model | RMSE ± std | MAE | MAPE (%) | R² ± std |
|---|---|---|---|---|
| **LightGBM** ⭐ | **140.43 ± 34.30** | **74.49** | **4.10** | **0.967 ± 0.009** |
| Gradient Boosting | 142.99 ± 39.68 | 74.93 | 4.08 | 0.966 ± 0.011 |
| XGBoost | 145.69 ± 35.09 | 75.95 | 4.10 | 0.965 ± 0.009 |
| Random Forest | 147.33 ± 39.77 | 77.39 | 4.23 | 0.964 ± 0.012 |

> **Best**: LightGBM — RMSE 140.43 LKR · R² 0.967 · MAPE 4.10%

---

### 1.3 Off-Grade — n = 105 sales

| Model | RMSE ± std | MAE | MAPE (%) | R² ± std |
|---|---|---|---|---|
| **LightGBM** ⭐ | **101.35 ± 17.57** | **77.82** | **9.59** | **0.484 ± 0.167** |
| XGBoost | 101.93 ± 17.23 | 80.40 | 9.97 | 0.483 ± 0.135 |
| Random Forest | 103.42 ± 18.32 | 81.10 | 10.10 | 0.468 ± 0.144 |
| Gradient Boosting | 106.33 ± 18.06 | 82.45 | 10.23 | 0.441 ± 0.129 |

> **Best**: LightGBM — RMSE 101.35 LKR · R² 0.484 · MAPE 9.59%

---

### 1.4 Dust — n = 105 sales

| Model | RMSE ± std | MAE | MAPE (%) | R² ± std |
|---|---|---|---|---|
| **Random Forest** ⭐ | **108.35 ± 46.96** | **75.36** | **7.37** | **0.624 ± 0.342** |
| XGBoost | 110.95 ± 47.85 | 77.31 | 7.51 | 0.606 ± 0.366 |
| LightGBM | 112.27 ± 39.48 | 78.37 | 7.67 | 0.613 ± 0.291 |
| Gradient Boosting | 116.70 ± 34.98 | 85.77 | 8.44 | 0.593 ± 0.243 |

> **Best**: Random Forest — RMSE 108.35 LKR · R² 0.624 · MAPE 7.37%

---

### 1.5 Best Model Summary

| Catalogue | Best Model | RMSE (LKR) | MAE (LKR) | MAPE (%) | R² |
|---|---|---|---|---|---|
| High Grown | LightGBM | 159.97 | 102.72 | 8.60 | 0.461 |
| Low Grown | LightGBM | 140.43 | 74.49 | 4.10 | 0.967 |
| Off-Grade | LightGBM | 101.35 | 77.82 | 9.59 | 0.484 |
| Dust | Random Forest | 108.35 | 75.36 | 7.37 | 0.624 |

LightGBM is the best model for three of the four catalogues. Dust is the only catalogue where Random Forest marginally outperforms.

### 1.6 Why Segment-Specific Modelling Makes Sense (Simple Rationale)

In plain terms, each catalogue behaves like a slightly different market. Even though all are tea auctions, the price patterns, volatility, and weather sensitivity are not the same across High Grown, Low Grown, Off-Grade, and Dust.

Segment-specific models performed better in 3 out of 4 catalogues in the OOF comparison (High Grown, Low Grown, Off-Grade). This means a model trained only on one catalogue usually learns that catalogue's own behavior more accurately than a single pooled model.

The unified model is still useful and strong overall, and it clearly wins for Dust. That suggests Dust benefits more from shared cross-catalogue information, while the other three segments benefit more from tailored, segment-level learning.

So the practical justification is:

- Use segment-specific modelling when catalogue behavior is distinct and weather effects differ by segment.
- Use unified modelling when a segment appears to share common market dynamics with others (as seen for Dust).
- Keep both approaches in the workflow: unified as a robust benchmark, segment-specific as the primary strategy for most catalogues.

---

## 2. Unified Modelling Pipeline

A single pooled model trained across all catalogues simultaneously. Same 5-fold `TimeSeriesSplit` CV. Dataset used: `final_clean_dataset_long.csv` with 31 engineered features (price lags 1–3, weather lags, rank encodings, label-encoded categoricals).

### 2.1 All-Model CV Summary (Pooled, n = 11,739)

| Rank | Model | RMSE ± std | MAE | MAPE (%) | R² ± std | RMSE CV% | R² CV% |
|---|---|---|---|---|---|---|---|
| 1 | **LightGBM** ⭐ | **137.95 ± 19.83** | **83.36** | **6.89** | **0.9515 ± 0.009** | 14.4 | 0.9 |
| 2 | Gradient Boosting | 140.14 ± 19.35 | 89.30 | 7.45 | 0.9499 ± 0.009 | 13.8 | 1.0 |
| 3 | XGBoost | 141.91 ± 21.12 | 83.64 | 6.82 | 0.9480 ± 0.014 | 14.9 | 1.4 |
| 4 | Random Forest | 142.28 ± 21.40 | 82.70 | 6.72 | 0.9483 ± 0.011 | 15.0 | 1.1 |

> **Best unified model**: LightGBM — RMSE 137.95 LKR · R² 0.9515 · MAPE 6.89%  
> All models achieve R² > 0.948, confirming strong pooled predictability.

---

### 2.2 OOF Comparison: Unified vs Segment-Specific

Out-of-fold (OOF) predictions used for leakage-free comparison.

| Catalogue | Approach | Model | N (OOF) | RMSE | MAE | MAPE (%) | R² | Winner |
|---|---|---|---|---|---|---|---|---|
| **Dust** | Unified | LightGBM | 1,792 | **100.76** | **72.29** | **7.23** | **0.717** | ✅ Unified |
| Dust | Segment-Specific | Random Forest | 1,805 | 116.21 | 75.36 | 7.37 | 0.624 | |
| **High Grown** | Segment-Specific | LightGBM | 1,765 | **165.72** | **102.72** | **8.60** | **0.457** | ✅ Segment |
| High Grown | Unified | LightGBM | 1,773 | 175.81 | 103.65 | 8.78 | 0.391 | |
| **Low Grown** | Segment-Specific | LightGBM | 4,440 | **143.75** | **74.49** | **4.10** | **0.966** | ✅ Segment |
| Low Grown | Unified | LightGBM | 4,448 | 146.62 | 79.25 | 4.53 | 0.965 | |
| **Off-Grade** | Segment-Specific | LightGBM | 1,760 | **102.56** | **77.82** | **9.59** | **0.507** | ✅ Segment |
| Off-Grade | Unified | LightGBM | 1,767 | 107.93 | 84.56 | 10.62 | 0.457 | |

**Summary**: Segment-specific models outperform the unified model for High Grown, Low Grown, and Off-Grade. The unified model wins only for Dust, suggesting that Dust pricing is driven by cross-catalogue market dynamics.

---

## 3. ADF Stationarity Test Results

Augmented Dickey-Fuller test (AIC lag selection, α = 0.05) applied to each price and weather series at the sale level per catalogue. Non-stationary series (p ≥ 0.05) were first-differenced before Granger testing.

### 3.1 Price Series

| Catalogue | ADF p-value | Stationary? | Action |
|---|---|---|---|
| High Grown | > 0.05 | No | First-differenced |
| Low Grown | > 0.05 | No | First-differenced |
| Off-Grade | < 0.05 | Yes | Levels |
| Dust | < 0.05 | Yes | Levels |

Low Grown and High Grown price series are **non-stationary in levels**, indicating a price trend or unit root. Off-Grade and Dust prices are stationary in levels.

### 3.2 Weather Series (Reconstructed Current Week)

| Catalogue | Precipitation | Temperature | Sunshine |
|---|---|---|---|
| High Grown | Stationary (levels) | Stationary (levels) | Constant — excluded |
| Low Grown | Stationary (levels) | Stationary (levels) | Stationary (levels) |
| Off-Grade | Stationary (levels) | Stationary (levels) | Stationary (levels) |
| Dust | Stationary (levels) | Stationary (levels) | Stationary (levels) |

> High Grown sunshine was constant across the sample period and was excluded from Granger testing (N/A).

---

## 4. Granger Causality Results

Tests whether lagged weather (precipitation, temperature, sunshine) provides additional predictive power for next-week tea auction prices beyond past prices alone.

**H₀**: Weather does NOT Granger-cause price  
**H₁**: Weather Granger-causes price (reject H₀)  
Significance: `***` p < 0.01 · `**` p < 0.05 · `*` p < 0.10 · `(t)` p < 0.15

### 4.1 Full Results Table — F-statistic [p-value]

| Catalogue | Variable | Lag 1 | Lag 2 | Lag 3 |
|---|---|---|---|---|
| High Grown | Precipitation | 1.899 [0.172] | 0.892 [0.414] | 1.471 [0.229] |
| High Grown | Temperature | 1.717 [0.194] | 0.906 [0.408] | 0.563 [0.641] |
| High Grown | Sunshine | N/A (constant) | N/A | N/A |
| Low Grown | Precipitation | 5.275 [0.024] `**` | 4.414 [0.015] `**` | 3.222 [0.026] `**` |
| Low Grown | Temperature | 1.283 [0.260] | 0.947 [0.392] | 0.563 [0.641] |
| Low Grown | Sunshine | 5.607 [0.020] `**` | 3.704 [0.028] `**` | 2.399 [0.073] `*` |
| Off-Grade | Precipitation | 0.579 [0.448] | 0.351 [0.705] | 0.426 [0.735] |
| Off-Grade | Temperature | 6.097 [0.015] `**` | 3.572 [0.032] `**` | 2.957 [0.036] `**` |
| Off-Grade | Sunshine | 2.585 [0.111] `(t)` | 1.409 [0.249] | 1.079 [0.362] |
| Dust | Precipitation | 0.001 [0.969] | 0.301 [0.741] | 0.802 [0.496] |
| Dust | Temperature | 1.552 [0.216] | 3.769 [0.027] `**` | 3.583 [0.017] `**` |
| Dust | Sunshine | 0.504 [0.480] | 2.265 [0.109] `(t)` | 2.659 [0.053] `*` |

### 4.2 Significant Findings (p < 0.10)

| Catalogue | Variable | Lag | F-stat | p-value | Sig |
|---|---|---|---|---|---|
| Low Grown | Precipitation | 1 | 5.275 | 0.0238 | `**` |
| Low Grown | Precipitation | 2 | 4.414 | 0.0147 | `**` |
| Low Grown | Precipitation | 3 | 3.222 | 0.0263 | `**` |
| Low Grown | Sunshine | 1 | 5.607 | 0.0199 | `**` |
| Low Grown | Sunshine | 2 | 3.704 | 0.0283 | `**` |
| Low Grown | Sunshine | 3 | 2.399 | 0.0730 | `*` |
| Off-Grade | Temperature | 1 | 6.097 | 0.0152 | `**` |
| Off-Grade | Temperature | 2 | 3.572 | 0.0318 | `**` |
| Off-Grade | Temperature | 3 | 2.957 | 0.0363 | `**` |
| Dust | Temperature | 2 | 3.769 | 0.0265 | `**` |
| Dust | Temperature | 3 | 3.583 | 0.0167 | `**` |
| Dust | Sunshine | 3 | 2.659 | 0.0527 | `*` |

**12 significant relationships** found (p < 0.10) across 4 catalogues × 3 variables × 3 lags.

### 4.3 Key Findings by Weather Variable

- **Precipitation** → Granger-causes price in **Low Grown** at all lags 1–3 (strongest at lag 2, p = 0.015). No effect in High Grown, Off-Grade, or Dust.
- **Temperature** → Granger-causes price in **Off-Grade** at lags 1–3 (strongest at lag 1, p = 0.015) and in **Dust** at lags 2–3 (strongest at lag 3, p = 0.017). No effect in High Grown or Low Grown.
- **Sunshine** → Granger-causes price in **Low Grown** at lags 1–3 (strongest at lag 1, p = 0.020) and marginally in **Dust** at lag 3 (p = 0.053).

---

## 5. Feature Selection Recommendations

Derived from Granger causality results. Columns named as they appear in `final_clean_dataset_long.csv`.

| Catalogue | ADF (price) | Variable | Recommended Lag Columns | Best p |
|---|---|---|---|---|
| High Grown | 1st-diff | Precipitation | none significant | 0.172 |
| High Grown | 1st-diff | Temperature | none significant | 0.194 |
| High Grown | 1st-diff | Sunshine | N/A (constant series) | — |
| Low Grown | 1st-diff | Precipitation | `precipitation_sum_total_lag1`, `_lag2`, `_lag3` | **0.015** |
| Low Grown | 1st-diff | Temperature | none significant | 0.260 |
| Low Grown | 1st-diff | Sunshine | `sunshine_duration_total_lag1`, `_lag2`, `_lag3` | **0.020** |
| Off-Grade | levels | Precipitation | none significant | 0.448 |
| Off-Grade | levels | Temperature | `temperature_2m_mean_mean_lag1`, `_lag2`, `_lag3` | **0.015** |
| Off-Grade | levels | Sunshine | none significant | 0.111 |
| Dust | levels | Precipitation | none significant | 0.497 |
| Dust | levels | Temperature | `temperature_2m_mean_mean_lag2`, `_lag3` | **0.017** |
| Dust | levels | Sunshine | `sunshine_duration_total_lag3` | 0.053 |

### Recommended Feature Sets per Catalogue

**High Grown**: No weather lag features supported by Granger evidence. Rely on price lags and structural features (grade rank, tier rank, seasonality).

**Low Grown**: Include `precipitation_sum_total_lag1/2/3` and `sunshine_duration_total_lag1/2/3`. Price series requires differencing (non-stationary).

**Off-Grade**: Include `temperature_2m_mean_mean_lag1/2/3`. Price series is stationary in levels.

**Dust**: Include `temperature_2m_mean_mean_lag2/3` and `sunshine_duration_total_lag3`. Price series is stationary in levels.

---

## 6. Figures

The following plots are saved to `data/processed-2024/`:

| File | Description |
|---|---|
| `benchmark_rmse_by_catalogue.png` | RMSE bar chart per catalogue, all 4 models |
| `benchmark_r2_mape_heatmap.png` | R² and MAPE heatmaps across catalogue × model |
| `best_model_feature_importance.png` | Top-15 feature importances for best model per catalogue |
| `best_model_actual_vs_predicted.png` | Actual vs predicted scatter, best model per catalogue (test set) |
| `grade_level_rmse_breakdown.png` | RMSE per grade within each catalogue (best model) |
| `granger_acf_pacf.png` | ACF & PACF of price series per catalogue |
| `granger_pvalue_heatmap.png` | Granger p-value heatmap (−log₁₀ scale) |
| `granger_fstat_profiles.png` | F-statistic across lags per weather variable |
| `granger_timeseries_overlays.png` | Top significant weather–price overlay plots |
| `granger_adf_stationarity.png` | ADF p-values for all tested series |

---

## 7. Discussion of Granger Findings

The Granger results show that weather sensitivity is catalogue-specific rather than uniform across the market. The clearest signal appears in **Low Grown**, where both precipitation and sunshine significantly Granger-cause next-week prices at lags 1-3. This consistent multi-lag pattern suggests that Low Grown prices absorb weather information over several weeks, not only immediately, likely reflecting delayed effects through leaf growth, plucking volume, and short-horizon auction expectations.

For **Off-Grade**, temperature is the dominant driver, with significant causality at lags 1-3 and the strongest impact at lag 1. This indicates rapid price responsiveness to thermal conditions, followed by persistence into subsequent weeks. In practical terms, Off-Grade forecasting should prioritize short-horizon temperature dynamics, because temperature appears to carry both immediate and follow-through predictive value.

In **Dust**, temperature effects emerge mainly at lags 2-3, while sunshine is marginally significant at lag 3. Compared with Off-Grade, this lagged response implies a slower transmission channel from weather to price formation, potentially through blending behavior, inventory adjustments, or delayed quality sorting effects before auction.

**High Grown** shows no significant weather causality across tested variables and lags (with sunshine excluded as constant). This does not imply weather is irrelevant biologically; rather, within this sample and lag structure, weather does not add predictive power beyond autoregressive price behavior and structural market features. The finding supports a parsimonious feature strategy for High Grown models, centered on price lags, grade hierarchy, and seasonal signals.

Taken together, the evidence argues against a one-size-fits-all weather feature set. A pooled model with all weather lags may dilute signal in weather-insensitive catalogues while adding noise. Segment-wise feature selection aligned to Granger evidence is therefore justified: precipitation and sunshine lags for Low Grown, temperature lags for Off-Grade, temperature (plus selective sunshine) for Dust, and no weather lags for High Grown.

From an operational perspective, these results support catalogue-specific monitoring dashboards. Low Grown procurement and pricing teams should track rainfall and sunshine shocks continuously across the prior 1-3 weeks; Off-Grade teams should emphasize current and recent temperature anomalies; Dust teams should monitor temperature trends with a 2-3 week horizon. This lag-aware approach can improve weekly reserve-price setting and short-term trading decisions.
