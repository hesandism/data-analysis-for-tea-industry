# Research Plan: Data-Driven Analysis and Multi-Segment Forecasting of Weather Impacts on Tea Prices

> **Research Topic:** Data-Driven Analysis and Multi-Segment Forecasting of Weather Impacts on Tea Prices
> **Data Source:** Colombo Tea Auction Weekly Sale Reports (PDFs → `tea_output/`)
> **Scope:** Sale 01–10, Year 2026 (Jan–Mar); three growing regions: Western/Nuwara Eliya, Uva/Udapussellawa, Low Grown
> **Focal Grades:** High Grown BOPF · Medium Grown FBOP · Low Grown OP1

---

## 1. Dataset Architecture

Two master datasets will be constructed from the raw `tea_output` tables.

---

### 1.1 Dataset A — Comprehensive Analysis Dataset (`ds_analysis.csv`)

**Purpose:** Broad exploratory analysis, market dynamics, weather impact assessment, sentiment analysis, and supply-demand profiling. One row per `sale_id`.

| Source Table | Columns to Include | Notes |
|---|---|---|
| `01_sales_index` | `sale_id`, `sale_number`, `sale_year`, `sale_month`, `total_lots`, `total_kgs`, `reprint_lots`, `reprint_quantity` | Core auction metadata |
| `01_sales_index` | `commentary`, `sentiment_overall`, `sentiment_ex_estate`, `sentiment_low_grown` | NLP sentiment features |
| `01_sales_index` | `western_nuwara_eliya_weather_score`, `uva_udapussellawa_weather_score`, `low_grown_weather_score`, `avg_weather_severity` | Text-derived weather severity |
| `01_sales_index` | `western_nuwara_eliya_weather_desc`, `uva_udapussellawa_weather_desc`, `low_grown_weather_desc` | Raw weather text (qualitative reference) |
| `01_sales_index` | `crop_nuwara_eliya_trend`, `crop_western_trend`, `crop_uva_trend`, `crop_low_grown_trend` | Crop intake direction (−1 / 0 / +1) |
| `01_sales_index` | `private_sales_weekly_2026`, `public_auction_weekly_2026`, `forward_contracts_weekly_2026`, `total_sold_weekly_2026` | Current-year weekly volume channels |
| `01_sales_index` | `private_sales_weekly_2025`, `public_auction_weekly_2025`, `total_sold_weekly_2025` | YoY comparison |
| `01_sales_index` | `fx_usd_2026`, `fx_gbp_2026`, `fx_eur_2026`, `fx_jpy_2026` | LKR exchange rates (current year) |
| `01_sales_index` | `sl_production_mkgs`, `sl_production_yoy_variance` | National production & YoY change |
| `01_sales_index` | All `gross_lkr_weekly_*_2026` columns | Weekly average gross LKR by segment/elevation |
| `02_auction_offerings` | `category`, `qty_mkgs`, `demand_score`, `lots`, `kgs` | Demand and supply per category — join on `sale_id` |
| `03_quantity_sold` | Already captured in `01_sales_index`; skip duplicates | — |
| `04_high_grown_prices` | `segment`, `grade`, `price_lo_lkr`, `price_hi_lkr` | Pivot or keep long — join on `sale_id` |
| `05_low_grown_prices` | `grade`, `tier`, `price_lo_lkr`, `price_hi_lkr` | Join on `sale_id` |
| `06_offgrade_dust_prices` | `category_type`, `category`, `elevation`, `price_lo_lkr`, `price_hi_lkr` | Join on `sale_id` |
| `07_top_prices` | `estate`, `grade`, `fw_sold`, `price_lkr`, `region` | For top-price analysis; join on `sale_id` |

**Engineered columns to derive:**
- `price_mid_lkr` = `(price_lo_lkr + price_hi_lkr) / 2` — midpoint price per grade/segment
- `price_range_lkr` = `price_hi_lkr - price_lo_lkr` — spread as a quality/uncertainty proxy
- `sold_to_offered_ratio` = `total_sold_weekly_2026 / total_kgs` — clearance rate
- `yoy_volume_change_pct` = `(total_sold_weekly_2026 - total_sold_weekly_2025) / total_sold_weekly_2025 × 100`

**What to exclude from Dataset A:**
- Settlement dates (`settlement_10pct`, `settlement_buyers`, `settlement_sellers`) — administrative, not analytical
- `source_file`, `extracted_at` — pipeline metadata
- `sale_date_raw` — use structured `sale_year` + `sale_month` + `sale_number` instead
- `crop_notes` — text duplicate of structured `crop_*_trend` columns
- `private_sales_todate_*`, `public_auction_todate_*`, `forward_contracts_todate_*` — cumulative YTD columns contain compounding noise; weekly columns are more comparable
- FX for 2024 and 2025 years (keep only current year for modelling; retain 2025 for YoY comparisons in EDA)
- `gross_lkr_todate_*` columns — use weekly for consistency
- `fw_sold` from top_prices — broker flag, not market-level

---

### 1.2 Dataset B — Weather-Price Prediction Dataset (`ds_prediction.csv`)

**Purpose:** Machine learning model input. One row per `(sale_id, region)` — three rows per auction (High Grown Western/NE, High Grown Uva/Udapussellawa, Low Grown). Weather data is spatially matched to region.

| Source Table | Columns to Include | Notes |
|---|---|---|
| `09_weather_features` | `sale_id`, `auction_date`, `region` | Core identifiers — lat/lon dropped (fixed per region) |
| `09_weather_features` | `text_condition_score`, `text_crop_change`, `text_has_rain`, `text_has_mist`, `text_has_bright` | Text-derived crop and condition signals — primary qualitative features |
| `09_weather_features` | `precipitation_sum_total`, `sunshine_duration_total`, `temperature_2m_mean_mean` | One representative per weather dimension: rainfall, sunlight, temperature |
| `09_weather_features` | `relative_humidity_2m_max_mean` | Single humidity indicator |
| `09_weather_features` | `precipitation_sum_total_lag1`, `precipitation_sum_total_lag2` | Rain lag features — 1 and 2 weeks prior (lag3 dropped; diminishing signal) |
| `09_weather_features` | `sunshine_duration_total_lag1`, `sunshine_duration_total_lag2` | Sunshine lag features — 1 and 2 weeks prior |
| `09_weather_features` | `text_condition_score_lag1`, `text_condition_score_lag2` | Lagged text condition — captures prior-week crop report signal |
| `01_sales_index` | `fx_usd_2026`, `sl_production_mkgs` | Key macro controls: currency and national supply |
| `01_sales_index` | `sentiment_overall` | Market sentiment from commentary |
| `02_auction_offerings` | `demand_score` (grade-matched category only) | Demand score for the focal grade's auction category |

**Dropped from earlier version:**
- All `*_max_max`, `*_max_min`, `*_min_*` temperature variants — `temperature_2m_mean_mean` is sufficient
- `rain_sum_total`, `precipitation_sum_max_day` — redundant with `precipitation_sum_total`
- `windspeed_10m_max_mean` — low relevance to price for these grades
- `et0_fao_evapotranspiration_total` — correlated with temperature + sunshine; removed to reduce multicollinearity
- `relative_humidity_2m_min_mean` — keep only the max mean
- `sunshine_duration_max_day` — weekly total is sufficient
- `*_lag3` columns — lag 1 and 2 capture the processing window adequately
- `text_has_thunder` — very rare event, low signal
- `lat`, `lon`, `region_label` — fixed constants per region, not model features
- `sl_production_yoy_variance` — correlated with `sl_production_mkgs`; use the level only

**Target variables:**

| Target | Source | Grade |
|---|---|---|
| `price_mid_bopf_lkr` | `04_high_grown_prices`, grade = `bopf`, midpoint per sub-segment | BOPF |
| `price_mid_pekoe_fbop_lkr` | `04_high_grown_prices`, grade = `pekoe_fbop`, midpoint per sub-segment | FBOP |
| `price_mid_op1_lkr` | `05_low_grown_prices`, grade = `op1`, midpoint per tier | OP1 |
| `price_direction` | Week-over-week change of the grade's gross LKR summary | All (classification) |

---

## 2. Focal Grade Definitions

The analysis is narrowed to three specific grades, each representing a distinct elevation and quality profile. All analyses in Phases 1–3 are conducted **independently per grade**, then compared cross-grade in Phase 4.

---

### Grade 1 — High Grown BOPF

| Attribute | Detail |
|---|---|
| **Grade code** | `bopf` |
| **Source table** | `04_high_grown_prices` |
| **Filter** | `elevation == 'high_grown'` AND `grade == 'bopf'` |
| **Sub-segments available** | `best_western`, `below_best_western`, `plainer_western`, `nuwara_eliya`, `brighter_udapussellawa`, `other_udapussellawa`, `best_uva`, `other_uva` |
| **Elevation** | High grown (above 1200m): Western slopes, Nuwara Eliya, Uva, Udapussellawa |
| **Auction category** | `ex_estate` + `high_medium` in `02_auction_offerings` |
| **Weather region** | `western_nuwara_eliya` and `uva_udapussellawa` (from `09_weather_features`) |
| **Gross LKR reference** | `gross_lkr_weekly_western_high_2026`, `gross_lkr_weekly_uva_high_2026` |
| **Character** | Broken Orange Pekoe Fannings — liquoring, bright cup; key export grade for UK and CIS markets |

**Note on sub-segments:** Each sub-segment has its own price range. Analysis will track all sub-segments independently and also compute a weighted/unweighted average BOPF price per sale as the summary metric. `best_western` and `nuwara_eliya` BOPF command premiums and are the most sensitive to seasonal quality shifts.

---

### Grade 2 — Medium Grown FBOP (Pekoe/FBOP)

| Attribute | Detail |
|---|---|
| **Grade code** | `pekoe_fbop` |
| **Source table** | `04_high_grown_prices` |
| **Filter** | `elevation == 'high_grown'` AND `grade == 'pekoe_fbop'` |
| **Sub-segments available** | `best_western`, `below_best_western`, `plainer_western`, `nuwara_eliya`, `brighter_udapussellawa`, `other_udapussellawa`, `best_uva`, `other_uva` |
| **Elevation** | Mid/high grown (medium-elevation Western slopes, Uva, Udapussellawa) |
| **Auction category** | `high_medium` in `02_auction_offerings` |
| **Weather region** | `western_nuwara_eliya` and `uva_udapussellawa` (from `09_weather_features`) |
| **Gross LKR reference** | `gross_lkr_weekly_western_medium_2026`, `gross_lkr_weekly_uva_medium_2026`, `gross_lkr_weekly_ctc_medium_2026` |
| **Character** | Flowery Broken Orange Pekoe — whole-leaf type, Japan/specialty market grade; high price premium for select invoices; largest price dispersion across segments |

**Note on grade label:** In the auction reports, Pekoe and FBOP are quoted together as a combined price range (`pekoe_fbop` column). The `brighter_udapussellawa` and `best_uva` sub-segments for this grade consistently command the highest ranges and are most weather-sensitive (premium quality from specific microclimates).

---

### Grade 3 — Low Grown OP1

| Attribute | Detail |
|---|---|
| **Grade code** | `op1` |
| **Source table** | `05_low_grown_prices` |
| **Filter** | `elevation == 'low_grown'` AND `grade == 'op1'` |
| **Tiers available** | `select_best`, `best`, `below_best`, `others` |
| **Elevation** | Low grown (below 600m): Matara, Galle, Ratnapura, Sabaragamuwa |
| **Auction category** | `leafy`, `semi_leafy` in `02_auction_offerings` |
| **Weather region** | `low_grown` (from `09_weather_features`) |
| **Gross LKR reference** | `gross_lkr_weekly_orthodox_low_2026` |
| **Character** | Orange Pekoe 1 — large whole-leaf, specialty/speciality market; highest price grade in low grown; `select_best` tier commands Rs. 3,000–3,900/kg (the most premium low grown price range in the dataset) |

**Note on tiers:** OP1 is structured by quality tier (`select_best` → `others`), not by region sub-segment. The spread between `select_best` and `others` is extreme (often Rs. 2,000+/kg difference), making tier-level tracking essential. The `select_best` tier is the premium signal; the `others` tier tracks the floor price.

---

### Cross-Grade Comparison Summary

| | High Grown BOPF | Medium Grown FBOP | Low Grown OP1 |
|---|---|---|---|
| **Source** | `04_high_grown_prices` | `04_high_grown_prices` | `05_low_grown_prices` |
| **Grade column** | `bopf` | `pekoe_fbop` | `op1` |
| **Breakdown dimension** | Segment (region quality tier) | Segment (region quality tier) | Quality tier |
| **Weather table region** | `high_grown` (western + uva) | `high_grown` (western + uva) | `low_grown` |
| **Typical price range (LKR/kg)** | 1,000–1,700 | 700–1,700 | 900–3,900 |
| **Price driver** | Liquoring quality, seasonal flush | Leaf appearance + liquor, Japan demand | Leaf size, appearance, CIS/Middle East |
| **Key competitor markets** | UK, CIS, Middle East | Japan, specialty buyers | CIS, Türkiye, Iran |

---

## 3. Analysis Plan

### Phase 1: Exploratory Data Analysis (EDA)

#### 1a. Temporal Price Dynamics
- Line charts of weekly `gross_lkr` by segment: High Grown (Western, Uva, CTC), Low Grown (Orthodox, CTC)
- Week-over-week price change (`Δ LKR/kg`) across all sale numbers
- Price range spread (`price_hi − price_lo`) over time per grade — measures quality variance
- YoY price comparison: 2026 vs 2025 gross averages

#### 1b. Supply and Demand Analysis
- Total lots and kgs offered per sale vs quantity sold — clearance rate trend
- Volume breakdown by channel: public auction vs private sales vs forward contracts
- Category-level demand scores over time (ex_estate, leafy, semi_leafy, tippy, premium_flowery, off_grade, dust)
- Reprint lot fraction as a quality/overstock signal

#### 1c. Weather Pattern Analysis
- Heatmap of weather severity scores by region × sale number
- Rainfall timeline (actual mm from `09_weather_features`) vs text-reported severity score — validate text scoring
- Sunshine duration and temperature trends across the season
- Humidity profiles by region
- Crop change direction frequency by region

#### 1d. FX and Macro Context
- LKR/USD over time and its inverse correlation with international buying activity
- SL national production monthly trend and YoY variance
- Relationship between FX weakness and market sentiment

#### 1e. Sentiment Analysis
- Distribution of `sentiment_overall` across sales
- Correlation between `sentiment_overall` and `gross_lkr_weekly_total_2026`
- Segment-level sentiment vs corresponding prices (ex_estate, low_grown)

#### 1f. Top Price Analysis
- Top-performing estates by average `price_lkr` and grade
- Regional comparison of top prices: Western High vs Nuwara Eliya vs Uva vs Low Grown
- Broker (Forbes & Walker) premium: `fw_sold = 1` vs `fw_sold = 0` average prices

---

### Phase 1G: Per-Grade Weekly Analysis (BOPF · FBOP · OP1)

Each of the three focal grades is analysed independently in this phase. The structure below is applied to each grade separately, then results are compared in Phase 4.

---

#### Grade 1 — High Grown BOPF Weekly Analysis

**1G-A. Price Level and Range — Weekly**
- Line chart: `price_mid_lkr` (midpoint of lo/hi) per sub-segment over sale numbers 1–10
  - Sub-segments tracked: `best_western`, `below_best_western`, `plainer_western`, `nuwara_eliya`, `brighter_udapussellawa`, `other_udapussellawa`, `best_uva`, `other_uva`
- Line chart: `price_range_lkr` (`price_hi − price_lo`) per sub-segment — tracks quality spread width over time
- Summary metric: unweighted mean BOPF price across all sub-segments per sale (single trend line)
- Stacked area chart: price band (lo to hi) per sub-segment by sale — visual of full market range

**1G-B. Week-over-Week Price Change**
- Bar chart: `Δ price_mid_lkr` from previous sale, per sub-segment
- Highlight: which sub-segment is most volatile vs most stable week-to-week
- Flag outlier sale events (e.g., Sale 10 Middle East tension, Sale 6 seasonal Western flush)

**1G-C. Sub-Segment Premium/Discount Patterns**
- Compute premium of `best_western` BOPF over `plainer_western` BOPF each week — seasonal premium spread
- Compute `nuwara_eliya` vs `best_uva` BOPF price differential — regional rivalry
- Chart: rolling premium ratios over sales 1–10

**1G-D. YoY Comparison**
- Compare `gross_lkr_weekly_western_high_2026` vs `gross_lkr_weekly_western_high_2025`
- Compare `gross_lkr_weekly_uva_high_2026` vs `gross_lkr_weekly_uva_high_2025`
- Identify sales where 2026 is outperforming or underperforming 2025

**1G-E. Demand Context**
- Overlay BOPF price with `demand_score` from `02_auction_offerings` for `ex_estate` category
- Check if demand score leads or lags the price movement

---

#### Grade 2 — Medium Grown FBOP (pekoe_fbop) Weekly Analysis

**1G-A. Price Level and Range — Weekly**
- Line chart: `price_mid_lkr` for `pekoe_fbop` per sub-segment over sale numbers 1–10
  - Sub-segments: `best_western`, `below_best_western`, `plainer_western`, `nuwara_eliya`, `brighter_udapussellawa`, `other_udapussellawa`, `best_uva`, `other_uva`
- Note: `pekoe_fbop` typically has the **widest price range** of all four high-grown grades — this spread reflects the combined Pekoe + FBOP price band in the report
- Line chart: `price_range_lkr` — track whether the pekoe/FBOP spread widens or narrows over the season

**1G-B. Week-over-Week Price Change**
- Bar chart: `Δ price_mid_lkr` per sub-segment, sale-over-sale
- Focus: `brighter_udapussellawa` and `best_uva` show the highest variance — track these separately
- Flag: Sale 7 onward expected to show premium seasonal Westerns driving up `best_western` pekoe_fbop

**1G-C. Sub-Segment Premium Patterns**
- Premium of `brighter_udapussellawa` pekoe_fbop over `plainer_western` pekoe_fbop — Udapussellawa quality signal
- Track whether `nuwara_eliya` pekoe_fbop emerges later in season (seasonal flowering pattern)

**1G-D. YoY Comparison**
- Compare `gross_lkr_weekly_western_medium_2026` vs `gross_lkr_weekly_western_medium_2025`
- Compare `gross_lkr_weekly_uva_medium_2026` vs `gross_lkr_weekly_uva_medium_2025`
- Compare `gross_lkr_weekly_ctc_medium_2026` vs `gross_lkr_weekly_ctc_medium_2025`

**1G-E. Japan/Specialty Market Demand Signal**
- Commentary mentions Japan demand for liquoring teas explicitly — track sentiment mentions
- Overlay: `sentiment_overall` with `price_mid` for `best_western` pekoe_fbop — specialty demand proxy

---

#### Grade 3 — Low Grown OP1 Weekly Analysis

**1G-A. Price Level by Tier — Weekly**
- Line chart: `price_mid_lkr` for `op1` per quality tier (`select_best`, `best`, `below_best`, `others`) over sales 1–10
- Note: `select_best` OP1 is the highest-priced grade in the entire dataset (Rs. 3,100–3,900/kg in recent sales) — track as premium signal
- Line chart: `price_range_lkr` per tier — the `others` tier has very wide spreads (Rs. 900–1,600) reflecting heterogeneous quality

**1G-B. Week-over-Week Price Change**
- Bar chart: `Δ price_mid_lkr` per tier, sale-over-sale
- Focus: `select_best` tier — driven by specialty/Japan buyers; `others` tier — driven by CIS/Middle East volume buyers
- Flag: whether `select_best` and `others` move in sync or diverge (divergence indicates demand stratification)

**1G-C. Tier Premium Spread Analysis**
- Chart: `select_best` OP1 price − `others` OP1 price each week — premium spread trend
- A widening spread indicates strengthening specialty demand; narrowing spread indicates weakening
- Compare this spread to low_grown demand score from `02_auction_offerings`

**1G-D. YoY Comparison**
- Compare `gross_lkr_weekly_orthodox_low_2026` vs `gross_lkr_weekly_orthodox_low_2025`
- OP1 is orthodox (not CTC), so use the orthodox low grown gross LKR series

**1G-E. Middle East / CIS Demand Sensitivity**
- OP1 in the `others` tier is heavily influenced by CIS and Middle East buyer activity (mentioned in commentary)
- Overlay: sale weeks with commentary mentions of "CIS", "Middle East", "Iran" with `others` tier OP1 price
- Cross-reference Sale 9 and 10 (Middle East tension) — expected price dip in `others` tier

**1G-F. Volume and Clearance**
- Track `kgs` offered for `leafy` + `semi_leafy` categories (OP1's auction categories) vs demand_score
- Compare weeks where demand is `good` vs `fair` vs `less` to see price impact

---

### Phase 1H: Cross-Grade Comparison

After completing individual grade analyses, compare across BOPF, FBOP, and OP1:

- **Price level comparison:** Plot all three grade summary price lines on one chart (requires normalisation — use % change from Sale 1 baseline, since absolute LKR levels differ significantly)
- **Volatility comparison:** Standard deviation of week-over-week `Δ price_mid` per grade — which is most stable?
- **Weather sensitivity ranking:** Which grade's prices move most in response to the same weather event?
- **YoY performance:** Which grade gained/lost most vs 2025?
- **Price range spread:** Which grade has the most quality uncertainty (widest lo-hi range)?
- **Seasonal pattern:** Do all three grades follow the same seasonal arc, or does BOPF peak while OP1 does not?

---

### Phase 2: Weather–Price Correlation Analysis

#### 2a. Bivariate Correlations
- Spearman correlation matrix: weather variables vs price midpoints per region
- Key relationships to test:
  - `precipitation_sum_total` ↔ `price_mid` (excess rain → flush growth → lower quality → lower prices)
  - `sunshine_duration_total` ↔ `price_mid` (more sun → better quality → higher prices)
  - `relative_humidity_2m_max_mean` ↔ `price_mid`
  - `et0_fao_evapotranspiration_total` ↔ `crop_*_trend`
  - `text_condition_score` ↔ `price_mid` (validate rule-based scores)

#### 2b. Lagged Effect Analysis
- Scatter plots and correlations: `precipitation_lag1`, `lag2`, `lag3` vs current-week prices
- Determine optimal lag length (weather 1–3 weeks prior)
- Rationale: tea harvested in week T is sold at auction T+1 to T+3 depending on processing time

#### 2c. Grade-Stratified Correlation

Run the full correlation analysis separately for each focal grade, matched to its weather region:

| Grade | Weather region (from `09_weather_features`) | Price target |
|---|---|---|
| BOPF | `western_nuwara_eliya` + `uva_udapussellawa` | `price_mid_bopf` per sub-segment |
| FBOP (pekoe_fbop) | `western_nuwara_eliya` + `uva_udapussellawa` | `price_mid_pekoe_fbop` per sub-segment |
| OP1 | `low_grown` | `price_mid_op1` per tier |

For BOPF and FBOP, use weather from the sub-segment's corresponding region:
- `best_western`, `below_best_western`, `plainer_western`, `nuwara_eliya` → `western_nuwara_eliya` weather
- `brighter_udapussellawa`, `other_udapussellawa`, `best_uva`, `other_uva` → `uva_udapussellawa` weather

#### 2d. Regression Analysis — Per Grade

Run separate OLS regressions for each grade:

- **BOPF:** `price_mid_bopf ~ precipitation_western + sunshine_western + humidity_western + precipitation_uva + lag1_precipitation + FX + sentiment`
- **FBOP:** `price_mid_fbop ~ precipitation_western + sunshine_udapussellawa + humidity_uva + lag1_precipitation + lag2_precipitation + FX + production`
- **OP1:** `price_mid_op1 ~ precipitation_low_grown + rain_sum_total + sunshine_low_grown + humidity_low_grown + lag1_rain + FX + demand_score_leafy`
- Include interaction terms: `precipitation × humidity` (fog/mist effect on BOPF quality)
- Check VIF for multicollinearity
- Residual analysis to identify sale-level anomalies (e.g., Middle East demand shock in Sale 9–10 affecting OP1 `others` tier)

---

### Phase 3: Multi-Segment Price Forecasting

#### 3a. Problem Framing

Two forecasting tasks:

| Task | Type | Target | Horizon |
|---|---|---|---|
| Price Level Forecasting | Regression | `price_mid_lkr` per segment-region | 1 week ahead |
| Price Direction Forecasting | Binary Classification | Up / Down vs previous week | 1 week ahead |

Grades modelled independently (one model pipeline per grade):

| Model ID | Grade | Target Variable | Breakdown |
|---|---|---|---|
| M1 | High Grown BOPF | `price_mid_bopf_lkr` | Per sub-segment (best_western, nuwara_eliya, uva, etc.) |
| M2 | Medium Grown FBOP | `price_mid_pekoe_fbop_lkr` | Per sub-segment (best_western, udapussellawa, uva, etc.) |
| M3 | Low Grown OP1 | `price_mid_op1_lkr` | Per tier (select_best, best, below_best, others) |
| M4 | BOPF direction | Binary Up/Down | `gross_lkr_weekly_high_summary_2026` WoW change |
| M5 | FBOP direction | Binary Up/Down | `gross_lkr_weekly_medium_summary_2026` WoW change |
| M6 | OP1 direction | Binary Up/Down | `gross_lkr_weekly_orthodox_low_2026` WoW change |

#### 3b. Feature Engineering

**Time features:**
- `sale_number` (proxy for seasonality within year)
- `sale_month_encoded` (cyclical sin/cos encoding)

**Weather features (current week):**
- `precipitation_sum_total`, `sunshine_duration_total`, `temperature_2m_mean_mean`, `relative_humidity_2m_max_mean`, `et0_fao_evapotranspiration_total`
- Text flags: `text_has_rain`, `text_has_mist`, `text_has_bright`

**Lagged weather features:**
- All `*_lag1`, `*_lag2`, `*_lag3` (already available in `09_weather_features`)

**Market features:**
- `demand_score` per category
- `sentiment_overall`
- `sold_to_offered_ratio` (clearance rate)
- `sl_production_mkgs`
- `sl_production_yoy_variance`
- `fx_usd_2026`

**Lagged price:**
- `price_mid_lkr_lag1` — previous week's price (autoregressive component)

#### 3c. Model Pipeline

**Step 1 — Baseline Models**
- Naive last-value (persistence model): predict = previous week's price
- Seasonal naive: predict = same week last year (if multi-year data becomes available)
- ARIMA on the price series (univariate, weather-blind baseline)

**Step 2 — ML Models (weather-informed)**
- Ridge / Lasso Regression: linear model with regularisation; establishes a transparent weather coefficient
- Random Forest Regressor: captures non-linear interactions; provides feature importance
- XGBoost / LightGBM: gradient boosting for tabular data; handles the small dataset well
- SVR (Support Vector Regression): robust to small sample sizes

**Step 3 — Deep Learning (if dataset grows)**
- LSTM with weather + price sequence input (requires ≥ 30 data points per segment)
- Will be applicable when multi-year data is incorporated

**Classification variants:**
- Logistic Regression, Random Forest Classifier, XGBoost Classifier for direction (Up/Down)

#### 3d. Validation Strategy

Given the small number of sales (10 at extraction time), the following validation approach is used:

- **Time-based leave-one-out (LOOCV):** Train on sales 1–(n−1), test on sale n; roll forward
- **Walk-forward validation:** incrementally expand training window; evaluate on each new week
- No random shuffle — data is temporal and shuffling would leak future information

#### 3e. Evaluation Metrics

| Metric | Task | Interpretation |
|---|---|---|
| MAE (LKR/kg) | Regression | Average absolute price error |
| RMSE (LKR/kg) | Regression | Penalises large errors more |
| MAPE (%) | Regression | Scale-independent error |
| Directional Accuracy (%) | Both | % of correct Up/Down calls |
| F1-Score | Classification | Precision-recall balance for direction |

---

## 3. Methodology Summary

### 3.1 Data Wrangling
1. Join all `tea_output` CSVs on `sale_id`
2. Compute derived columns: `price_mid`, `price_range`, `clearance_rate`, `yoy_volume_change_pct`
3. Pivot `04_high_grown_prices` from long to wide (segment × grade columns) — or model per-segment separately
4. Align `09_weather_features` to auction sale by `sale_id` and `region`
5. Handle missing values: `gross_lkr` columns are NaN for early sales (data sparsity) — use forward-fill cautiously within segments; flag and exclude from regression if too sparse

### 3.2 Statistical Testing
- Normality: Shapiro-Wilk on price variables → guides use of Pearson vs Spearman
- Stationarity: ADF test on price time series → informs ARIMA differencing order
- Granger Causality: test whether lagged weather variables Granger-cause price changes per segment
- Multicollinearity: VIF scores on regression feature matrix; drop or PCA-reduce if VIF > 10

### 3.3 Prediction Workflow
```
Raw CSVs
  → ds_prediction.csv (feature matrix per sale × region)
  → Feature engineering (lag prices, cyclical encoding)
  → Train/Test split (walk-forward)
  → Fit: Baseline → Ridge/Lasso → Random Forest → XGBoost
  → Evaluate on held-out sales (MAE, MAPE, Directional Accuracy)
  → Interpret: SHAP values for XGBoost feature importances
  → Report: Which weather variables most influence which segment
```

### 3.4 Interpretability
- SHAP (SHapley Additive Explanations) for tree models: explains per-prediction contribution of each feature
- Partial Dependence Plots: shows marginal effect of precipitation / sunshine on price per segment
- Coefficient plots for Ridge regression: directly interpretable weather price elasticities

---

## 4. Expected Findings (Hypotheses to Test)

| Hypothesis | Grade | Mechanism | Region |
|---|---|---|---|
| High rainfall (lag 1–2 weeks) → lower BOPF price | BOPF | Flush growth dilutes liquoring quality; higher supply depresses price | Western, Nuwara Eliya |
| High sunshine duration → premium `best_western` BOPF seasonal flush → price appreciation | BOPF | Photosynthesis + slow drying enhances flavour; seasonal quality drives ex-estate premiums | Western slopes |
| Misty / dull conditions → improved Nuwara Eliya BOPF quality → dearer `nuwara_eliya` sub-segment | BOPF | Slow growth in cool mist concentrates flavour compounds in BOPF grade | Nuwara Eliya |
| High sunshine + low humidity → premium `brighter_udapussellawa` FBOP price spike | FBOP | Brighter appearance and denser leaf favoured by Japan buyers; clear days improve leaf grading | Udapussellawa |
| Wide pekoe_fbop price range (`price_range_lkr`) → signals quality-driven market, not volume-driven | FBOP | When weather produces heterogeneous quality, spread between pekoe and FBOP widens | Western, Uva |
| Excess rainfall in Low Grown belt → increased flush volume → `others` tier OP1 prices weaken | OP1 | Large leaf supply from mass flush reduces sorting quality; CIS/Middle East buyers are price-elastic | Low Grown (Matara/Galle) |
| `select_best` OP1 is weather-insensitive; `others` tier OP1 is highly weather-sensitive | OP1 | Specialty buyers for `select_best` focus on leaf appearance (stable); bulk buyers sensitive to volume | Low Grown |
| LKR depreciation → bearish market sentiment → lower prices across all three grades | All | Reduced purchasing power parity for international buyers; markdown pressure from brokers | All |
| Lag 1–2 week weather is a stronger predictor than same-week weather | All | Tea harvested in week T enters auction T+1 to T+3 depending on processing time | All |
| `select_best` OP1 − `others` OP1 spread widens when Middle East demand weakens | OP1 | Middle East buyers absorb `others` tier; when absent, floor price drops, widening the tier gap | Low Grown |

---

## 5. File Structure for Implementation

```
extract-csv/
├── tea_output/                     # Raw extracted CSVs (source)
│   ├── 01_sales_index.csv
│   ├── 02_auction_offerings.csv
│   ├── 03_quantity_sold.csv
│   ├── 04_high_grown_prices.csv
│   ├── 05_low_grown_prices.csv
│   ├── 06_offgrade_dust_prices.csv
│   ├── 07_top_prices.csv
│   ├── 08_column_dictionary.csv
│   └── 09_weather_features.csv
├── datasets/                       # To be created
│   ├── ds_analysis.csv             # Dataset A — comprehensive EDA dataset
│   └── ds_prediction.csv           # Dataset B — ML feature matrix (sale × region)
├── notebooks/                      # To be created
│   ├── 01_eda.ipynb                # Phase 1: Exploratory analysis
│   ├── 02_weather_correlation.ipynb # Phase 2: Weather–price relationships
│   └── 03_forecasting.ipynb        # Phase 3: Multi-segment prediction models
├── RESEARCH_PLAN.md                # This document
└── tea_pipeline_v2.py              # Extraction pipeline
```

---

## 6. Limitations and Considerations

- **Small sample size:** 10 sales at time of writing. Walk-forward validation will have very few test points. Results should be treated as directional, not definitive. Extending to multi-year historical data is strongly recommended before finalising models.
- **Weather data granularity:** Weather is fetched for fixed coordinates per region. Actual plantation-level microclimates vary. Text-based weather descriptions from PDFs provide a useful qualitative cross-check.
- **Confounding factors:** Middle East political events, CIS trade restrictions, and currency shocks are reflected in sentiment scores but not fully quantifiable. These should be noted as confounders in regression residual analysis.
- **Price ranges vs averages:** Auction reports give price ranges, not transaction-weighted averages. Midpoint prices are approximations. The `gross_lkr_weekly_*` columns from the summary table are more reliable aggregates for regression.
- **Grade heterogeneity within segments:** BOP and BOPF have very different demand profiles within the same segment. Models per segment should ideally be per grade; however, this further reduces observations per series.
