# Research Metadata

## Overview
This document consolidates all resource URLs, API configurations, geographic coordinates, data coverage dates, and other metadata related to the Sri Lanka Tea Auction Price Forecasting research.

---

## Weather API Information

### Open-Meteo API
- **Service**: Free historical weather data
- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **Authentication**: No API key required
- **Usage**: Used to fetch historical meteorological data for all four Sri Lanka tea-growing regions
- **Data Format**: JSON

### Weather Variables Tracked
The following daily meteorological variables are fetched from Open-Meteo:
- `temperature_mean` — Mean temperature (°C)
- `precipitation_sum` — Total daily precipitation (mm)
- `rain_sum` — Total daily rain sum (mm)
- `windspeed_max` — Maximum 10m wind speed (km/h)
- `sunshine_duration` — Daily bright sunshine duration (seconds)
- `relative_humidity_2m_max` — Maximum relative humidity at 2m (%)
- `relative_humidity_2m_min` — Minimum relative humidity at 2m (%)

### Weather Lookback Period
- **LOOKBACK_DAYS**: 7 days
- **Rationale**: Weather data is fetched for the 7-day crop week BEFORE each auction date, representing the weather conditions that influenced the teas on offer

---

## Geographic Coordinates

### Tea-Growing Regions in Sri Lanka

#### 1. Western High Grown (Maskeliya / Dickoya)
- **Latitude**: 6.9271°N
- **Longitude**: 80.5350°E
- **Elevation**: High altitude tea-growing region
- **Coordinates Type**: Centroid of region

#### 2. Nuwara Eliya
- **Latitude**: 6.9497°N
- **Longitude**: 80.7891°E
- **Elevation**: High altitude tea-growing region
- **Coordinates Type**: Centroid of region

#### 3. Uva / Uda Pussellawa
- **Latitude**: 6.8700°N
- **Longitude**: 81.0600°E
- **Elevation**: Mid to high altitude tea-growing region
- **Coordinates Type**: Centroid of region

#### 4. Low Grown (Matara / Galle Belt)
- **Latitude**: 6.2500°N
- **Longitude**: 80.3000°E
- **Elevation**: Lower altitude tea-growing region
- **Coordinates Type**: Centroid of region

---

## PDF Report Data Coverage

### Report Source
- **Publisher**: Forbes & Walker (Sri Lanka Tea Auction Market Reports)
- **Report Type**: Weekly Tea Market Reports
- **Report Location**: `data/Raw/2025/` (and historical years)

### Sale ID Convention
Sales are internally identified as:
- **Format**: `SALE_{year}_{sale_number:02d}`
- **Example**: `SALE_2025_32`

---

## Research Data Artifacts and Dates

### Dataset Timeline
- **Start Date**: 2023
- **End Date**: 2026 (ongoing)
- **Analysis Period**: Multi-year time series with weekly observations

### Processed Datasets
- `final_clean_dataset_long.csv`: 
  - **Rows**: 12,233
  - **Features**: 26 primary features
  - **Coverage**: 2023–2026
  - **Ready For**: Notebook analysis and modeling

### Result Output Dates
- **Segment-Specific Models**: Evaluated on segment-wise sales (High Grown n=90, Low Grown n=104, Off-Grade n=105, Dust n=105)
- **Unified Model**: Evaluated on pooled n=11,739 observations
- **Validation Protocol**: 5-fold TimeSeriesSplit (80/20 temporal train/test split)

---

## Catalog / Product Segments

The research analyzes four distinct catalogue segments:
1. **High Grown** — Premium high-altitude tea (n=90 sales)
2. **Low Grown** — Lower altitude tea (n=104 sales)
3. **Off-Grade** — Off-grade quality tea (n=105 sales)
4. **Dust** — Dust/fannings category (n=105 sales)

---
