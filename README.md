# Data Analysis for the Tea Industry  
## A Reproducible Research Workflow for Sri Lanka Tea Auction Price Analysis

## Overview

This repository presents a reproducible research workflow for studying price behavior in the Sri Lanka tea auction market. The project transforms weekly tea auction reports into structured analytical datasets and integrates market, production, and weather-related features for exploratory analysis and future predictive modeling.

The repository is organized as a research-focused workflow rather than a standalone software package. Its primary contribution is the transparent construction of analysis-ready data from semi-structured auction reports.

## Research Aim

The central aim of this project is to examine how tea auction prices vary across product categories and how those variations relate to weather conditions, auction volumes, production patterns, and broader market structure.

## Research Questions

The project is guided by the following questions:

1. What factors are associated with variation in Sri Lanka tea auction prices?
2. Do High Grown and Low Grown teas exhibit different price behavior?
3. Are weather conditions associated with auction prices directly or through lagged effects?
4. How do production, quantity sold, and offering composition relate to price movement?
5. Does segment-specific analysis provide better insight than treating the market as a single system?

## Scope of the Repository

This repository supports the following research tasks:

- extraction of structured information from weekly auction PDFs
- construction of normalized interim datasets
- enrichment with weather-based variables
- assembly of master and reduced analytical tables
- preprocessing for exploratory analysis and downstream modeling
- generation of figures and descriptive findings

## Repository Structure

```text
data-analysis-for-tea-industry/
│
├── README.md
├── requirements.txt
├── PROJECT_SUMMARY.md
├── SETUP.md
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── references/
│
├── src/
│   ├── ingestion/
│   │   ├── tea_pipeline_v2.py
│   │   └── weather_pipeline.py
│   └── processing/
│       ├── build_master_table.py
│       ├── build_reduced_master.py
│       └── preprocess_tea.py
│
├── notebooks/
│   ├── tea_eda.ipynb
│   ├── tea_eda_extended.ipynb
│   └── test.ipynb
│
└── reports/
    └── figures/
```

## Data Sources

The project is based on weekly Sri Lanka tea auction market reports and derived datasets.

### Primary Inputs
- weekly Forbes & Walker tea auction PDF reports
- extracted auction price and quantity tables
- textual market commentary from the reports
- historical weather data retrieved from external weather services

### Data Layers

#### Raw Data
The raw layer contains the original PDF auction reports.

#### Interim Data
The interim layer contains structured tables extracted from the raw reports, including:

- sales index data
- auction offerings
- quantity sold
- high-grown price tables
- low-grown price tables
- off-grade and dust price tables
- top price records
- generated column dictionary
- weather feature tables

#### Processed Data
The processed layer contains merged and transformed datasets used for analysis:

- full master analytical table
- reduced analytical table
- preprocessed dataset for EDA and modeling preparation

## Workflow

The workflow follows a staged research pipeline.

### 1. Data Ingestion
Auction PDFs are parsed into structured sale-level and category-level tables.

### 2. Weather Enrichment
Report-derived weather text and archived meteorological data are combined into region-based weather features.

### 3. Master Table Construction
Interim datasets are merged into a single analytical dataset.

### 4. Feature Reduction
Redundant, sparse, and high-collinearity variables are removed from the full master table.

### 5. Preprocessing
The reduced dataset is cleaned, transformed, and prepared for exploratory analysis and modeling.

### 6. Exploratory Analysis
Jupyter notebooks are used to evaluate price distributions, structural differences across tea groups, and possible relationships between weather and market behavior.

## Main Analytical Target

The principal outcome variable is:

- `price_mid_lkr`: midpoint of the reported auction price range in Sri Lankan Rupees

A transformed version is also used:

- `price_mid_lkr_log`: logarithm of midpoint price, used to reduce skew and improve interpretability in analysis

## Reproducibility

### Environment Setup

Clone the repository:

```bash
git clone https://github.com/hesandism/data-analysis-for-tea-industry.git
cd data-analysis-for-tea-industry
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Data Preparation

Place raw PDF files in:

```text
data/raw/
```

Use consistent source filenames such as:

```text
Sale of DD & DD Month YYYY.pdf
```

### Pipeline Execution

Run the scripts in the following order:

```bash
python src/ingestion/tea_pipeline_v2.py
python src/processing/build_master_table.py
python src/processing/build_reduced_master.py
python src/processing/preprocess_tea.py
```

### Notebook Execution

To run the main exploratory analysis:

```bash
jupyter notebook notebooks/tea_eda.ipynb
```

## Analytical Contribution

This repository contributes a structured workflow for converting semi-structured commodity market reports into research-ready data. It is particularly relevant for work in:

- agricultural economics
- commodity market analysis
- weather-linked market studies
- price behavior research
- applied feature engineering from document sources

## Preliminary Findings

Exploratory analysis conducted in this repository suggests the following broad patterns:

- price distributions are strongly right-skewed in raw form
- category and elevation structure are important determinants of price differences
- Low Grown and High Grown teas appear to exhibit different market behavior
- simple weather effects appear limited in isolation, but lagged and interaction effects may be more informative
- production, volume, and market structure likely operate jointly rather than independently

These observations should be interpreted as exploratory rather than causal.

## Methodological Notes

The project emphasizes reproducibility and traceability. Its structure is intended to make clear:

- how raw reports are transformed
- where each dataset is created
- how features are engineered
- which variables are used as targets, predictors, or excluded due to leakage risk

The repository is therefore suited both to exploratory market research and to future extension into forecasting or more formal statistical modeling.

## Limitations

Several limitations should be noted.

1. The current time coverage is limited and may not reflect long-run market dynamics.
2. PDF extraction may introduce parsing errors due to report-format variation.
3. Weather variables are region-linked approximations rather than estate-level observations.
4. Some documentation and notebook text may lag behind the latest dataset version.
5. Current outputs are descriptive and exploratory; they do not establish causal relationships.

## Recommended Future Development

The repository can be strengthened further by:

- extending the dataset across additional years
- formalizing schema and validation checks
- separating exploratory notebooks from final analytical notebooks
- adding a dedicated methodology document
- adding a final report or publication-ready summary
- building and evaluating baseline predictive models
- testing richer interaction and lag structures

## Suggested Citation

If this repository is used in academic or analytical work, please cite the project repository and acknowledge the original auction-report data sources.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0)

## Authors

**Nadil Kulathunge**
**Hesandi Mallawarachchi**
**Senilka**
**Thilokya Angeesa**
**Nethsith Gunaweera**

## Status

This repository is an evolving research workflow for tea auction price analysis. It is best understood as a reproducible analytical foundation for future research rather than a finalized production system.