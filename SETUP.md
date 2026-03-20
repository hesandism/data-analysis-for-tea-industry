# Environment Setup Guide

This guide explains how to set up the development environment for the Sri Lanka Tea Auction Research Project.

## Prerequisites

- Python 3.11.4 or higher
- pip (included with Python)
- git

## Quick Start

### 1. Clone or Navigate to Repository
```bash
cd {path_to_your_project}
```

### 2. Create Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import pandas, numpy, pdfplumber, requests, sklearn; print('All dependencies installed successfully!')"
```

## Project Structure

```
extract-csv/
├── data/
│   ├── Raw/              # Raw PDF reports
│   ├── Interim/          # Extracted structured CSVs
│   └── Processed/        # Analysis-ready datasets
├── src/
│   ├── ingestion/        # PDF parsing and data extraction
│   └── processing/       # Data transformation and preprocessing
├── notebooks/            # Jupyter notebooks for EDA and analysis
├── reports/              # Output figures and reports
├── requirements.txt      # Python dependencies
└── .python-version       # Python version specification
```

## Running the Pipeline

### Full Data Pipeline
```bash
# From project root
python src/ingestion/tea_pipeline_v2.py
```

This will:
1. Parse all PDF reports in `data/Raw/`
2. Extract structured tables into `data/Interim/`
3. Generate weather features automatically
4. Create interim CSVs with deduplication

### Build Processed Datasets
```bash
# Master table with all features
python src/processing/build_master_table.py

# Reduced feature set
python src/processing/build_reduced_master.py

# Preprocessed table for modeling
python src/processing/preprocess_tea.py
```

### Run EDA Notebook
```bash
jupyter notebook notebooks/tea_eda.ipynb
```

## Dependencies Overview

| Package | Purpose |
|---------|---------|
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computing |
| **scikit-learn** | Machine learning utilities |
| **pdfplumber** | PDF parsing and text extraction |
| **requests** | HTTP requests (weather API) |
| **matplotlib** | Data visualization |
| **seaborn** | Statistical visualization |
| **jupyter** | Interactive notebook environment |
| **openpyxl** | Excel file handling |
| **python-dotenv** | Environment variable management |

## Troubleshooting

### Virtual Environment Not Activating
- **Windows**: Use `.venv\Scripts\activate`
- **macOS/Linux**: Use `source .venv/bin/activate`
- Verify activation with `which python` (should show `.venv` path)

### Import Errors
1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install --force-reinstall -r requirements.txt`
3. Check Python version: `python --version`

### PDF Parsing Issues
- Ensure PDFs are in `data/Raw/` with naming convention: `Sale of DD & DD Month YYYY.pdf`
- Check PDF file integrity
- Review pdfplumber version compatibility

## Development Workflow

1. **Always activate the virtual environment** before working
2. **Make code changes** in the `src/` directory
3. **Test changes** by running individual pipeline steps or notebooks
4. **Commit changes** to git with descriptive messages

## Additional Resources

- See `PROJECT_SUMMARY.md` for detailed technical documentation
- Refer to notebook comments for data processing rationale
- Check code docstrings for function-level details
