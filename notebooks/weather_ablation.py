"""
Weather-feature ablation for catalogue-specific tea-price forecasting.

Compares:
1. Full model: all existing features
2. No lagged weather: removes the nine Open-Meteo lagged weather variables
3. No explicit weather: additionally removes current temperature and weather severity

The script also saves fold-level RMSE values for uncertainty reporting.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DATA_PATH = ROOT / "data" / "colombo_tea_auction_dataset.csv"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BASE = "price_mid_lkr"
TARGET = "price_next_week"
GROUP = ["catalogue", "grade", "tier"]

CATALOGUE_ORDER = ["high_grown", "low_grown", "off_grade", "dust"]

PRETTY = {
    "high_grown": "High Grown",
    "low_grown": "Low Grown",
    "off_grade": "Off-Grade",
    "dust": "Dust",
}

# Best tree-based model from the baseline experiment.
BEST_MODEL_BY_CATALOGUE = {
    "high_grown": "LightGBM",
    "low_grown": "LightGBM",
    "off_grade": "LightGBM",
    "dust": "XGBoost",
}

LAGGED_WEATHER = [
    "precipitation_sum_total_lag1",
    "sunshine_duration_total_lag1",
    "temperature_2m_mean_mean_lag1",
    "precipitation_sum_total_lag2",
    "sunshine_duration_total_lag2",
    "temperature_2m_mean_mean_lag2",
    "precipitation_sum_total_lag3",
    "sunshine_duration_total_lag3",
    "temperature_2m_mean_mean_lag3",
]

EXTRA_WEATHER = [
    "temperature_2m_mean_mean",
    "avg_weather_severity",
]

GRADE_RANK = {
    "bop": 4, "bopf": 4, "bop1": 4,
    "fbop": 3, "fbop1": 3, "fbopf": 3,
    "fbopf1": 3, "fbopf_tippy": 3,
    "op": 2, "op1": 2, "opa": 2,
    "pekoe": 1, "pek1": 1, "pekoe_fbop": 1,
}

TIER_RANK = {
    "select_best": 4,
    "best": 3,
    "below_best": 2,
    "others": 1,
}

ELEV_RANK = {
    "high_grown": 3,
    "medium_grown": 2,
    "low_grown": 1,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def parse_auction_date(raw):
    """Parse auction close date from sale_date_raw."""
    if pd.isna(raw):
        return pd.NaT

    value = str(raw)

    match = re.search(
        r"(\d{1,2})\w*\s*/\s*(\d{1,2})\w*\s+([A-Za-z]+)\s+(\d{4})",
        value,
    )
    if match:
        day, month, year = match.group(2), match.group(3), match.group(4)
        return pd.to_datetime(
            f"{day} {month} {year}",
            format="%d %B %Y",
            errors="coerce",
        )

    match = re.search(
        r"(\d{1,2})\w*\s+([A-Za-z]+)\s+(\d{4})",
        value,
    )
    if match:
        day, month, year = match.group(1), match.group(2), match.group(3)
        return pd.to_datetime(
            f"{day} {month} {year}",
            format="%d %B %Y",
            errors="coerce",
        )

    return pd.NaT


def load_and_prepare():
    """Load data, enforce chronological order, and create the next-week target."""
    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_BASE, "catalogue"]).copy()

    df["auction_date"] = df["sale_date_raw"].apply(parse_auction_date)

    invalid_dates = int(df["auction_date"].isna().sum())
    if invalid_dates:
        raise ValueError(
            f"{invalid_dates} rows have invalid sale_date_raw values."
        )

    df = df.sort_values(
        ["auction_date", "catalogue", "grade", "tier"]
    ).reset_index(drop=True)

    df[TARGET] = df.groupby(GROUP)[TARGET_BASE].shift(-1)
    df = df.dropna(subset=[TARGET]).copy().reset_index(drop=True)

    df["time_idx"] = df.groupby("auction_date").ngroup()

    return df


def engineer_features(df):
    """Reproduce the feature engineering used by the baseline experiment."""
    df = df.copy()

    df["grade_rank"] = (
        df["grade"].astype(str).str.lower().map(GRADE_RANK).fillna(0)
    )
    df["tier_rank"] = (
        df["tier"].astype(str).str.lower().map(TIER_RANK).fillna(0)
    )
    df["prestige_rank"] = (
        df["elevation"].astype(str).str.lower().map(ELEV_RANK).fillna(0)
    )

    for lag in [1, 2, 3]:
        df[f"price_lag{lag}"] = df.groupby(GROUP)[TARGET_BASE].shift(lag)

    df["price_roll_mean3"] = df.groupby(GROUP)[TARGET_BASE].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )

    df["price_roll_std3"] = df.groupby(GROUP)[TARGET_BASE].transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).std().fillna(0)
    )

    categorical_columns = [
        "catalogue",
        "grade",
        "tier",
        "region",
        "segment",
        "elevation",
        "text_crop_change",
        "category",
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[f"{column}_enc"] = LabelEncoder().fit_transform(
                df[column].fillna("unknown").astype(str)
            )

    drop_columns = {
        TARGET_BASE,
        TARGET,
        "sale_id",
        "sale_date_raw",
        "auction_date",
        "catalogue",
        "grade",
        "tier",
        "region",
        "segment",
        "elevation",
        "text_crop_change",
        "category",
        "text_keywords",
    }

    leakage_substrings = ["price_next_week", "t_plus", TARGET]

    feature_columns = []

    for column in df.select_dtypes(include=[np.number]).columns:
        if column in drop_columns:
            continue

        if any(text in column for text in leakage_substrings):
            continue

        feature_columns.append(column)

    return df, feature_columns


def make_model(model_name):
    """Return the selected tree-based model."""
    if model_name == "LightGBM":
        return LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=-1,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def calculate_metrics(y_true, y_pred):
    """Calculate raw-scale forecasting metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(
        np.mean(
            np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))
        ) * 100
    )
    r2 = float(r2_score(y_true, y_pred))

    return rmse, mae, mape, r2


def evaluate_feature_set(block, feature_columns, catalogue, model_name, feature_set):
    """Evaluate one model and one feature set using chronological OOF prediction."""
    block = block.reset_index(drop=True)

    X = block[feature_columns].copy()
    y = block[TARGET].to_numpy(dtype=float)

    model = make_model(model_name)
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    oof_predictions = np.full(len(block), np.nan)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", clone(model)),
        ])

        pipeline.fit(X.iloc[train_idx], y[train_idx])
        predictions = pipeline.predict(X.iloc[test_idx])

        oof_predictions[test_idx] = predictions

        rmse, mae, mape, r2 = calculate_metrics(
            y[test_idx],
            predictions,
        )

        fold_rows.append({
            "catalogue": PRETTY[catalogue],
            "model": model_name,
            "feature_set": feature_set,
            "fold": fold,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "n_test": len(test_idx),
        })

    valid = ~np.isnan(oof_predictions)

    rmse, mae, mape, r2 = calculate_metrics(
        y[valid],
        oof_predictions[valid],
    )

    summary_row = {
        "catalogue": PRETTY[catalogue],
        "model": model_name,
        "feature_set": feature_set,
        "rmse_oof": rmse,
        "mae_oof": mae,
        "mape_oof": mape,
        "r2_oof": r2,
        "n_oof": int(valid.sum()),
        "feature_count": len(feature_columns),
    }

    return summary_row, fold_rows


def main():
    print("=" * 78)
    print(" WEATHER-FEATURE ABLATION -- catalogue-specific forecasting")
    print("=" * 78)

    df = load_and_prepare()
    df, all_features = engineer_features(df)

    lagged_weather_available = [
        column for column in LAGGED_WEATHER
        if column in all_features
    ]

    all_weather_available = [
        column for column in (LAGGED_WEATHER + EXTRA_WEATHER)
        if column in all_features
    ]

    feature_sets = {
        "Full features": all_features,
        "No lagged weather": [
            column for column in all_features
            if column not in lagged_weather_available
        ],
        "No explicit weather": [
            column for column in all_features
            if column not in all_weather_available
        ],
    }

    print(f"Dataset rows after t+1 target: {len(df):,}")
    print(f"Total ML features: {len(all_features)}")
    print(f"Lagged weather removed: {lagged_weather_available}")
    print(f"All explicit weather removed: {all_weather_available}")
    print()

    summary_rows = []
    all_fold_rows = []

    for catalogue in CATALOGUE_ORDER:
        block = df[df["catalogue"] == catalogue].copy()

        if len(block) < 50:
            print(f"Skipping {catalogue}: insufficient rows.")
            continue

        model_name = BEST_MODEL_BY_CATALOGUE[catalogue]

        print(
            f"[{PRETTY[catalogue]}] "
            f"model={model_name}, rows={len(block):,}"
        )

        for feature_set, columns in feature_sets.items():
            summary_row, fold_rows = evaluate_feature_set(
                block=block,
                feature_columns=columns,
                catalogue=catalogue,
                model_name=model_name,
                feature_set=feature_set,
            )

            summary_rows.append(summary_row)
            all_fold_rows.extend(fold_rows)

    summary_df = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(all_fold_rows)

    fold_stats = (
        folds_df.groupby(["catalogue", "model", "feature_set"])["rmse"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "fold_rmse_mean",
            "std": "fold_rmse_std",
        })
    )

    summary_df = summary_df.merge(
        fold_stats,
        on=["catalogue", "model", "feature_set"],
        how="left",
    )

    comparison_rows = []

    for catalogue in summary_df["catalogue"].unique():
        subset = summary_df[summary_df["catalogue"] == catalogue]

        full = subset[subset["feature_set"] == "Full features"].iloc[0]
        no_lagged = subset[
            subset["feature_set"] == "No lagged weather"
        ].iloc[0]
        no_weather = subset[
            subset["feature_set"] == "No explicit weather"
        ].iloc[0]

        comparison_rows.append({
            "Catalogue": catalogue,
            "Model": full["model"],
            "Full_RMSE": round(full["rmse_oof"], 2),
            "No_Lagged_Weather_RMSE": round(no_lagged["rmse_oof"], 2),
            "Lagged_Weather_RMSE_Gain": round(
                no_lagged["rmse_oof"] - full["rmse_oof"],
                2,
            ),
            "No_Explicit_Weather_RMSE": round(no_weather["rmse_oof"], 2),
            "Explicit_Weather_RMSE_Gain": round(
                no_weather["rmse_oof"] - full["rmse_oof"],
                2,
            ),
            "Full_Fold_RMSE_SD": round(full["fold_rmse_std"], 2),
            "No_Lagged_Fold_RMSE_SD": round(
                no_lagged["fold_rmse_std"],
                2,
            ),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "weather_ablation_oof_results.csv",
        index=False,
    )

    folds_df.to_csv(
        OUTPUT_DIR / "weather_ablation_fold_metrics.csv",
        index=False,
    )

    comparison_df.to_csv(
        OUTPUT_DIR / "weather_ablation_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 78)
    print(" WEATHER ABLATION SUMMARY")
    print("=" * 78)
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\nInterpretation:")
    print("Positive RMSE gain means the full model benefits from weather features.")
    print("Negative RMSE gain means weather features did not improve that model.")

    print("\nSaved:")
    print(OUTPUT_DIR / "weather_ablation_oof_results.csv")
    print(OUTPUT_DIR / "weather_ablation_fold_metrics.csv")
    print(OUTPUT_DIR / "weather_ablation_comparison.csv")


if __name__ == "__main__":
    main()