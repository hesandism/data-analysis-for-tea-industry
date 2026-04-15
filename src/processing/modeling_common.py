from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

SEED = 42
TARGET = "price_mid_lkr_log"
RANK_COL = "sale_rank"

DEFAULT_EXCLUDE_COLS = {
    "sale_id", "sale_date_raw", "sale_month", "table_source", "category_type", "grade", "tier", "category",
    "price_mid_lkr", "price_mid_lkr_log", "has_price_target", "price_lo_lkr", "price_hi_lkr", "price_range_lkr",
    "price_mid_usd",
}


LEAKAGE_PATTERNS = (
    "price_mid_lkr",
    "price_lo_lkr",
    "price_hi_lkr",
    "price_range_lkr",
    "price_mid_usd",
    "roll3_mean__price",
    "roll3_std__price",
)


def resolve_project_root(start_path=None):
    start = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    return start


def resolve_data_path(root, filename):
    candidates = [
        root / "data" / "processed" / filename,
        root / "data" / "Processed" / filename,
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Could not find {filename} in: {candidates}")
    return path


def resolve_preprocessed_path(root):
    candidates = [
        root / "data" / "processed" / "tea_preprocessed_v2.csv",
        root / "data" / "Processed" / "tea_preprocessed_v2.csv",
        root / "data" / "processed" / "tea_preprocessed.csv",
        root / "data" / "Processed" / "tea_preprocessed.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Could not find tea_preprocessed_v2.csv or tea_preprocessed.csv in: {candidates}")
    return path


def resolve_unified_summary_path(root):
    return resolve_data_path(root, "unified_model_cv_results.csv")


def load_preprocessed_df(root, keep_target_only=True, filename=None):
    data_path = resolve_preprocessed_path(root) if filename is None else resolve_data_path(root, filename)
    df = pd.read_csv(data_path)
    if keep_target_only and "has_price_target" in df.columns:
        df = df[df["has_price_target"] == 1].copy()
    return df, data_path


def is_leakage_feature(column_name):
    return any(pattern in column_name for pattern in LEAKAGE_PATTERNS)


def build_segment_filters(df):
    ts = df["table_source"].astype(str).str.strip().str.lower()
    ct = df["category_type"].astype(str).str.strip().str.lower()
    return {
        "High Grown": ts.isin(["04_high_grown", "04_high_grown_prices"]),
        "Low Grown": ts.isin(["05_low_grown", "05_low_grown_prices"]),
        "Off-Grade": ts.isin(["06_offgrade_dust", "06_offgrade_dust_prices"]) & (ct == "off_grade"),
        "Dust": ts.isin(["06_offgrade_dust", "06_offgrade_dust_prices"]) & (ct == "dust"),
    }


def get_segment_data(df, target=TARGET, rank_col=RANK_COL, exclude_cols=None):
    excludes = set(DEFAULT_EXCLUDE_COLS if exclude_cols is None else exclude_cols)
    filters = build_segment_filters(df)
    segment_data = {}
    for seg, mask in filters.items():
        sdf = df[mask].copy()
        numeric_cols = [c for c in sdf.columns if pd.api.types.is_numeric_dtype(sdf[c])]
        feature_cols = [c for c in numeric_cols if c not in excludes and c != target and not is_leakage_feature(c)]
        if rank_col in sdf.columns:
            sdf = sdf.sort_values(rank_col).reset_index(drop=True)
        segment_data[seg] = (sdf, feature_cols)
    return segment_data


def build_model_registry(seed=SEED):
    return {
        "Ridge": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=seed))]),
        "Lasso": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Lasso(alpha=0.001, random_state=seed, max_iter=10000))]),
        "ElasticNet": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=seed, max_iter=10000))]),
        "Random Forest": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(random_state=seed))]),
        "SVR (RBF)": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale"))]),
        "XGBoost": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0))]),
        "LightGBM": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=seed, n_jobs=-1, verbosity=-1))]),
    }




def get_param_grids():
    return {
        "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "Lasso": {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1]},
        "ElasticNet": {
            "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
        "Random Forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
        "SVR (RBF)": {
            "model__C": [1.0, 10.0, 50.0],
            "model__epsilon": [0.01, 0.05, 0.1],
            "model__gamma": ["scale", "auto"],
        },
        "XGBoost": {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
            "model__subsample": [0.8, 1.0],
        },
        "LightGBM": {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__num_leaves": [31, 63],
            "model__subsample": [0.8, 1.0],
        },
    }

def compute_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    r2 = float(r2_score(y_true_log, y_pred_log))
    return rmse, mae, mape, r2


def run_timeseries_cv(sdf, feature_cols, model_name, model_obj, target=TARGET, k=5):
    X = sdf[feature_cols].copy()
    y = sdf[target].copy()
    tscv = TimeSeriesSplit(n_splits=k)
    fold_rows = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        model_obj.fit(X_tr, y_tr)
        pred_log = model_obj.predict(X_te)
        rmse, mae, mape, r2 = compute_metrics(y_te.values, pred_log)

        fold_rows.append(
        {
            "Model": model_name,
            "Fold": fold,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
        }
)


    return pd.DataFrame(fold_rows)


def evaluate_estimator_timeseries(sdf, feature_cols, estimator, target=TARGET, k=5):
    X = sdf[feature_cols].copy()
    y = sdf[target].copy()
    tscv = TimeSeriesSplit(n_splits=k)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        estimator.fit(X_tr, y_tr)
        pred_log = estimator.predict(X_te)
        rmse, mae, mape, r2 = compute_metrics(y_te.values, pred_log)

        rows.append(
            {
                "Fold": fold,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape,
                "R2": r2,
                "n_train": len(tr_idx),
                "n_test": len(te_idx),
            }
)
    return pd.DataFrame(rows)
