"""
Naive baselines for the tea-price one-week-ahead forecasting paper.

Addresses the reviewer concern that high R2 values (e.g. Low Grown R2~0.967)
reflect price persistence (autocorrelation) rather than the weather features /
gradient-boosting models adding real value.

Two trivial baselines are computed on the *identical* CV folds used by the
real models so the comparison is fair:

  1. NAIVE PERSISTENCE -- predict price_next_week = current week's price_mid_lkr
     (last observed price carried forward). No training.

  2. AR(1) -- one-lag autoregression using ONLY the previous price_mid_lkr as the
     single predictor (no weather, no structural features). Fit per fold on the
     training portion, predict on the test portion, within each product stream.

Pipeline conventions mirror notebooks/unified_pooled.ipynb exactly:
  * target price_next_week = price_mid_lkr shifted forward ONE auction week
    within each (catalogue, grade, tier) product stream;
  * 5-fold sklearn TimeSeriesSplit, out-of-fold (OOF) predictions;
  * metrics RMSE / MAE / MAPE(%) / R2 on the raw LKR scale;
  * per-catalogue AND unified/pooled.

ONE DISCIPLINE FIX vs. the original notebooks: chronological ordering is by
PARSED DATE (sale_date_raw), not by sale_number. Sale numbers are not unique
across years and -- in this dataset -- 2026 sale #14 is dated 07/08 March 2026,
which numerically sorts last but chronologically belongs between sales #9 and
#10. Sorting by sale_number would leak future-dated rows into earlier folds.
The ML models are re-run here on these same date-sorted folds so every row of
the augmented table is directly comparable.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
SEED = 42
N_SPLITS = 5

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_PATH = ROOT / "data" / "colombo_tea_auction_dataset.csv"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BASE = "price_mid_lkr"
TARGET = "price_next_week"           # one-week-ahead target
SEGMENT_COL = "catalogue"
GROUP = ["catalogue", "grade", "tier"]  # product stream

CATALOGUE_ORDER = ["high_grown", "low_grown", "off_grade", "dust"]
PRETTY = {
    "high_grown": "High Grown",
    "low_grown": "Low Grown",
    "off_grade": "Off-Grade",
    "dust": "Dust",
    "__pooled__": "Pooled (Unified)",
}


# ---------------------------------------------------------------------------
# 1. Load + chronological order by PARSED DATE (not sale_number)
# ---------------------------------------------------------------------------
def parse_auction_date(raw):
    """Parse 'DDTH/DDTH Month YYYY' -> auction end day as a datetime.

    Uses the SECOND day of the two-day auction window (the close). Falls back to
    a single-day pattern. Returns NaT only if nothing matches.
    """
    if pd.isna(raw):
        return pd.NaT
    s = str(raw)
    m = re.search(r"(\d{1,2})\w*\s*/\s*(\d{1,2})\w*\s+([A-Za-z]+)\s+(\d{4})", s)
    if m:
        day, month, year = m.group(2), m.group(3), m.group(4)
        return pd.to_datetime(f"{day} {month} {year}", format="%d %B %Y", errors="coerce")
    m2 = re.search(r"(\d{1,2})\w*\s+([A-Za-z]+)\s+(\d{4})", s)
    if m2:
        return pd.to_datetime(
            f"{m2.group(1)} {m2.group(2)} {m2.group(3)}", format="%d %B %Y", errors="coerce"
        )
    return pd.NaT


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_BASE, SEGMENT_COL]).copy()

    # Parse the real auction date and use it for chronological discipline.
    df["auction_date"] = df["sale_date_raw"].apply(parse_auction_date)
    n_bad = int(df["auction_date"].isna().sum())
    if n_bad:
        raise ValueError(f"{n_bad} rows have an unparseable sale_date_raw; fix before modelling.")

    # Stable chronological sort: by parsed date, then product stream for determinism.
    df = df.sort_values(["auction_date", "catalogue", "grade", "tier"]).reset_index(drop=True)

    # One-week-ahead target within each product stream.
    df[TARGET] = df.groupby(GROUP)[TARGET_BASE].shift(-1)
    df = df.dropna(subset=[TARGET]).copy().reset_index(drop=True)

    # Absolute time index from the date ordering (ties = same auction date).
    df["time_idx"] = df.groupby("auction_date").ngroup()
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering for the ML models (mirrors unified_pooled.ipynb)
# ---------------------------------------------------------------------------
GRADE_RANK = {
    "bop": 4, "bopf": 4, "bop1": 4,
    "fbop": 3, "fbop1": 3, "fbopf": 3, "fbopf1": 3, "fbopf_tippy": 3,
    "op": 2, "op1": 2, "opa": 2,
    "pekoe": 1, "pek1": 1, "pekoe_fbop": 1,
}
TIER_RANK = {"select_best": 4, "best": 3, "below_best": 2, "others": 1}
ELEV_RANK = {"high_grown": 3, "medium_grown": 2, "low_grown": 1, "high": 3, "medium": 2, "low": 1}


def engineer_features(df):
    df = df.copy()
    df["grade_rank"] = df["grade"].astype(str).str.lower().map(GRADE_RANK).fillna(0)
    df["tier_rank"] = df["tier"].astype(str).str.lower().map(TIER_RANK).fillna(0)
    df["prestige_rank"] = df["elevation"].astype(str).str.lower().map(ELEV_RANK).fillna(0)

    for lag in [1, 2, 3]:
        df[f"price_lag{lag}"] = df.groupby(GROUP)[TARGET_BASE].shift(lag)
    df["price_roll_mean3"] = df.groupby(GROUP)[TARGET_BASE].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["price_roll_std3"] = df.groupby(GROUP)[TARGET_BASE].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std().fillna(0)
    )

    for col in ["catalogue", "grade", "tier", "region", "segment",
                "elevation", "text_crop_change", "category"]:
        if col in df.columns:
            df[f"{col}_enc"] = LabelEncoder().fit_transform(df[col].fillna("unknown").astype(str))

    # Leakage guard: never let any column containing the target substring in.
    leak_substrings = ["price_next_week", "t_plus", TARGET]
    drop = {
        TARGET_BASE, TARGET, "sale_id", "sale_date_raw", "auction_date",
        "catalogue", "grade", "tier", "region", "segment",
        "elevation", "text_crop_change", "category", "text_keywords",
    }
    feature_cols = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if c in drop:
            continue
        if any(sub in c for sub in leak_substrings):
            continue
        feature_cols.append(c)
    return df, feature_cols


MODELS = {
    "XGBoost": XGBRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbosity=0,
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=-1,
        random_state=SEED, n_jobs=-1, verbose=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=3, random_state=SEED,
    ),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
}


# ---------------------------------------------------------------------------
# 3. Metrics (raw LKR scale, on OOF predictions)
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, mape, r2


# ---------------------------------------------------------------------------
# 4. OOF evaluation on a single (sub)frame using shared TimeSeriesSplit folds
# ---------------------------------------------------------------------------
def evaluate_block(block, feature_cols, label):
    """Run persistence, AR(1) and the ML zoo on one already-time-sorted block.

    All methods use the *same* TimeSeriesSplit(n_splits=5) folds on the block,
    so every reported number is computed on identical OOF test rows.
    Returns a list of result dicts (one per method) for this `label`.
    """
    block = block.reset_index(drop=True)
    n = len(block)
    y = block[TARGET].to_numpy(dtype=float)

    # Predictor columns the trivial baselines need.
    cur_price = block[TARGET_BASE].to_numpy(dtype=float)

    # Proper AR(1): use current week's observed price (p_t)
    # to forecast next week's price (p_t+1).
    ar1_input = cur_price.copy()

    X = block[feature_cols].copy()

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    oof_persist = np.full(n, np.nan)
    oof_ar1 = np.full(n, np.nan)
    oof_ml = {name: np.full(n, np.nan) for name in MODELS}
    test_mask = np.zeros(n, dtype=bool)  # rows that are ever in a test fold

    for tr_idx, te_idx in tscv.split(X):
        test_mask[te_idx] = True

        # 1. Naive persistence: prediction is just the current price. No fit.
        oof_persist[te_idx] = cur_price[te_idx]

        # 2. AR(1): fit y ~ a + b * current-week price on the TRAIN portion only,
        # predict on the TEST portion.
        ar = LinearRegression()
        tr_x, tr_y = ar1_input[tr_idx], y[tr_idx]
        ok = ~np.isnan(tr_x)

        if ok.sum() >= 2:
            ar.fit(tr_x[ok].reshape(-1, 1), tr_y[ok])

            te_x = ar1_input[te_idx]
            # Where the lag is missing (first obs of a stream), fall back to persistence.
            pred = np.where(
            np.isnan(te_x),
            cur_price[te_idx],
            ar.predict(np.nan_to_num(te_x, nan=0.0).reshape(-1, 1)),
            )
            oof_ar1[te_idx] = pred
        else:
            oof_ar1[te_idx] = cur_price[te_idx]

        # 3. ML zoo: median-impute then fit on the same fold.
        for name, model in MODELS.items():
            pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("model", clone(model))])
            pipe.fit(X.iloc[tr_idx], y[tr_idx])
            oof_ml[name][te_idx] = pipe.predict(X.iloc[te_idx])

    yt = y[test_mask]
    rows = []
    rmse, mae, mape, r2 = metrics(yt, oof_persist[test_mask])
    rows.append(dict(block=label, method="Naive Persistence", kind="baseline",
                     rmse=rmse, mae=mae, mape=mape, r2=r2, n_oof=int(test_mask.sum())))
    rmse, mae, mape, r2 = metrics(yt, oof_ar1[test_mask])
    rows.append(dict(block=label, method="AR(1)", kind="baseline",
                     rmse=rmse, mae=mae, mape=mape, r2=r2, n_oof=int(test_mask.sum())))
    for name in MODELS:
        rmse, mae, mape, r2 = metrics(yt, oof_ml[name][test_mask])
        rows.append(dict(block=label, method=name, kind="ml",
                         rmse=rmse, mae=mae, mape=mape, r2=r2, n_oof=int(test_mask.sum())))
    return rows


# ---------------------------------------------------------------------------
# 5. Run everything
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("  NAIVE BASELINES vs ML  --  one-week-ahead tea auction price forecast")
    print("=" * 78)

    df = load_and_prepare()
    df, feature_cols = engineer_features(df)
    print(f"Dataset: {DATA_PATH.name}  |  rows after t+1 target: {len(df):,}")
    print(f"ML features (target-substring columns excluded): {len(feature_cols)}")
    print(f"Catalogues: {', '.join(sorted(df[SEGMENT_COL].unique()))}")
    print(f"Chronological order: PARSED DATE (auction close day)")
    print()

    all_rows = []

    # Per-catalogue blocks (already globally date-sorted; subset preserves order).
    for cat in CATALOGUE_ORDER:
        block = df[df[SEGMENT_COL] == cat].copy()
        if len(block) < 50:
            print(f"  Skipping {cat}: only {len(block)} rows.")
            continue
        all_rows.extend(evaluate_block(block, feature_cols, cat))
        print(f"  [{PRETTY[cat]:<16}] evaluated  (n={len(block):,})")

    # Pooled / unified block (all catalogues, single global date order).
    all_rows.extend(evaluate_block(df.copy(), feature_cols, "__pooled__"))
    print(f"  [{PRETTY['__pooled__']:<16}] evaluated  (n={len(df):,})")
    print()

    res = pd.DataFrame(all_rows)
    res.to_csv(OUTPUT_DIR / "baseline_vs_ml_oof_results.csv", index=False)

    build_augmented_tables(res)
    sanity_summary(res)


# ---------------------------------------------------------------------------
# 6. Augmented results tables (CSV + printed) for IEEE LaTeX Tables II & III
# ---------------------------------------------------------------------------
def build_augmented_tables(res):
    blocks = CATALOGUE_ORDER + ["__pooled__"]
    method_order = ["Naive Persistence", "AR(1)", "XGBoost", "LightGBM",
                    "Gradient Boosting", "Random Forest"]

    augmented = []
    improvement_rows = []

    for blk in blocks:
        sub = res[res["block"] == blk]
        if sub.empty:
            continue
        persist_rmse = sub.loc[sub["method"] == "Naive Persistence", "rmse"].iloc[0]
        ml = sub[sub["kind"] == "ml"]
        best_ml = ml.loc[ml["rmse"].idxmin()]

        print("\n" + "=" * 78)
        print(f"  {PRETTY[blk].upper()}   (OOF n = {int(sub['n_oof'].iloc[0]):,})")
        print("=" * 78)
        print(f"  {'Method':<20}{'RMSE':>10}{'MAE':>10}{'MAPE%':>9}{'R2':>9}")
        print("  " + "-" * 56)
        for m in method_order:
            r = sub[sub["method"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            star = "  <- best ML" if (r["kind"] == "ml" and r["method"] == best_ml["method"]) else ""
            tag = "  (baseline)" if r["kind"] == "baseline" else ""
            print(f"  {m:<20}{r['rmse']:>10.2f}{r['mae']:>10.2f}{r['mape']:>9.2f}"
                  f"{r['r2']:>9.4f}{star}{tag}")
            augmented.append(dict(
                Catalogue=PRETTY[blk], Method=m, Kind=r["kind"],
                RMSE=round(r["rmse"], 2), MAE=round(r["mae"], 2),
                MAPE=round(r["mape"], 2), R2=round(r["r2"], 4),
                N_OOF=int(r["n_oof"]),
            ))

        impr = (persist_rmse - best_ml["rmse"]) / persist_rmse * 100
        improvement_rows.append(dict(
            Catalogue=PRETTY[blk],
            Best_ML=best_ml["method"],
            Best_ML_RMSE=round(best_ml["rmse"], 2),
            Persistence_RMSE=round(persist_rmse, 2),
            RMSE_improvement_pct=round(impr, 2),
            Best_ML_R2=round(best_ml["r2"], 4),
            Persistence_R2=round(sub.loc[sub["method"] == "Naive Persistence", "r2"].iloc[0], 4),
        ))
        print(f"  >> Best ML ({best_ml['method']}) RMSE {best_ml['rmse']:.2f} vs "
              f"persistence {persist_rmse:.2f}  =>  {impr:+.2f}% RMSE improvement")

    aug_df = pd.DataFrame(augmented)
    aug_df.to_csv(OUTPUT_DIR / "augmented_results_table.csv", index=False)

    impr_df = pd.DataFrame(improvement_rows)
    impr_df.to_csv(OUTPUT_DIR / "ml_vs_persistence_improvement.csv", index=False)

    print("\n" + "=" * 78)
    print("  ML IMPROVEMENT OVER NAIVE PERSISTENCE (RMSE)")
    print("=" * 78)
    print(f"  {'Catalogue':<18}{'Best ML':<18}{'ML RMSE':>9}{'Persist':>9}{'Improv%':>9}")
    print("  " + "-" * 61)
    for r in improvement_rows:
        print(f"  {r['Catalogue']:<18}{r['Best_ML']:<18}{r['Best_ML_RMSE']:>9.2f}"
              f"{r['Persistence_RMSE']:>9.2f}{r['RMSE_improvement_pct']:>8.2f}%")

    print(f"\n  Saved: {(OUTPUT_DIR / 'augmented_results_table.csv')}")
    print(f"  Saved: {(OUTPUT_DIR / 'ml_vs_persistence_improvement.csv')}")
    print(f"  Saved: {(OUTPUT_DIR / 'baseline_vs_ml_oof_results.csv')}")


# ---------------------------------------------------------------------------
# 7. Honest sanity summary
# ---------------------------------------------------------------------------
def sanity_summary(res, meaningful_pct=5.0):
    """Flag any block where the best ML model does NOT meaningfully beat
    persistence. 'Meaningful' = RMSE improvement >= `meaningful_pct`%."""
    print("\n" + "=" * 78)
    print("  SANITY CHECK  --  does ML meaningfully beat a trivial forecast?")
    print(f"  (threshold: best-ML RMSE must improve on persistence by >= {meaningful_pct:.0f}%)")
    print("=" * 78)

    blocks = CATALOGUE_ORDER + ["__pooled__"]
    flagged = []
    for blk in blocks:
        sub = res[res["block"] == blk]
        if sub.empty:
            continue
        persist = sub[sub["method"] == "Naive Persistence"].iloc[0]
        ml = sub[sub["kind"] == "ml"]
        best_ml = ml.loc[ml["rmse"].idxmin()]
        impr = (persist["rmse"] - best_ml["rmse"]) / persist["rmse"] * 100
        verdict = "OK -- ML adds value" if impr >= meaningful_pct else "FLAG -- persistence ~= ML"
        if impr < meaningful_pct:
            flagged.append(PRETTY[blk])
        print(f"  {PRETTY[blk]:<18} persistence R2={persist['r2']:.4f}  "
              f"best-ML R2={best_ml['r2']:.4f}  RMSE improv={impr:+.2f}%   [{verdict}]")

    print("  " + "-" * 70)
    if flagged:
        print(f"  >> HONEST FLAG: ML does NOT meaningfully beat persistence for: "
              f"{', '.join(flagged)}.")
        print("     For these segments, the high R2 is largely explained by price")
        print("     persistence (autocorrelation); the weather/ML machinery adds little.")
    else:
        print("  >> Every catalogue's best ML model beats persistence by the threshold;")
        print("     the weather/ML features add value beyond a trivial last-value forecast.")
    print("=" * 78)


if __name__ == "__main__":
    main()
