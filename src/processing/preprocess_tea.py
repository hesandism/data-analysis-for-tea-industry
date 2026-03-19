"""
Tea Auction Price Preprocessing Pipeline
Forbes & Walker Colombo Auction Data 2026
CS3121 Research Project

Run: python preprocess_tea.py
Input:  reduced_master_tea_prices.csv  (must be in same directory)
Output: tea_preprocessed.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────
df = pd.read_csv(r"D:\Projects\data-analysis-for-tea-industry\outputs\reduced_master_tea_prices.csv")
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# ══════════════════════════════════════════════════════════
# CRITICAL FIXES (C1–C3)
# ══════════════════════════════════════════════════════════

# ── C1: Price leakage ─────────────────────────────────────
# price_lo_lkr and price_hi_lkr are the two inputs that
# compute price_mid_lkr (the target). Correlation ~0.97–0.99.
# They are RETAINED in the file for reference, but must be
# EXCLUDED from any feature matrix used for modelling.
# See: LEAKAGE_COLS list at bottom of this file.
LEAKAGE_COLS = ["price_lo_lkr", "price_hi_lkr", "price_range_lkr"]
print("\n[C1] Leakage cols identified — exclude from model features:", LEAKAGE_COLS)

# ── C2: Sale 3 volume anomaly ─────────────────────────────
# The master_tea_prices.csv (full version) had a year-to-date
# figure (257M kg) in the weekly column for SALE_03_2026.
# This is already corrected in reduced_master_tea_prices.csv.
val_s3 = df.loc[df["sale_id"] == "SALE_03_2026", "total_sold_weekly_2026"].iloc[0]
assert val_s3 < 10_000_000, f"C2 anomaly still present! Value: {val_s3}"
print(f"\n[C2] SALE_03_2026 total_sold_weekly_2026 = {val_s3:,} kg  ✓ sane")

# ── C3: Harmonise elevation encoding ─────────────────────
# Table 06 used low/medium/high; Tables 04+05 used low_grown/high_grown
# Unified mapping applied below.
ELEVATION_MAP = {"high": "high_grown", "low": "low_grown", "medium": "medium_grown"}
before = df["elevation"].value_counts().to_dict()
df["elevation"] = df["elevation"].replace(ELEVATION_MAP)
after = df["elevation"].value_counts().to_dict()
print(f"\n[C3] Elevation normalised:")
print(f"     Before: {before}")
print(f"     After:  {after}")

# ══════════════════════════════════════════════════════════
# HIGH PRIORITY FIXES (H1–H4)
# ══════════════════════════════════════════════════════════

# ── H1: Target skewness ───────────────────────────────────
# Handled in Step 4 (target derivation). See price_mid_lkr_log.

# ── H2: Feature scaling ───────────────────────────────────
# StandardScaler should be applied at modelling time (not here)
# to avoid data leakage from validation folds.
# See: NUMERIC_FEATURE_COLS at bottom of this file.

# ── H3: Structural nulls in grade / tier ─────────────────
# grade and tier ONLY apply to 05_low_grown rows.
# DO NOT impute. Use table_source to subset first.
pct_grade_null = df["grade"].isna().mean() * 100
pct_tier_null  = df["tier"].isna().mean() * 100
print(f"\n[H3] grade null = {pct_grade_null:.1f}%  |  tier null = {pct_tier_null:.1f}%")
print("     Structural nulls — NOT imputed. Subset by table_source='05_low_grown' first.")

# ── H4: Near-constant sentiment_low_grown ─────────────────
n_unique = df["sentiment_low_grown"].nunique()
print(f"\n[H4] sentiment_low_grown has {n_unique} unique values → near-zero signal")
print("     Flagged. Drop during feature selection for regression models.")

# ══════════════════════════════════════════════════════════
# MEDIUM FIXES (M1–M4)
# ══════════════════════════════════════════════════════════

# ── M1: Weather lag null imputation ───────────────────────
# Lag nulls exist because early sales have no prior sales to lag from.
# Strategy: forward-fill within each table_source group sorted by sale_number,
#           then backward-fill any remaining (edge case at first sale).
lag_cols = [c for c in df.columns if "_lag1" in c or "_lag2" in c or "_lag3" in c]
null_before = df[lag_cols].isna().sum().sum()

df_s = df.sort_values("sale_number").copy()
df_s[lag_cols] = df_s.groupby("table_source")[lag_cols].ffill()
df_s[lag_cols] = df_s.groupby("table_source")[lag_cols].bfill()
df = df_s.sort_index()

null_after = df[lag_cols].isna().sum().sum()
print(f"\n[M1] Weather lag nulls: {null_before} → {null_after}")

# ── M2: sl_production_mkgs imputation ────────────────────
df["is_production_known"] = df["sl_production_mkgs"].notna().astype(int)
prod_mean = df.loc[df["is_production_known"] == 1, "sl_production_mkgs"].mean()
df["sl_production_mkgs"] = df["sl_production_mkgs"].fillna(prod_mean)
print(f"\n[M2] sl_production_mkgs imputed with mean = {prod_mean:.2f} mkgs")
print(f"     is_production_known flag column added")

# ── M3: FX — derive USD price ────────────────────────────
# FX barely varies across 10 sales (CV < 0.3–1%), so it has
# near-zero explanatory power as a raw feature.
# Better use: currency conversion for comparability.
df["price_mid_usd"] = df["price_mid_lkr"] / df["fx_usd_2026"]
print(f"\n[M3] price_mid_usd derived (mean = {df['price_mid_usd'].mean():.2f} USD)")

# ── M4: Todate volumes are cumulative ────────────────────
# Use *_weekly_* cols for EDA. Keep *_todate_* as a separate
# momentum/cumulative signal if needed — not interchangeable.
print("\n[M4] Todate cols retained. Use weekly cols as primary volume features.")

# ══════════════════════════════════════════════════════════
# STEP 3: ENCODING
# ══════════════════════════════════════════════════════════

# sale_month — ordinal
MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
df["sale_month_enc"] = df["sale_month"].map(MONTH_ORDER)

# tier — ordinal (0 = not applicable / null)
TIER_ORDER = {"others": 1, "below_best": 2, "best": 3, "select_best": 4}
df["tier_enc"] = df["tier"].map(TIER_ORDER).fillna(0).astype(int)

# elevation — ordinal
ELEV_ORDER = {"low_grown": 1, "medium_grown": 2, "high_grown": 3}
df["elevation_enc"] = df["elevation"].map(ELEV_ORDER)

# category_type — label encode
le_cat = LabelEncoder()
df["category_type_enc"] = le_cat.fit_transform(df["category_type"].fillna("unknown"))
CATEGORY_TYPE_MAP = dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_).tolist()))

# table_source — label encode
le_src = LabelEncoder()
df["table_source_enc"] = le_src.fit_transform(df["table_source"].fillna("unknown"))
TABLE_SOURCE_MAP = dict(zip(le_src.classes_, le_src.transform(le_src.classes_).tolist()))

print(f"\n[ENC] Encodings applied:")
print(f"      sale_month_enc  (ordinal Jan=1 … Dec=12)")
print(f"      tier_enc        (0=N/A, 1=others, 2=below_best, 3=best, 4=select_best)")
print(f"      elevation_enc   (1=low_grown, 2=medium_grown, 3=high_grown)")
print(f"      category_type_enc {CATEGORY_TYPE_MAP}")
print(f"      table_source_enc  {TABLE_SOURCE_MAP}")

# ══════════════════════════════════════════════════════════
# STEP 4: TARGET DERIVATION
# ══════════════════════════════════════════════════════════

# Log transform to address right skew (skew 2.49 → ~0.55)
df["price_mid_lkr_log"] = np.log1p(df["price_mid_lkr"])
skew_orig = df["price_mid_lkr"].skew()
skew_log  = df["price_mid_lkr_log"].skew()
print(f"\n[TARGET] price_mid_lkr_log derived")
print(f"         Skew before: {skew_orig:.3f}  →  after log1p: {skew_log:.3f}")

# Flag rows without a price target
df["has_price_target"] = df["price_mid_lkr"].notna().astype(int)
missing_target = (df["has_price_target"] == 0).sum()
print(f"         has_price_target flag: {missing_target} rows lack target (exclude from supervised ML)")

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
new_cols = [
    "is_production_known", "price_mid_usd",
    "sale_month_enc", "tier_enc", "elevation_enc",
    "category_type_enc", "table_source_enc",
    "price_mid_lkr_log", "has_price_target",
]
print(f"\n{'='*55}")
print(f"Final shape: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"New cols added ({len(new_cols)}): {new_cols}")
print(f"Leakage cols to exclude from models: {LEAKAGE_COLS}")
print(f"{'='*55}")

# ──────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────
output_path = "tea_preprocessed.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")

# ══════════════════════════════════════════════════════════
# REFERENCE: Feature column lists for modelling
# ══════════════════════════════════════════════════════════

# Columns to ALWAYS exclude from model feature matrix
EXCLUDE_FROM_FEATURES = [
    "sale_id", "sale_date_raw",            # identifiers
    "sale_month",                           # replaced by sale_month_enc
    "elevation",                            # replaced by elevation_enc
    "category_type",                        # replaced by category_type_enc
    "table_source",                         # replaced by table_source_enc
    "tier",                                 # replaced by tier_enc
    "grade",                                # free text / high null
    "price_lo_lkr", "price_hi_lkr",        # C1 leakage
    "price_range_lkr",                      # derived from lo/hi
    "price_mid_lkr",                        # raw target (use _log version)
    "price_mid_lkr_log",                    # the modelling target itself
    "has_price_target",                     # masking flag
]
