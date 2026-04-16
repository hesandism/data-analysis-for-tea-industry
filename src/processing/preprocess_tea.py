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
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate column that exists in the DataFrame."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def add_market_structure_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Add market-structure features using only past observations:
      - supply_pressure_index
      - demand_intensity_ratio
      - market_tightness_indicator

    Historical baselines are built from shifted expanding means so the current
    sale never contributes to its own denominator.
    """
    df_local = frame.copy()

    if "sale_id" not in df_local.columns:
        raise ValueError("sale_id is required to build market-structure features")

    # Prefer sale-level total offered quantity when available.
    supply_col = _first_existing(
        df_local.columns.tolist(),
        ["total__qty_mkgs", "qty_mkgs", "total_kgs", "total_lots"],
    )
    demand_cols = [c for c in df_local.columns if c.endswith("__demand_score")]
    total_demand_col = _first_existing(
        df_local.columns.tolist(),
        ["total__demand_score", "demand_score", "market_demand_score"],
    )

    if supply_col is None:
        raise ValueError("Could not detect an offered-volume column")
    if not demand_cols and total_demand_col is None:
        raise ValueError("Could not detect any demand-score columns")

    # Build historical baselines within the main segment axis when available.
    if "category_type" in df_local.columns:
        group_cols = ["category_type"]
    elif "table_source" in df_local.columns:
        group_cols = ["table_source"]
    else:
        group_cols = []

    # Stable sale ordering for time-aware historical averages.
    if "sale_date_raw" in df_local.columns:
        # Parse values like "01ST/02ND September 2025" using the first sale day.
        sale_date_parts = (
            df_local["sale_date_raw"]
            .astype("string")
            .str.extract(r"^(\d{1,2})(?:ST|ND|RD|TH)/\d{1,2}(?:ST|ND|RD|TH)\s+([A-Za-z]+)\s+(\d{4})$")
        )
        sale_date_str = (
            sale_date_parts[0].str.zfill(2)
            + " "
            + sale_date_parts[1]
            + " "
            + sale_date_parts[2]
        )
        df_local["_sale_date_parsed"] = pd.to_datetime(
            sale_date_str, format="%d %B %Y", errors="coerce"
        )
    else:
        df_local["_sale_date_parsed"] = pd.NaT
    if "sale_number" in df_local.columns:
        df_local["_sale_number_numeric"] = pd.to_numeric(df_local["sale_number"], errors="coerce")
    else:
        df_local["_sale_number_numeric"] = np.nan

    sort_identity = ["sale_id"]
    if "report_id" in df_local.columns:
        sort_identity.append("report_id")
    df_local = df_local.sort_values(
        ["_sale_date_parsed", "_sale_number_numeric"] + sort_identity, kind="mergesort"
    ).reset_index(drop=True)
    df_local["_sale_order"] = np.arange(len(df_local))

    key_cols = ["sale_id"]
    if "report_id" in df_local.columns:
        key_cols.append("report_id")
    key_cols += group_cols
    keep_cols = key_cols + ["_sale_order", supply_col]
    if total_demand_col is not None:
        keep_cols.append(total_demand_col)
    else:
        keep_cols.extend(demand_cols)

    sale_level = df_local[keep_cols].drop_duplicates(subset=key_cols).copy()
    sort_cols = group_cols + ["_sale_order"] if group_cols else ["_sale_order"]
    sale_level = sale_level.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    sale_level["_current_supply_proxy"] = pd.to_numeric(sale_level[supply_col], errors="coerce")
    if total_demand_col is not None:
        sale_level["_current_demand_proxy"] = pd.to_numeric(
            sale_level[total_demand_col], errors="coerce"
        )
    else:
        sale_level["_current_demand_proxy"] = (
            sale_level[demand_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        )

    if group_cols:
        sale_level["_historical_supply_avg"] = sale_level.groupby(group_cols, sort=False)[
            "_current_supply_proxy"
        ].transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
        sale_level["_historical_demand_avg"] = sale_level.groupby(group_cols, sort=False)[
            "_current_demand_proxy"
        ].transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
    else:
        sale_level["_historical_supply_avg"] = (
            sale_level["_current_supply_proxy"].expanding(min_periods=1).mean().shift(1)
        )
        sale_level["_historical_demand_avg"] = (
            sale_level["_current_demand_proxy"].expanding(min_periods=1).mean().shift(1)
        )

    sale_level["supply_pressure_index"] = np.divide(
        sale_level["_current_supply_proxy"],
        sale_level["_historical_supply_avg"].replace(0, np.nan),
    )
    sale_level["demand_intensity_ratio"] = np.divide(
        sale_level["_current_demand_proxy"],
        sale_level["_historical_demand_avg"].replace(0, np.nan),
    )
    sale_level["market_tightness_indicator"] = np.divide(
        sale_level["demand_intensity_ratio"],
        sale_level["supply_pressure_index"].replace(0, np.nan),
    )

    feature_cols = key_cols + [
        "supply_pressure_index",
        "demand_intensity_ratio",
        "market_tightness_indicator",
    ]
    df_local = df_local.merge(sale_level[feature_cols], on=key_cols, how="left")

    # Remove internal helper columns used for construction.
    df_local = df_local.drop(columns=["_sale_date_parsed", "_sale_number_numeric", "_sale_order"])

    mapping = {
        "supply_column": supply_col,
        "demand_columns": total_demand_col if total_demand_col is not None else ", ".join(demand_cols),
        "history_group_columns": ", ".join(group_cols) if group_cols else "global",
    }
    return df_local, mapping


# ──────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
INPUT_FILE = _ROOT / "data" / "processed" / "reduced_master_tea_prices.csv"
OUTPUT_FILE = _ROOT / "data" / "processed" / "tea_preprocessed.csv"

df = pd.read_csv(INPUT_FILE)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

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
# Repair any inconsistent weekly totals using component weekly columns.
weekly_components = (
    df["private_sales_weekly_2026"]
    + df["public_auction_weekly_2026"]
    + df["forward_contracts_weekly_2026"]
)
weekly_mismatch = (
    df["total_sold_weekly_2026"].notna()
    & weekly_components.notna()
    & (df["total_sold_weekly_2026"] != weekly_components)
)
num_weekly_fixes = int(weekly_mismatch.sum())
if num_weekly_fixes > 0:
    df.loc[weekly_mismatch, "total_sold_weekly_2026"] = weekly_components[weekly_mismatch]
    print(f"\n[C2] Corrected weekly totals from components for {num_weekly_fixes} row(s)")

val_s3 = df.loc[df["sale_id"] == "SALE_03_2026", "total_sold_weekly_2026"].iloc[0]
assert val_s3 < 10_000_000, f"C2 anomaly still present after correction! Value: {val_s3}"
print(f"\n[C2] SALE_03_2026 total_sold_weekly_2026 = {val_s3:,} kg  ✓ sane")

# ── C3: Harmonise elevation encoding ─────────────────────
# Table 06 used low/medium/high; Tables 04+05 used low_grown/high_grown
# Unified mapping applied below.
ELEVATION_MAP = {"high": "high_grown", "low": "low_grown", "medium": "medium_grown"}
before = df["elevation"].value_counts().to_dict()
df["elevation"] = df["elevation"].replace(ELEVATION_MAP)
after = df["elevation"].value_counts().to_dict()
print("\n[C3] Elevation normalised:")
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
pct_tier_null = df["tier"].isna().mean() * 100
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
print("     is_production_known flag column added")

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

# ── M5: Supply-demand market structure indices ──────────
# Build three pooled-model features required by the paper hypothesis:
#   supply_pressure_index      = current offered volume / historical avg offered volume
#   demand_intensity_ratio     = current demand score / historical avg demand score
#   market_tightness_indicator = demand_intensity_ratio / supply_pressure_index
df, market_feature_mapping = add_market_structure_features(df)
print("\n[M5] Market-structure features added:")
print(f"     supply column: {market_feature_mapping['supply_column']}")
print(f"     demand column(s): {market_feature_mapping['demand_columns']}")
print(f"     history grouping: {market_feature_mapping['history_group_columns']}")
for col in ["supply_pressure_index", "demand_intensity_ratio", "market_tightness_indicator"]:
    non_null = int(df[col].notna().sum())
    print(f"     {col}: non-null={non_null:,}  mean={df[col].mean():.4f}")

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

print("\n[ENC] Encodings applied:")
print("      sale_month_enc  (ordinal Jan=1 ... Dec=12)")
print("      tier_enc        (0=N/A, 1=others, 2=below_best, 3=best, 4=select_best)")
print("      elevation_enc   (1=low_grown, 2=medium_grown, 3=high_grown)")
print(f"      category_type_enc {CATEGORY_TYPE_MAP}")
print(f"      table_source_enc  {TABLE_SOURCE_MAP}")

# ══════════════════════════════════════════════════════════
# STEP 4: TARGET DERIVATION
# ══════════════════════════════════════════════════════════

# Log transform to address right skew (skew 2.49 → ~0.55)
df["price_mid_lkr_log"] = np.log1p(df["price_mid_lkr"])
skew_orig = df["price_mid_lkr"].skew()
skew_log = df["price_mid_lkr_log"].skew()
print("\n[TARGET] price_mid_lkr_log derived")
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
    "supply_pressure_index", "demand_intensity_ratio", "market_tightness_indicator",
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
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")

# ══════════════════════════════════════════════════════════
# REFERENCE: Feature column lists for modelling
# ══════════════════════════════════════════════════════════

# Columns to ALWAYS exclude from model feature matrix
EXCLUDE_FROM_FEATURES = [
    "sale_id", "report_id", "sale_date_raw",  # identifiers
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
