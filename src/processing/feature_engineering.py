"""
Interaction & Derived Feature Construction

This script builds three groups of engineered features on top of tea_preprocessed.csv:
  1. Interaction Terms   — weather × grade, rainfall × elevation, sentiment × demand
  2. Rolling Statistics  — 3-sale moving average for price, volume, and weather (per segment)
  3. Polynomial Features — degree-2 terms for the top-5 weather predictors (by correlation)

Output: tea_preprocessed_v2.csv  (+new feature columns appended)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = _ROOT / "data" / "processed" / "tea_preprocessed.csv"
OUTPUT_PATH = _ROOT / "data" / "processed" / "tea_preprocessed_v2.csv"

print("Loading data ...")
df = pd.read_csv(INPUT_PATH)
print(f"  Loaded  : {df.shape[0]} rows × {df.shape[1]} columns")

# Sort so rolling windows respect chronological order
df = df.sort_values(["sale_number", "category_type"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: segment-relevant weather column selector
# ─────────────────────────────────────────────────────────────────────────────
# Each tea segment is grown in specific regions.
# Map segment → primary and secondary weather region prefixes.
SEGMENT_WEATHER_MAP = {
    "high_grown": ["western_high", "nuwara_eliya"],
    "low_grown": ["low_grown"],
    "off_grade": ["low_grown", "western_high"],  # mixed origin
    "dust": ["low_grown", "western_high"],  # mixed origin
}


def segment_weather_col(df_cols, segment, metric, suffix=""):
    """
    Return the first existing column matching a segment's primary weather region
    for the given metric (e.g. 'precipitation_sum_total') and optional suffix.
    """
    for prefix in SEGMENT_WEATHER_MAP.get(segment, []):
        col = f"{prefix}__{metric}{suffix}"
        if col in df_cols:
            return col
    # Fallback: all-regions average precipitation composite
    fallback = f"all_regions__avg_precipitation{suffix}"
    return fallback if fallback in df_cols else None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — INTERACTION TERMS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/3] Building interaction terms ...")

new_interaction_cols = {}

for segment, group in df.groupby("category_type"):

    # ── 1a. Weather × Grade  ─────────────────────────────────────────────────
    # Use segment-relevant precipitation and sunshine
    precip_col = segment_weather_col(df.columns, segment, "precipitation_sum_total")
    sunshine_col = segment_weather_col(df.columns, segment, "sunshine_duration_total")
    temp_col = segment_weather_col(df.columns, segment, "temperature_2m_mean_mean")

    mask = df["category_type"] == segment

    if precip_col:
        col_name = f"interact__{segment}__precip_x_grade"
        new_interaction_cols.setdefault(col_name, pd.Series(np.nan, index=df.index))
        new_interaction_cols[col_name][mask] = (
            df.loc[mask, precip_col] * df.loc[mask, "grade_enc"]
            if "grade_enc" in df.columns
            else df.loc[mask, precip_col] * df.loc[mask, "tier_enc"]
        )

    if sunshine_col:
        col_name = f"interact__{segment}__sunshine_x_grade"
        new_interaction_cols.setdefault(col_name, pd.Series(np.nan, index=df.index))
        new_interaction_cols[col_name][mask] = (
            df.loc[mask, sunshine_col] * df.loc[mask, "tier_enc"]
        )

    if temp_col:
        col_name = f"interact__{segment}__temp_x_grade"
        new_interaction_cols.setdefault(col_name, pd.Series(np.nan, index=df.index))
        new_interaction_cols[col_name][mask] = (
            df.loc[mask, temp_col] * df.loc[mask, "tier_enc"]
        )

    # ── 1b. Rainfall × Elevation  ────────────────────────────────────────────
    if precip_col:
        col_name = f"interact__{segment}__rainfall_x_elevation"
        new_interaction_cols.setdefault(col_name, pd.Series(np.nan, index=df.index))
        new_interaction_cols[col_name][mask] = (
            df.loc[mask, precip_col] * df.loc[mask, "elevation_enc"]
        )

    # ── 1c. Sentiment × Demand  ──────────────────────────────────────────────
    # Map each segment to the most relevant demand score
    demand_col_map = {
        "high_grown": "leafy__demand_score",  # leafy / premium flowery
        "low_grown": "ex_estate__demand_score",
        "off_grade": "off_grade__demand_score",
        "dust": "dust__demand_score",
    }
    demand_col = demand_col_map.get(segment)

    if demand_col and demand_col in df.columns:
        # sentiment_overall × demand
        col_name = f"interact__{segment}__sentiment_x_demand"
        new_interaction_cols.setdefault(col_name, pd.Series(np.nan, index=df.index))
        new_interaction_cols[col_name][mask] = (
            df.loc[mask, "sentiment_overall"] * df.loc[mask, demand_col]
        )

        # sentiment_ex_estate × demand (extra signal for non-estate teas)
        col_name2 = f"interact__{segment}__sentiment_ex_estate_x_demand"
        new_interaction_cols.setdefault(col_name2, pd.Series(np.nan, index=df.index))
        new_interaction_cols[col_name2][mask] = (
            df.loc[mask, "sentiment_ex_estate"] * df.loc[mask, demand_col]
        )

    # ── 1d. Weather Condition Score × Demand  ────────────────────────────────
    cond_col = segment_weather_col(df.columns, segment, "text_condition_score")
    if cond_col and demand_col and demand_col in df.columns:
        col_name = f"interact__{segment}__weather_cond_x_demand"
        new_interaction_cols.setdefault(col_name, pd.Series(np.nan, index=df.index))
        new_interaction_cols[col_name][mask] = (
            df.loc[mask, cond_col] * df.loc[mask, demand_col]
        )

# Attach interaction columns
interaction_df = pd.DataFrame(new_interaction_cols, index=df.index)
df = pd.concat([df, interaction_df], axis=1)
print(f"  Added   : {len(interaction_df.columns)} interaction features")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ROLLING STATISTICS (3-sale moving average, per segment)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/3] Building rolling statistics (window=3, per segment) ...")

# Columns to roll
PRICE_COLS = ["price_mid_lkr", "price_lo_lkr", "price_hi_lkr", "price_range_lkr"]
VOLUME_COLS = ["total__qty_mkgs", "total_sold_weekly_2026", "volume_yoy_change_pct"]
WEATHER_ROLL = [
    "all_regions__avg_precipitation",
    "low_grown__precipitation_sum_total",
    "western_high__precipitation_sum_total",
    "nuwara_eliya__precipitation_sum_total",
    "low_grown__sunshine_duration_total",
    "western_high__sunshine_duration_total",
    "low_grown__temperature_2m_mean_mean",
    "western_high__temperature_2m_mean_mean",
]

ROLL_COLS = (
    [c for c in PRICE_COLS if c in df.columns] +
    [c for c in VOLUME_COLS if c in df.columns] +
    [c for c in WEATHER_ROLL if c in df.columns]
)

WINDOW = 3
rolling_frames = []

for segment, grp in df.groupby("category_type"):
    # Sort within segment by sale_number to respect time order
    grp = grp.sort_values("sale_number")

    roll_data = {}
    for col in ROLL_COLS:
        # Compute rolling mean over the PREVIOUS 3 sales (min_periods=1 avoids NaN at start)
        roll_data[f"roll3_mean__{col}"] = (
            grp[col].shift(1).rolling(window=WINDOW, min_periods=1).mean().values
        )
        # Rolling std for price (captures volatility)
        if col in PRICE_COLS:
            roll_data[f"roll3_std__{col}"] = (
                grp[col].shift(1).rolling(window=WINDOW, min_periods=1).std().fillna(0).values
            )

    roll_df = pd.DataFrame(roll_data, index=grp.index)
    rolling_frames.append(roll_df)

rolling_df = pd.concat(rolling_frames).reindex(df.index)
df = pd.concat([df, rolling_df], axis=1)
print(f"  Added   : {len(rolling_df.columns)} rolling statistic features")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — POLYNOMIAL FEATURES (degree=2, top-5 weather predictors)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Building polynomial features (degree=2, top-5 weather predictors) ...")

# ── 3a. Use a fixed, non-target-based weather feature set to avoid selection leakage
fixed_weather_candidates = [
    "all_regions__avg_precipitation",
    "western_high__precipitation_sum_total",
    "nuwara_eliya__precipitation_sum_total",
    "uva_udapussellawa__precipitation_sum_total",
    "low_grown__precipitation_sum_total",
    "western_high__sunshine_duration_total",
    "nuwara_eliya__sunshine_duration_total",
    "uva_udapussellawa__sunshine_duration_total",
    "low_grown__sunshine_duration_total",
    "western_high__temperature_2m_mean_mean",
    "nuwara_eliya__temperature_2m_mean_mean",
    "uva_udapussellawa__temperature_2m_mean_mean",
    "low_grown__temperature_2m_mean_mean",
    "western_high__relative_humidity_2m_max_mean",
    "low_grown__relative_humidity_2m_max_mean",
]

top5_weather = [c for c in fixed_weather_candidates if c in df.columns][:5]
print("  Fixed weather predictors used for polynomial features:")
for rank, col in enumerate(top5_weather, 1):
    print(f"    {rank}. {col:60s}")

# ── 3b. Generate degree-2 polynomial features for those 5 columns
poly_input = df[top5_weather].fillna(df[top5_weather].median())

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
poly_array = poly.fit_transform(poly_input)
poly_feature_names = poly.get_feature_names_out(top5_weather)

# Remove the original linear terms (already in df) — keep only new squared/cross terms
new_poly_names = [n for n in poly_feature_names if n not in top5_weather]
poly_df_all = pd.DataFrame(poly_array, columns=poly_feature_names, index=df.index)
poly_df_new = poly_df_all[new_poly_names].copy()

# Rename for clarity
rename_map = {
    col: f"poly2__{col.replace(' ', '__x__')}"
    for col in poly_df_new.columns
}
poly_df_new = poly_df_new.rename(columns=rename_map)

df = pd.concat([df, poly_df_new], axis=1)
print(f"  Added   : {len(poly_df_new.columns)} polynomial features "
    f"({len([n for n in poly_feature_names if '^2' in n or ' ' in n])} sq + cross terms)")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY & SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)

original_cols = 182
added_interaction = len(interaction_df.columns)
added_rolling = len(rolling_df.columns)
added_poly = len(poly_df_new.columns)
total_new = added_interaction + added_rolling + added_poly

print(f"  Original features          : {original_cols}")
print(f"  + Interaction features     : {added_interaction}")
print(f"  + Rolling stat features    : {added_rolling}")
print(f"  + Polynomial features      : {added_poly}")
print("  ─────────────────────────────────────")
print(f"  Total new features added   : {total_new}")
print(f"  Final shape                : {df.shape[0]} rows × {df.shape[1]} columns")

# Check NaN summary for new columns
new_cols = list(interaction_df.columns) + list(rolling_df.columns) + list(poly_df_new.columns)
nan_counts = df[new_cols].isna().sum()
nan_nonzero = nan_counts[nan_counts > 0]
if len(nan_nonzero) > 0:
    print("\n  NaN counts in new features (non-zero only):")
    for col, cnt in nan_nonzero.items():
        print(f"    {col[:70]:70s}  NaN={cnt}")
else:
    print("\n  No NaN values in new features.")

print(f"\nSaving to: {OUTPUT_PATH} ...")
df.to_csv(OUTPUT_PATH, index=False)
print("Done.")
