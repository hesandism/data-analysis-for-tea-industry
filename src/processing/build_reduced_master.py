# flake8: noqa
"""
build_reduced_master.py
=======================
Builds the REDUCED analytical master table — 177 essential columns from 270.

Runs in two stages:
  Stage 1 — Full build   : combines all 9 source CSVs (same logic as build_master_table.py)
  Stage 2 — Column audit : drops redundant, constant, or high-null columns with clear reasoning

WHAT GETS DROPPED AND WHY
--------------------------
┌─────────────────────────────────────────────────────────────────────────────┐
│ Column(s)                            │ Reason                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ segment                              │ 84% null; identical to `category`    │
│                                      │ for the 04_high_grown rows           │
│ sl_production_yoy_variance           │ 89% null + single unique value       │
│ fx_usd/gbp/eur/jpy_2024             │ 2025 is sufficient YoY baseline      │
│ gross_lkr_*/gross_usd_*             │ 70% null (extraction gap in 7/10     │
│                                      │ sales); pipeline matures over time   │
│ *__rain_sum_total (current week)     │ r=1.00 with precipitation_sum_total  │
│ *__et0_fao_evapotranspiration_total  │ Derivable from temperature (r=0.68) │
│ *__text_has_thunder                  │ Always 0 across all 10 sales        │
│ western_high__text_has_bright        │ Always 1 — zero variance            │
│ low/nuwara__relative_humidity_max_max│ Always 100 — zero variance          │
│ *__temperature_2m_max/min_mean       │ Redundant; mean is sufficient       │
│ *__windspeed_10m_max_max             │ r=0.55–0.83 with max_mean; keep mean│
│ *__precipitation_sum_max_day         │ r=0.94 with sum_total               │
│ *__sunshine_duration_max_day         │ r=0.67 with total; total preferred  │
│ *__relative_humidity_2m_min_min      │ r=0.95 with min_mean                │
│ *__relative_humidity_max_max_lag1/2/3│ Source col near-constant (100)      │
│ *__lots, *__kgs (offerings)          │ r=0.93–0.98 with qty_mkgs           │
│ high_medium/tippy/total__demand_score│ Single unique value (=3) all sales  │
└─────────────────────────────────────────────────────────────────────────────┘

NOTE ON 'gross_lkr' AND 'sl_production' COLUMNS:
  These are structurally valid — the nulls are extraction gaps in the current
  10-sale dataset, not permanent. Re-enable them once >50% of sales have data
  by removing them from DROP_GROSS_AVG and DROP_PRODUCTION_YOY below.

NOTE ON 'sale_year':
  Currently all 2026 (zero variance with 10 sales) but intentionally KEPT
  because it will carry meaningful signal as you add data across years.

OUTPUT:  reduced_master_tea_prices.csv

USAGE:
  python build_reduced_master.py
  python build_reduced_master.py --data_dir ./my_csvs --out ./outputs/reduced.csv
  python build_reduced_master.py --null_threshold 0.5   # drop cols > 50% null
  python build_reduced_master.py --variance_threshold 1 # drop cols with ≤1 unique value
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_ROOT            = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = _ROOT / "data" / "Interim" / "interim_combined"
DEFAULT_OUT      = _ROOT / "data" / "processed" / "reduced_master_tea_prices.csv"

# ---------------------------------------------------------------------------
# STAGE 1 CONFIG  (identical to build_master_table.py)
# ---------------------------------------------------------------------------
REGIONS = ["low_grown", "nuwara_eliya", "uva_udapussellawa", "western_high"]

SALES_INDEX_KEEP = [
    "sale_id", "sale_number", "sale_date_raw", "sale_year", "sale_month",
    "total_lots", "total_kgs", "reprint_lots", "reprint_quantity",
    "sentiment_overall", "sentiment_ex_estate", "sentiment_low_grown",
    "western_nuwara_eliya_weather_score", "uva_udapussellawa_weather_score",
    "low_grown_weather_score", "avg_weather_severity",
    "crop_nuwara_eliya_trend", "crop_western_trend",
    "crop_uva_trend", "crop_low_grown_trend",
    "sl_production_mkgs", "sl_production_yoy_variance",
    "fx_usd_2026", "fx_usd_2025", "fx_usd_2024",
    "fx_gbp_2026", "fx_gbp_2025", "fx_gbp_2024",
    "fx_eur_2026", "fx_eur_2025", "fx_eur_2024",
    "fx_jpy_2026", "fx_jpy_2025", "fx_jpy_2024",
    "gross_lkr_weekly_high_summary_2026",  "gross_lkr_weekly_high_summary_2025",
    "gross_lkr_todate_high_summary_2026",  "gross_lkr_todate_high_summary_2025",
    "gross_lkr_weekly_medium_summary_2026","gross_lkr_weekly_medium_summary_2025",
    "gross_lkr_todate_medium_summary_2026","gross_lkr_todate_medium_summary_2025",
    "gross_lkr_weekly_low_summary_2026",   "gross_lkr_weekly_low_summary_2025",
    "gross_lkr_todate_low_summary_2026",   "gross_lkr_todate_low_summary_2025",
    "gross_lkr_weekly_total_2026",         "gross_lkr_weekly_total_2025",
    "gross_lkr_todate_total_2026",         "gross_lkr_todate_total_2025",
]

QTY_SOLD_UNIQUE = [
    "private_sales_weekly_2026",  "private_sales_weekly_2025",
    "private_sales_todate_2026",  "private_sales_todate_2025",
    "public_auction_weekly_2026", "public_auction_weekly_2025",
    "public_auction_todate_2026", "public_auction_todate_2025",
    "forward_contracts_weekly_2026", "forward_contracts_weekly_2025",
    "forward_contracts_todate_2026", "forward_contracts_todate_2025",
    "total_sold_weekly_2026",     "total_sold_weekly_2025",
    "total_sold_todate_2026",     "total_sold_todate_2025",
]

WEATHER_METRICS = [
    "text_condition_score",
    "text_has_rain", "text_has_mist", "text_has_bright", "text_has_thunder",
    "temperature_2m_max_mean", "temperature_2m_min_mean", "temperature_2m_mean_mean",
    "precipitation_sum_total", "precipitation_sum_max_day",
    "rain_sum_total", "rain_sum_max_day",
    "sunshine_duration_total", "sunshine_duration_max_day",
    "windspeed_10m_max_mean", "windspeed_10m_max_max",
    "et0_fao_evapotranspiration_total",
    "relative_humidity_2m_max_mean", "relative_humidity_2m_max_max",
    "relative_humidity_2m_min_mean", "relative_humidity_2m_min_min",
    "precipitation_sum_total_lag1", "precipitation_sum_total_lag2", "precipitation_sum_total_lag3",
    "rain_sum_total_lag1", "rain_sum_total_lag2", "rain_sum_total_lag3",
    "temperature_2m_mean_mean_lag1", "temperature_2m_mean_mean_lag2", "temperature_2m_mean_mean_lag3",
    "sunshine_duration_total_lag1", "sunshine_duration_total_lag2", "sunshine_duration_total_lag3",
    "relative_humidity_2m_max_max_lag1", "relative_humidity_2m_max_max_lag2", "relative_humidity_2m_max_max_lag3",
    "text_condition_score_lag1", "text_condition_score_lag2", "text_condition_score_lag3",
]

OFFERING_METRICS = ["qty_mkgs", "demand_score", "lots", "kgs"]

# ---------------------------------------------------------------------------
# STAGE 2 CONFIG — explicit drop lists (each with a reason tag)
# ---------------------------------------------------------------------------

# [A] segment: 84% null, identical to `category` for the rows that have it
DROP_SEGMENT = ["segment"]

# [B] FX historical: only 2024 dropped; 2025 kept as YoY baseline
DROP_FX_2024 = ["fx_usd_2024", "fx_gbp_2024", "fx_eur_2024", "fx_jpy_2024"]

# [C] Gross average prices: 70% null extraction gap; remove until data matures
DROP_GROSS_AVG = [
    "gross_lkr_weekly_high_summary_2026",  "gross_lkr_weekly_high_summary_2025",
    "gross_lkr_todate_high_summary_2026",  "gross_lkr_todate_high_summary_2025",
    "gross_lkr_weekly_medium_summary_2026","gross_lkr_weekly_medium_summary_2025",
    "gross_lkr_todate_medium_summary_2026","gross_lkr_todate_medium_summary_2025",
    "gross_lkr_weekly_low_summary_2026",   "gross_lkr_weekly_low_summary_2025",
    "gross_lkr_todate_low_summary_2026",   "gross_lkr_todate_low_summary_2025",
    "gross_lkr_weekly_total_2026",         "gross_lkr_weekly_total_2025",
    "gross_lkr_todate_total_2026",         "gross_lkr_todate_total_2025",
    "gross_usd_weekly_total_2026",
]

# [D] SL production YoY: 89% null + single unique value across available data
DROP_PRODUCTION_YOY = ["sl_production_yoy_variance"]

# [E] Weather — redundant current-week columns
DROP_WEATHER_REDUNDANT = [
    # rain_sum_total = precipitation_sum_total (r=1.00)
    *[f"{r}__rain_sum_total" for r in REGIONS],
    # et0 is derivable from temperature (r=0.68)
    *[f"{r}__et0_fao_evapotranspiration_total" for r in REGIONS],
    # temperature max & min means are redundant when mean_mean is present
    *[f"{r}__temperature_2m_max_mean" for r in REGIONS],
    *[f"{r}__temperature_2m_min_mean" for r in REGIONS],
    # windspeed max_max: r=0.55–0.83 with max_mean; keep max_mean only
    *[f"{r}__windspeed_10m_max_max" for r in REGIONS],
    # precipitation max_day: r=0.94 with total
    *[f"{r}__precipitation_sum_max_day" for r in REGIONS],
    # rain max_day: same metric as precipitation max_day, also redundant
    *[f"{r}__rain_sum_max_day" for r in REGIONS],
    # sunshine max_day: r=0.67 with total; total captures the week better
    *[f"{r}__sunshine_duration_max_day" for r in REGIONS],
    # humidity min_min: r=0.95 with min_mean
    *[f"{r}__relative_humidity_2m_min_min" for r in REGIONS],
]

# [F] Weather — zero-variance current columns (constant across all sales so far)
DROP_WEATHER_CONSTANT = [
    # thunder: always 0 in all 10 sales, all 4 regions
    *[f"{r}__text_has_thunder" for r in REGIONS],
    # western_high bright: always 1
    "western_high__text_has_bright",
    # humidity max_max: always 100 for low_grown and nuwara_eliya
    "low_grown__relative_humidity_2m_max_max",
    "nuwara_eliya__relative_humidity_2m_max_max",
]

# [G] Weather — lag columns whose source metric is near-constant
#     relative_humidity_2m_max_max is always 100 for low_grown and nuwara_eliya,
#     so its lags carry no information; uva/western lags have some variance but
#     we drop all for consistency (and because the source column itself is dropped)
DROP_WEATHER_CONSTANT_LAGS = [
    *[f"{r}__relative_humidity_2m_max_max_lag{l}"
      for r in REGIONS for l in [1, 2, 3]],
]

# [H] Offering lots and kgs: r=0.93–0.98 with qty_mkgs — qty_mkgs is the clean unit
DROP_OFFERING_LOTS_KGS = []   # built dynamically in stage 2 (column names not known until pivot)

# [I] Constant demand scores: single unique value across all 10 sales
DROP_DEMAND_CONSTANT = [
    "high_medium__demand_score",
    "tippy__demand_score",
    "total__demand_score",
]

# Combine all explicit drops into one set for fast lookup
ALL_EXPLICIT_DROPS: set[str] = set(
    DROP_SEGMENT
    + DROP_FX_2024
    + DROP_GROSS_AVG
    + DROP_PRODUCTION_YOY
    + DROP_WEATHER_REDUNDANT
    + DROP_WEATHER_CONSTANT
    + DROP_WEATHER_CONSTANT_LAGS
    + DROP_DEMAND_CONSTANT
)

# Columns that look zero-variance NOW (only 10 sales, all 2026) but WILL gain
# signal as more data accumulates. The dynamic audit must never touch these.
NEVER_DROP_DYNAMIC: set[str] = {
    "sale_year",           # all 2026 today; will span multiple years soon
    "sl_production_mkgs",  # 57% null today; fills as pipeline matures
}


# ---------------------------------------------------------------------------
# STAGE 1 HELPERS  (identical logic to build_master_table.py)
# ---------------------------------------------------------------------------

def build_spine(data_dir: Path) -> pd.DataFrame:
    frames = []

    df4 = pd.read_csv(data_dir / "04_high_grown_prices.csv")
    df4 = df4.assign(
        table_source="04_high_grown",
        category_type="high_grown",
        tier=None,
        category=df4["segment"],
    )
    frames.append(df4)

    df5 = pd.read_csv(data_dir / "05_low_grown_prices.csv")
    df5 = df5.assign(
        table_source="05_low_grown",
        category_type="low_grown",
        segment=None,
        category=df5["grade"],
    )
    frames.append(df5)

    df6 = pd.read_csv(data_dir / "06_offgrade_dust_prices.csv")
    df6 = df6.assign(
        table_source="06_offgrade_dust",
        segment=None,
        grade=None,
        tier=None,
    )
    frames.append(df6)

    spine_cols = [
        "sale_id", "table_source", "elevation", "category_type",
        "grade", "segment", "tier", "category",
        "price_lo_lkr", "price_hi_lkr",
    ]
    spine = pd.concat(frames, ignore_index=True)[spine_cols]
    spine["price_mid_lkr"]   = ((spine["price_lo_lkr"].fillna(0) + spine["price_hi_lkr"].fillna(0)) / 2).where(spine["price_lo_lkr"].notna())
    spine["price_range_lkr"] = (spine["price_hi_lkr"] - spine["price_lo_lkr"]).clip(lower=0)
    return spine


def build_sale_context(data_dir: Path) -> pd.DataFrame:
    df1 = pd.read_csv(data_dir / "01_sales_index.csv")
    df3 = pd.read_csv(data_dir / "03_quantity_sold.csv")

    available_01 = [c for c in SALES_INDEX_KEEP if c in df1.columns]
    ctx = df1[available_01].copy()

    available_03 = [c for c in QTY_SOLD_UNIQUE if c in df3.columns and c not in ctx.columns]
    if available_03:
        ctx = ctx.merge(df3[["sale_id"] + available_03], on="sale_id", how="left")

    if "gross_lkr_weekly_total_2026" in ctx.columns and "fx_usd_2026" in ctx.columns:
        ctx["gross_usd_weekly_total_2026"] = (
            ctx["gross_lkr_weekly_total_2026"] / ctx["fx_usd_2026"]
        ).round(4)

    if "total_sold_weekly_2026" in ctx.columns and "total_sold_weekly_2025" in ctx.columns:
        ctx["volume_yoy_change_pct"] = (
            (ctx["total_sold_weekly_2026"] - ctx["total_sold_weekly_2025"])
            / ctx["total_sold_weekly_2025"].replace(0, np.nan) * 100
        ).round(2)

    return ctx


def build_offerings_pivot(data_dir: Path) -> pd.DataFrame:
    df2 = pd.read_csv(data_dir / "02_auction_offerings.csv")
    available_metrics = [m for m in OFFERING_METRICS if m in df2.columns]
    pivoted = df2.pivot_table(
        index="sale_id", columns="category", values=available_metrics, aggfunc="first"
    )
    pivoted.columns = [f"{cat}__{metric}" for metric, cat in pivoted.columns]
    return pivoted.reset_index()


def build_weather_pivot(data_dir: Path) -> pd.DataFrame:
    df9 = pd.read_csv(data_dir / "09_weather_features.csv")
    available_metrics = [m for m in WEATHER_METRICS if m in df9.columns]
    pivoted = df9.pivot_table(
        index="sale_id", columns="region", values=available_metrics, aggfunc="first"
    )
    pivoted.columns = [f"{region}__{metric}" for metric, region in pivoted.columns]
    pivoted = pivoted.reset_index()

    rain_cols = [c for c in pivoted.columns if c.endswith("__precipitation_sum_total") and "lag" not in c]
    if rain_cols:
        pivoted["all_regions__avg_precipitation"] = pivoted[rain_cols].mean(axis=1).round(2)

    return pivoted


# ---------------------------------------------------------------------------
# STAGE 2 — Column reduction logic
# ---------------------------------------------------------------------------

def drop_lots_kgs(df: pd.DataFrame) -> list[str]:
    """Identify and return all __lots and __kgs offering columns."""
    return [c for c in df.columns if c.endswith("__lots") or c.endswith("__kgs")]


def dynamic_audit(
    df: pd.DataFrame,
    null_threshold: float,
    variance_threshold: int,
    verbose: bool,
) -> tuple[list[str], dict[str, str]]:
    """
    Scan numeric columns for near-zero variance or excess nulls.
    Returns (cols_to_drop, {col: reason}) for any columns caught dynamically
    that are NOT already in ALL_EXPLICIT_DROPS.

    This acts as a safety net for columns that are currently fine
    but degrade as new data arrives (e.g. a metric that is always 0
    in early sales but gains variance later — the explicit list handles
    known cases; this catches surprises).
    """
    auto_drop: dict[str, str] = {}
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if col in ALL_EXPLICIT_DROPS or col in NEVER_DROP_DYNAMIC:
            continue
        vals = df[col].dropna()
        null_pct = df[col].isna().mean()

        if null_pct > null_threshold:
            auto_drop[col] = f"auto: {null_pct*100:.0f}% null > threshold {null_threshold*100:.0f}%"
        elif len(vals) > 0 and vals.nunique() <= variance_threshold:
            auto_drop[col] = f"auto: only {vals.nunique()} unique value(s) — zero variance"

    if verbose and auto_drop:
        print(f"\n  [audit] {len(auto_drop)} additional columns caught by dynamic audit:")
        for col, reason in auto_drop.items():
            print(f"    {col}: {reason}")

    return list(auto_drop.keys()), auto_drop


def apply_column_reduction(
    df: pd.DataFrame,
    null_threshold: float,
    variance_threshold: int,
    verbose: bool,
) -> pd.DataFrame:
    """Apply all drop rules and return the reduced DataFrame."""

    # Build the full drop set
    explicit_in_df  = [c for c in ALL_EXPLICIT_DROPS if c in df.columns]
    lots_kgs        = drop_lots_kgs(df)
    auto_drops, _   = dynamic_audit(df, null_threshold, variance_threshold, verbose)

    all_drops = set(explicit_in_df + lots_kgs + auto_drops)

    if verbose:
        print(f"\n  [reduce] Drop breakdown:")
        print(f"    Explicit rules  : {len(explicit_in_df):3d} cols")
        print(f"    Lots/kgs (pivot): {len(lots_kgs):3d} cols")
        print(f"    Dynamic audit   : {len(auto_drops):3d} cols")
        print(f"    ─────────────────────────")
        print(f"    Total dropped   : {len(all_drops):3d} cols")
        print(f"    Kept            : {len(df.columns) - len(all_drops):3d} cols")

    return df.drop(columns=[c for c in all_drops if c in df.columns])


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def build_reduced_master(
    data_dir: Path,
    out_path: Path,
    null_threshold: float = 0.85,
    variance_threshold: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: build → join → reduce → save.

    Parameters
    ----------
    data_dir          : folder containing the 9 source CSV files
    out_path          : where to write the output CSV
    null_threshold    : dynamic audit drops numeric cols with null% > this (default 0.85)
    variance_threshold: dynamic audit drops numeric cols with ≤ this many unique values (default 1)
    verbose           : print progress and column group summaries
    """
    print("\n" + "="*62)
    print("  STAGE 1 — Build full master table")
    print("="*62)

    spine    = build_spine(data_dir)
    context  = build_sale_context(data_dir)
    offering = build_offerings_pivot(data_dir)
    weather  = build_weather_pivot(data_dir)

    master_full = (
        spine
        .merge(context,  on="sale_id", how="left")
        .merge(offering, on="sale_id", how="left")
        .merge(weather,  on="sale_id", how="left")
    )

    if verbose:
        print(f"\n  Full master : {master_full.shape[0]:,} rows × {master_full.shape[1]} cols")

    # -----------------------------------------------------------------------
    print("\n" + "="*62)
    print("  STAGE 2 — Column reduction")
    print("="*62)

    master_reduced = apply_column_reduction(
        master_full,
        null_threshold=null_threshold,
        variance_threshold=variance_threshold,
        verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # Ordering: identity cols first, then price, then sale context, then rest
    identity_cols = [
        "sale_id", "sale_number", "sale_year", "sale_month", "sale_date_raw",
        "table_source", "elevation", "category_type", "grade", "tier", "category",
        "price_lo_lkr", "price_hi_lkr", "price_mid_lkr", "price_range_lkr",
    ]
    ordered = [c for c in identity_cols if c in master_reduced.columns]
    rest    = [c for c in master_reduced.columns if c not in ordered]
    master_reduced = master_reduced[ordered + rest]

    # Sort
    sort_by = [c for c in ["sale_year", "sale_number", "table_source", "category", "tier"]
               if c in master_reduced.columns]
    master_reduced = master_reduced.sort_values(sort_by).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master_reduced.to_csv(out_path, index=False)

    # -----------------------------------------------------------------------
    # Summary report
    null_pct = (master_reduced.isnull().sum() / len(master_reduced) * 100).round(1)
    zero_null  = (null_pct == 0).sum()
    low_null   = ((null_pct > 0) & (null_pct <= 35)).sum()
    med_null   = ((null_pct > 35) & (null_pct <= 60)).sum()
    high_null  = (null_pct > 60).sum()

    print(f"\n{'='*62}")
    print(f"  REDUCED MASTER TABLE COMPLETE")
    print(f"{'='*62}")
    print(f"  Rows        : {len(master_reduced):,}")
    print(f"  Columns     : {master_reduced.shape[1]}  (from {master_full.shape[1]})")
    print(f"  Sales       : {master_reduced['sale_id'].nunique()} unique sales")
    print(f"  Output      : {out_path}")
    print(f"\n  Null profile of kept columns:")
    print(f"    0% null   : {zero_null} cols  — fully populated")
    print(f"    1–35% null: {low_null} cols  — sparse (mainly weather lags; fills as data grows)")
    print(f"    36–60%    : {med_null} cols  — use with caution")
    print(f"    >60%      : {high_null} cols  — remaining after reduction")

    if verbose:
        # Print column groups
        groups = {
            "Row identity"           : [c for c in master_reduced.columns if c in
                                         ["sale_id","sale_number","sale_year","sale_month",
                                          "sale_date_raw","table_source","elevation",
                                          "category_type","grade","tier","category"]],
            "Price (target)"         : [c for c in master_reduced.columns if "price_" in c],
            "Auction totals": [c for c in master_reduced.columns if c in
                               ["total_lots", "total_kgs", "reprint_lots", "reprint_quantity"]],
            "Sentiment": [c for c in master_reduced.columns if "sentiment" in c],
            "Weather severity (text)": [c for c in master_reduced.columns if "weather_score" in c or "avg_weather" in c],
            "Crop trends": [c for c in master_reduced.columns if "crop_" in c],
            "FX rates": [c for c in master_reduced.columns if "fx_" in c],
            "Volume sold": [c for c in master_reduced.columns if any(
                               k in c for k in ["sold", "auction_weekly", "auction_todate",
                                                 "private_sales", "forward_contracts"])],
            "Demand (pivot)": [c for c in master_reduced.columns if "__demand_score" in c],
            "Qty offered (pivot)": [c for c in master_reduced.columns if "__qty_mkgs" in c],
            "Production": [c for c in master_reduced.columns if "sl_production" in c],
            "Derived signals": [c for c in master_reduced.columns if c in
                                 ["volume_yoy_change_pct", "all_regions__avg_precipitation"]],
            "Weather API (current)": [c for c in master_reduced.columns if
                                      any(r + "__" in c for r in REGIONS) and "lag" not in c],
            "Weather API (lags)": [c for c in master_reduced.columns if
                                    any(r + "__" in c for r in REGIONS) and "lag" in c],
        }
        print("\n  Column group breakdown:")
        total_accounted = 0
        for grp, cols in groups.items():
            cols = [c for c in cols if c in master_reduced.columns]
            total_accounted += len(cols)
            avg_null = master_reduced[cols].isnull().mean().mean() * 100 if cols else 0
            print(f"    {grp:28s}: {len(cols):3d} cols  avg_null={avg_null:5.1f}%")
        unaccounted = master_reduced.shape[1] - total_accounted
        if unaccounted:
            print("    {'(other)':28s}: {unaccounted:3d} cols")

    print("\n  Sample — first 3 rows, identity + price cols:")
    sample_cols = [c for c in identity_cols if c in master_reduced.columns]
    print(master_reduced[sample_cols].head(3).to_string(index=False))

    return master_reduced


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build reduced (essential-columns-only) master tea auction table"
    )
    parser.add_argument(
        "--data_dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Folder with the 9 source CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--null_threshold", type=float, default=0.85,
        help="Dynamic audit: drop numeric cols with null%% above this fraction (default: 0.85)",
    )
    parser.add_argument(
        "--variance_threshold", type=int, default=1,
        help="Dynamic audit: drop numeric cols with ≤ this many unique values (default: 1)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed column-group breakdown",
    )
    args = parser.parse_args()

    build_reduced_master(
        data_dir=args.data_dir,
        out_path=args.out,
        null_threshold=args.null_threshold,
        variance_threshold=args.variance_threshold,
        verbose=not args.quiet,
    )
