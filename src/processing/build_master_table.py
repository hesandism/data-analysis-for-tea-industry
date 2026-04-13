# flake8: noqa
"""
build_master_table.py
=====================
Builds a master analytical table from the 9 Forbes & Walker CSV files.

GRANULARITY: One row = one price observation (sale × grade/category × tier/segment/elevation)
  - ~115 price rows per sale → ~1,154 rows for 10 sales → grows linearly with every new PDF added
  - Each row is enriched with all sale-level context: auction stats, demand, weather, FX, sentiment

SOURCES:
  04_high_grown_prices  → Orthodox high-grown price bands  (segment × grade)
  05_low_grown_prices   → Low-grown price bands            (grade × tier)
  06_offgrade_dust_prices → Off-grade & dust price bands   (category_type × category × elevation)
  -- above three form the SPINE of the master table --
  01_sales_index        → Sale-level context (sentinel, totals, sentiment, crop, FX, gross avgs)
  02_auction_offerings  → Demand scores & qty offered, pivoted wide per category
  03_quantity_sold      → Channel volumes (private/public/forward) + FX rates
  09_weather_features   → API weather + lag features, pivoted wide per region

OUTPUT:
  master_tea_prices.csv

HOW TO USE:
  python build_master_table.py                         # uses defaults below
  python build_master_table.py --data_dir ./my_csvs   # custom input folder
  python build_master_table.py --out ./outputs/master.csv

ADDING NEW SALES:
  Just re-run the script after your extraction pipeline has updated the 9 CSVs.
  The master table is fully rebuilt from scratch each run (idempotent).
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

_ROOT            = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = _ROOT / "data" / "interim"
DEFAULT_OUT      = _ROOT / "data" / "processed" / "master_tea_prices.csv"

# Columns from 01_sales_index we want as sale-level context.
# Excludes: source_file, extracted_at (pipeline metadata), raw commentary text (too long),
#           settlement dates (not analytical), and the 70+ individual gross avg breakdown
#           columns (we keep only the *_summary_* rollups which are the meaningful ones).
SALES_INDEX_KEEP = [
    # identity
    "sale_id", "sale_number", "sale_date_raw", "sale_year", "sale_month",
    # auction totals
    "total_lots", "total_kgs", "reprint_lots", "reprint_quantity",
    # sentiment (rule-based NLP scores from commentary)
    "sentiment_overall", "sentiment_ex_estate", "sentiment_low_grown",
    # weather (text-extracted severity 1–5 per region)
    "western_nuwara_eliya_weather_score",
    "uva_udapussellawa_weather_score",
    "low_grown_weather_score",
    "avg_weather_severity",
    # crop intake trends (+1 increase / 0 stable / -1 decrease)
    "crop_nuwara_eliya_trend", "crop_western_trend",
    "crop_uva_trend", "crop_low_grown_trend",
    # Sri Lanka national production
    "sl_production_mkgs", "sl_production_yoy_variance",
    # FX rates (LKR per foreign currency unit)
    "fx_usd_2026", "fx_usd_2025", "fx_usd_2024",
    "fx_gbp_2026", "fx_gbp_2025", "fx_gbp_2024",
    "fx_eur_2026", "fx_eur_2025", "fx_eur_2024",
    "fx_jpy_2026", "fx_jpy_2025", "fx_jpy_2024",
    # Gross sales average summary (weekly + todate, 2026 vs 2025) — high/medium/low + total
    "gross_lkr_weekly_high_summary_2026",  "gross_lkr_weekly_high_summary_2025",
    "gross_lkr_todate_high_summary_2026",  "gross_lkr_todate_high_summary_2025",
    "gross_lkr_weekly_medium_summary_2026","gross_lkr_weekly_medium_summary_2025",
    "gross_lkr_todate_medium_summary_2026","gross_lkr_todate_medium_summary_2025",
    "gross_lkr_weekly_low_summary_2026",   "gross_lkr_weekly_low_summary_2025",
    "gross_lkr_todate_low_summary_2026",   "gross_lkr_todate_low_summary_2025",
    "gross_lkr_weekly_total_2026",         "gross_lkr_weekly_total_2025",
    "gross_lkr_todate_total_2026",         "gross_lkr_todate_total_2025",
]

# Columns from 03_quantity_sold that are NOT already in 01_sales_index
# (01 has all the same channel volume and FX columns; 03 is effectively a clean extract)
# We still pull from 03 as a fallback in case 01 is missing some of these.
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

# Weather metrics to pivot per region (excluding text fields, lat/lon, fetch dates)
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
    # Lag features (previous 1, 2, 3 sales)
    "precipitation_sum_total_lag1", "precipitation_sum_total_lag2", "precipitation_sum_total_lag3",
    "rain_sum_total_lag1", "rain_sum_total_lag2", "rain_sum_total_lag3",
    "temperature_2m_mean_mean_lag1", "temperature_2m_mean_mean_lag2", "temperature_2m_mean_mean_lag3",
    "sunshine_duration_total_lag1", "sunshine_duration_total_lag2", "sunshine_duration_total_lag3",
    "relative_humidity_2m_max_max_lag1", "relative_humidity_2m_max_max_lag2", "relative_humidity_2m_max_max_lag3",
    "text_condition_score_lag1", "text_condition_score_lag2", "text_condition_score_lag3",
]

# Offering metrics to pivot per category (from 02_auction_offerings)
OFFERING_METRICS = ["qty_mkgs", "demand_score", "lots", "kgs"]


# ---------------------------------------------------------------------------
# STEP 1: Build the price observation spine
# ---------------------------------------------------------------------------

def build_spine(data_dir: Path) -> pd.DataFrame:
    """
    Combine 04, 05, 06 into a unified long table.
    Every row = one price observation (sale × grade/category combination).
    
    Unified schema:
      sale_id | table_source | elevation | category_type | grade | segment | tier | category | price_lo_lkr | price_hi_lkr
    """
    frames = []

    # --- 04: High Grown prices ---
    # Schema: sale_id, elevation, segment, grade, price_lo_lkr, price_hi_lkr
    df4 = pd.read_csv(data_dir / "04_high_grown_prices.csv")
    df4 = df4.assign(
        table_source="04_high_grown",
        category_type="high_grown",
        tier=None,
        category=df4["segment"],         # segment plays the role of category here
    )
    frames.append(df4)

    # --- 05: Low Grown prices ---
    # Schema: sale_id, elevation, grade, tier, price_lo_lkr, price_hi_lkr
    df5 = pd.read_csv(data_dir / "05_low_grown_prices.csv")
    df5 = df5.assign(
        table_source="05_low_grown",
        category_type="low_grown",
        segment=None,
        category=df5["grade"],           # grade plays the role of category
    )
    frames.append(df5)

    # --- 06: Off-grade & Dust prices ---
    # Schema: sale_id, category_type, category, elevation, price_lo_lkr, price_hi_lkr
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

    # Derived: price midpoint (useful as a single numeric target)
    spine["price_mid_lkr"] = (spine["price_lo_lkr"].fillna(0) + spine["price_hi_lkr"].fillna(0)) / 2
    spine.loc[spine["price_hi_lkr"].isna(), "price_mid_lkr"] = spine.loc[spine["price_hi_lkr"].isna(), "price_lo_lkr"]

    # Derived: price range width (proxy for quality spread / market uncertainty)
    spine["price_range_lkr"] = (spine["price_hi_lkr"] - spine["price_lo_lkr"]).clip(lower=0)

    print(f"  [spine]    {len(spine):,} rows × {len(spine.columns)} cols  "
          f"({spine['sale_id'].nunique()} unique sales)")
    return spine


# ---------------------------------------------------------------------------
# STEP 2: Build sale-level context from 01 + 03
# ---------------------------------------------------------------------------

def build_sale_context(data_dir: Path) -> pd.DataFrame:
    """
    Pull sale-level context from 01_sales_index (primary) and 03_quantity_sold (supplement).
    Returns one row per sale_id.
    """
    df1 = pd.read_csv(data_dir / "01_sales_index.csv")
    df3 = pd.read_csv(data_dir / "03_quantity_sold.csv")

    # Keep only the columns we care about from 01
    available_01 = [c for c in SALES_INDEX_KEEP if c in df1.columns]
    ctx = df1[available_01].copy()

    # Add qty_sold columns from 03 (they may already be in 01, but 03 is the canonical source)
    available_03 = [c for c in QTY_SOLD_UNIQUE if c in df3.columns and c not in ctx.columns]
    if available_03:
        ctx = ctx.merge(df3[["sale_id"] + available_03], on="sale_id", how="left")

    # Derived: USD-adjusted gross average price for current year
    # This helps compare across sales when FX is moving
    if "gross_lkr_weekly_total_2026" in ctx.columns and "fx_usd_2026" in ctx.columns:
        ctx["gross_usd_weekly_total_2026"] = (
            ctx["gross_lkr_weekly_total_2026"] / ctx["fx_usd_2026"]
        ).round(4)

    # Derived: YoY volume change (how much more/less was sold vs same period last year)
    if "total_sold_weekly_2026" in ctx.columns and "total_sold_weekly_2025" in ctx.columns:
        ctx["volume_yoy_change_pct"] = (
            (ctx["total_sold_weekly_2026"] - ctx["total_sold_weekly_2025"])
            / ctx["total_sold_weekly_2025"].replace(0, np.nan) * 100
        ).round(2)

    print(f"  [context]  {len(ctx):,} rows × {len(ctx.columns)} cols  "
          f"(sale-level context)")
    return ctx


# ---------------------------------------------------------------------------
# STEP 3: Pivot auction offerings wide (02_auction_offerings)
# ---------------------------------------------------------------------------

def build_offerings_pivot(data_dir: Path) -> pd.DataFrame:
    """
    Pivot 02_auction_offerings so each category becomes a set of columns.
    Output: one row per sale_id, columns like:
      off_grade__demand_score, leafy__qty_mkgs, total__lots ...
    """
    df2 = pd.read_csv(data_dir / "02_auction_offerings.csv")

    available_metrics = [m for m in OFFERING_METRICS if m in df2.columns]
    pivoted = df2.pivot_table(
        index="sale_id",
        columns="category",
        values=available_metrics,
        aggfunc="first",
    )
    # Flatten MultiIndex columns: (metric, category) → category__metric
    pivoted.columns = [f"{cat}__{metric}" for metric, cat in pivoted.columns]
    pivoted = pivoted.reset_index()

    print(f"  [offerings] {len(pivoted):,} rows × {len(pivoted.columns)} cols  "
          f"(pivoted demand/qty per category)")
    return pivoted


# ---------------------------------------------------------------------------
# STEP 4: Pivot weather features wide (09_weather_features)
# ---------------------------------------------------------------------------

def build_weather_pivot(data_dir: Path) -> pd.DataFrame:
    """
    Pivot 09_weather_features so each region becomes a set of columns.
    Output: one row per sale_id, columns like:
      low_grown__precipitation_sum_total, western_high__temperature_2m_mean_mean_lag1 ...
    """
    df9 = pd.read_csv(data_dir / "09_weather_features.csv")

    available_metrics = [m for m in WEATHER_METRICS if m in df9.columns]
    pivoted = df9.pivot_table(
        index="sale_id",
        columns="region",
        values=available_metrics,
        aggfunc="first",
    )
    # Flatten MultiIndex: (metric, region) → region__metric
    pivoted.columns = [f"{region}__{metric}" for metric, region in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Derived: cross-region average precipitation (useful as a single national weather signal)
    rain_cols = [c for c in pivoted.columns if c.endswith("__precipitation_sum_total") and "lag" not in c]
    if rain_cols:
        pivoted["all_regions__avg_precipitation"] = pivoted[rain_cols].mean(axis=1).round(2)

    print(f"  [weather]  {len(pivoted):,} rows × {len(pivoted.columns)} cols  "
          f"(pivoted weather per region)")
    return pivoted


# ---------------------------------------------------------------------------
# STEP 5: Join everything together
# ---------------------------------------------------------------------------

def build_master(data_dir: Path, out_path: Path) -> pd.DataFrame:
    print("\n=== Building master table ===\n")

    spine    = build_spine(data_dir)
    context  = build_sale_context(data_dir)
    offering = build_offerings_pivot(data_dir)
    weather  = build_weather_pivot(data_dir)

    # Join sale-level tables onto the spine (left join to keep all price rows)
    master = (
        spine
        .merge(context,  on="sale_id", how="left")
        .merge(offering, on="sale_id", how="left")
        .merge(weather,  on="sale_id", how="left")
    )

    # ---------------------------------------------------------------------------
    # Column ordering: identity first, then price cols, then sale context,
    # then demand/offering, then weather
    # ---------------------------------------------------------------------------
    identity_cols = [
        "sale_id", "sale_number", "sale_year", "sale_month", "sale_date_raw",
        "table_source", "elevation", "category_type", "grade", "segment", "tier", "category",
        "price_lo_lkr", "price_hi_lkr", "price_mid_lkr", "price_range_lkr",
    ]
    context_cols  = [c for c in context.columns  if c not in identity_cols and c != "sale_id"]
    offering_cols = [c for c in offering.columns if c not in identity_cols and c != "sale_id"]
    weather_cols  = [c for c in weather.columns  if c not in identity_cols and c != "sale_id"]

    ordered = identity_cols + context_cols + offering_cols + weather_cols
    ordered = [c for c in ordered if c in master.columns]       # safety: only keep present cols
    remaining = [c for c in master.columns if c not in ordered] # catch any stragglers
    master = master[ordered + remaining]

    # Sort by sale_year, sale_number, then price observation order
    sort_by = [c for c in ["sale_year", "sale_number", "table_source", "category", "tier"] if c in master.columns]
    master = master.sort_values(sort_by).reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"  MASTER TABLE COMPLETE")
    print(f"  Rows   : {len(master):,}")
    print(f"  Columns: {len(master.columns)}")
    print(f"  Sales  : {master['sale_id'].nunique()} unique sales")
    print(f"  Output : {out_path}")
    print(f"{'='*60}")

    # Column group summary
    print(f"\n  Column breakdown:")
    print(f"    Identity / price cols : {len(identity_cols)}")
    print(f"    Sale context (01+03)  : {len(context_cols)}")
    print(f"    Demand pivot (02)     : {len(offering_cols)}")
    print(f"    Weather pivot (09)    : {len(weather_cols)}")

    return master


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build master tea auction table")
    parser.add_argument(
        "--data_dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Folder containing the 9 source CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output CSV path (default: %(default)s)",
    )
    args = parser.parse_args()

    master = build_master(args.data_dir, args.out)

    # Quick sanity check printout
    print(f"\n  Sample (first 3 rows, first 10 cols):")
    print(master.iloc[:3, :10].to_string(index=False))
