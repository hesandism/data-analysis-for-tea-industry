"""
merge_price_weather.py
=======================
Joins your price CSV(s) with weather_features.csv on sale_id.

Usage:
    python merge_price_weather.py \
        --prices  prices.csv \
        --weather weather_features.csv \
        --output  modelling_dataset.csv

The price CSV is expected to have columns:
    sale_id | tea_type | region_category | grade | price_low | price_high
"""

import argparse
import pandas as pd
from pathlib import Path


# Maps the region_category values in your price CSV to the weather region keys
# produced by the pipeline. Adjust if your CSVs use different labels.
REGION_MAPPING = {
    "best_western":         "western_high",
    "below_best_western":   "western_high",
    "plainer_western":      "western_high",
    "best_uva":             "uva_udapussellawa",
    "other_uva":            "uva_udapussellawa",
    "brighter_udapussellawa": "uva_udapussellawa",
    "other_udapussellawa":  "uva_udapussellawa",
    "nuwara_eliya":         "nuwara_eliya",
    "low_grown":            "low_grown",
    "leafy":                "low_grown",
    "semi_leafy":           "low_grown",
    "tippy":                "low_grown",
}


def merge(prices_path: str, weather_path: str, output_path: str):
    prices = pd.read_csv(prices_path, sep=None, engine="python",
                          header=None if "sale_id" not in open(prices_path).readline() else 0)

    # Auto-assign column names if no header
    if isinstance(prices.columns[0], int):
        prices.columns = ["sale_id", "tea_type", "region_category",
                          "grade", "price_low", "price_high"]

    weather = pd.read_csv(weather_path)

    # Map price region labels → weather region keys
    prices["weather_region"] = prices["region_category"].str.lower().map(REGION_MAPPING)
    unmapped = prices[prices["weather_region"].isna()]["region_category"].unique()
    if len(unmapped):
        print(f"[WARN] Unmapped region categories (will drop): {unmapped}")
        prices = prices[prices["weather_region"].notna()]

    # Drop heavy text columns before merging (keep numeric + flags)
    drop_cols = [c for c in weather.columns if "raw_summary" in c or "label" in c]
    weather_slim = weather.drop(columns=drop_cols, errors="ignore")

    merged = prices.merge(
        weather_slim,
        left_on=["sale_id", "weather_region"],
        right_on=["sale_id", "region"],
        how="left",
    )

    # Add mid-price column for regression target
    merged["price_mid"] = (
        pd.to_numeric(merged["price_low"], errors="coerce") +
        pd.to_numeric(merged["price_high"], errors="coerce")
    ) / 2

    out = Path(output_path)
    merged.to_csv(out, index=False)
    print(f"✓ Merged dataset: {len(merged)} rows → {out}")
    print(f"  Columns: {list(merged.columns)}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices",  required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--output",  default="modelling_dataset.csv")
    args = parser.parse_args()
    merge(args.prices, args.weather, args.output)
