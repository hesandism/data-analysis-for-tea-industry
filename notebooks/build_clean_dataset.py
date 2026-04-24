"""
Build a clean, publication-ready dataset from interim_2024_oldpl CSV files.

Outputs:
1) final_clean_price_table.csv
   - Unified price observations from 04/05/06 with consistent schema.
2) final_clean_sale_weather.csv
   - Sale-level + weather-level features keyed by sale_id.
3) final_clean_dataset_long.csv
   - Merge of unified price table with sale+weather on sale_id.

Run:
  python notebook/build_clean_dataset.py
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT / "data" / "Interim" / "interim_2024_oldpl"
OUTPUT_DIR = ROOT / "data" / "processed-2024"

HIGH_GROWN_REGIONS = ["nuwara_eliya", "uva_udapussellawa", "western_high"]
LOW_GROWN_REGIONS = ["low_grown"]


# Explicit high-grown segment to weather-region mapping.
HIGH_GROWN_SEGMENT_TO_REGION = {
	"best_western": "western_high",
	"below_best_western": "western_high",
	"plainer_western": "western_high",
	"nuwara_eliya": "nuwara_eliya",
	"brighter_udapussellawa": "uva_udapussellawa",
	"other_udapussellawa": "uva_udapussellawa",
	"best_uva": "uva_udapussellawa",
	"other_uva": "uva_udapussellawa",
}


def _read_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Missing required file: {path}")
	return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise ValueError(f"{table_name} is missing required columns: {missing}")


def remove_unknown_sale_ids(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
	"""Drop rows with missing/unknown sale_id values (e.g., UNKNOWN_... / unknown_sale_...)."""
	if "sale_id" not in df.columns:
		return df

	out = df.copy()
	sale_id_str = out["sale_id"].astype("string")
	unknown_mask = sale_id_str.isna() | sale_id_str.str.match(r"(?i)^unknown(_sale_)?", na=False)
	removed = int(unknown_mask.sum())
	if removed:
		print(f"[filter] {table_name}: removed {removed} row(s) with unknown sale_id")
	out = out.loc[~unknown_mask].reset_index(drop=True)
	return out


def _normalize_region(value: object) -> str | None:
	if pd.isna(value):
		return None
	text = str(value).strip().lower().replace("-", " ").replace("/", " ")
	text = " ".join(text.split())

	if "low" in text or "matara" in text or "galle" in text:
		return "low_grown"
	if "nuwara" in text and "eliya" in text:
		return "nuwara_eliya"
	if "western" in text and "high" in text:
		return "western_high"
	if "uva" in text or "udapuss" in text:
		return "uva_udapussellawa"
	return None


def _regions_for_elevation(elevation: object, category_type: object) -> list[str]:
	elev = "" if pd.isna(elevation) else str(elevation).strip().lower()
	ctype = "" if pd.isna(category_type) else str(category_type).strip().lower()

	if elev in {"medium", "medium_grown"}:
		# Requested rule: use Uva/Udapussellawa for medium-grown.
		return ["uva_udapussellawa"]

	if ctype == "low_grown" or elev in {"low", "low_grown"}:
		return LOW_GROWN_REGIONS

	if ctype == "high_grown" or elev in {"high", "high_grown"}:
		return HIGH_GROWN_REGIONS

	# Off-grade/dust with missing elevation: keep both climate bands.
	return HIGH_GROWN_REGIONS + LOW_GROWN_REGIONS


def _regions_for_offgrade_dust(elevation: object, category_type: object) -> list[str]:
	"""Map off-grade/dust rows to weather regions strictly by elevation."""
	elev = "" if pd.isna(elevation) else str(elevation).strip().lower()
	ctype = "" if pd.isna(category_type) else str(category_type).strip().lower()

	if elev in {"high", "high_grown"}:
		return HIGH_GROWN_REGIONS
	if elev in {"medium", "medium_grown"}:
		# Requested rule for medium-grown fallback region.
		return ["uva_udapussellawa"]
	if elev in {"low", "low_grown"}:
		return LOW_GROWN_REGIONS

	# Fallback for rare missing/unknown elevations in 06 table.
	if ctype in {"off_grade", "dust"}:
		return HIGH_GROWN_REGIONS + LOW_GROWN_REGIONS
	return _regions_for_elevation(elevation, category_type)


def _regions_for_price_row(row: pd.Series) -> list[str]:
	# For high-grown rows, always map weather using the explicit segment mapping.
	if str(row.get("source_table", "")).strip().lower() == "04_high_grown_prices":
		segment = str(row.get("segment", "")).strip().lower()
		mapped = HIGH_GROWN_SEGMENT_TO_REGION.get(segment)
		if mapped:
			return [mapped]
		# Fallback for unexpected high-grown segment labels.
		return HIGH_GROWN_REGIONS

	# For off-grade and dust rows, use dedicated elevation-based mapping.
	if str(row.get("source_table", "")).strip().lower() == "06_offgrade_dust_prices":
		return _regions_for_offgrade_dust(row.get("elevation"), row.get("category_type"))

	# For other tables, use elevation/category-based weather mapping.
	return _regions_for_elevation(row.get("elevation"), row.get("category_type"))


def _first_region_for_price_row(row: pd.Series) -> str | None:
	regions = _regions_for_price_row(row)
	return regions[0] if regions else None


def build_sale_index(df_sales: pd.DataFrame) -> pd.DataFrame:
	keep = [
		"sale_id",
		"sale_number",
		"sale_year",
		"sale_date_raw",
		"avg_weather_severity",
		"fx_usd_2026",
		"fx_usd_2025",
		"fx_usd_2024",
	]
	_require_columns(df_sales, keep, "01_sales_index")

	sales = df_sales[keep].copy()
	sales = sales.drop_duplicates(subset=["sale_id"]).reset_index(drop=True)
	return sales


def consolidate_fx_usd(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Create one unified fx_usd column from year-specific fx_usd_YYYY columns.
	Rule: use fx_usd_<sale_year> when available; otherwise fallback to first non-null.
	Then drop the year-specific fx_usd columns.
	"""
	out = df.copy()

	# Ensure sale_year exists or derive from sale_id when possible.
	if "sale_year" not in out.columns:
		out["sale_year"] = pd.NA
	if out["sale_year"].isna().any() and "sale_id" in out.columns:
		derived_year = (
			out["sale_id"].astype("string").str.extract(r"SALE_(\d{4})_", expand=False)
		)
		out["sale_year"] = out["sale_year"].fillna(derived_year)
	out["sale_year"] = pd.to_numeric(out["sale_year"], errors="coerce")

	fx_cols = [c for c in out.columns if re.fullmatch(r"fx_usd_\d{4}", c)]
	if not fx_cols:
		out["fx_usd"] = pd.NA
		return out

	# Build a year -> column map, e.g. 2024 -> fx_usd_2024.
	year_to_col: dict[int, str] = {}
	for col in fx_cols:
		year = int(col.rsplit("_", 1)[-1])
		year_to_col[year] = col

	def pick_fx(row: pd.Series):
		year = row.get("sale_year")
		if pd.notna(year):
			year_col = year_to_col.get(int(year))
			if year_col is not None and pd.notna(row.get(year_col)):
				return row.get(year_col)
		# Fallback: first non-null from available fx_usd_YYYY columns.
		for col in sorted(fx_cols, reverse=True):
			val = row.get(col)
			if pd.notna(val):
				return val
		return pd.NA

	out["fx_usd"] = out.apply(pick_fx, axis=1)
	out = out.drop(columns=fx_cols)
	return out


def build_weather(df_weather: pd.DataFrame) -> pd.DataFrame:
	keep = [
		"sale_id",
		"region",
		"region_label",
		"text_crop_change",
		"text_keywords",
		"temperature_2m_mean_mean",
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
	_require_columns(df_weather, keep, "09_weather_features")

	weather = df_weather[keep].copy()
	region_norm_from_region = weather["region"].map(_normalize_region)
	region_norm_from_label = weather["region_label"].map(_normalize_region)
	weather["region_norm"] = region_norm_from_region.fillna(region_norm_from_label)
	weather = weather.dropna(subset=["region_norm"]).copy()

	# Validate sale-region consistency for text descriptors.
	text_cols_to_check = ["text_crop_change", "region", "region_label"]
	for col in text_cols_to_check:
		inconsistent = (
			weather.groupby(["sale_id", "region_norm"], dropna=False)[col]
			.nunique(dropna=True)
			.gt(1)
		)
		if inconsistent.any():
			count = int(inconsistent.sum())
			print(
				f"[warn] {count} sale-region group(s) have conflicting values in '{col}'. "
				"Keeping first non-null value."
			)

	# Collapse to exactly one deterministic row per sale_id + region_norm.
	def _first_non_null(series: pd.Series):
		non_null = series.dropna()
		return non_null.iloc[0] if not non_null.empty else pd.NA

	agg_map = {col: _first_non_null for col in keep}
	weather = (
		weather.groupby(["sale_id", "region_norm"], as_index=False)
		.agg(agg_map)
		.reset_index(drop=True)
	)
	return weather


def build_price_table(
	df_hg: pd.DataFrame,
	df_lg: pd.DataFrame,
	df_od: pd.DataFrame,
) -> pd.DataFrame:
	hg_need = ["sale_id", "elevation", "segment", "grade", "price_lo_lkr", "price_hi_lkr"]
	lg_need = ["sale_id", "elevation", "grade", "tier", "price_lo_lkr", "price_hi_lkr"]
	od_need = ["sale_id", "category_type", "category", "elevation", "price_lo_lkr", "price_hi_lkr"]

	_require_columns(df_hg, hg_need, "04_high_grown_prices")
	_require_columns(df_lg, lg_need, "05_low_grown_prices")
	_require_columns(df_od, od_need, "06_offgrade_dust_prices")

	hg = df_hg[hg_need].copy()
	hg["source_table"] = "04_high_grown_prices"
	hg["category_type"] = "high_grown"
	hg["tier"] = pd.NA
	hg["category"] = hg["segment"]

	lg = df_lg[lg_need].copy()
	lg["source_table"] = "05_low_grown_prices"
	lg["category_type"] = "low_grown"
	lg["segment"] = pd.NA
	lg["category"] = lg["grade"]

	od = df_od[od_need].copy()
	od["source_table"] = "06_offgrade_dust_prices"
	od["segment"] = pd.NA
	od["grade"] = pd.NA
	od["tier"] = pd.NA

	unified_cols = [
		"sale_id",
		"source_table",
		"category_type",
		"category",
		"elevation",
		"segment",
		"grade",
		"tier",
		"price_lo_lkr",
		"price_hi_lkr",
	]

	prices = pd.concat([hg[unified_cols], lg[unified_cols], od[unified_cols]], ignore_index=True)
	prices["price_mid_lkr"] = (
		prices["price_lo_lkr"].fillna(0) + prices["price_hi_lkr"].fillna(0)
	) / 2
	prices.loc[prices["price_hi_lkr"].isna(), "price_mid_lkr"] = prices.loc[
		prices["price_hi_lkr"].isna(), "price_lo_lkr"
	]

	prices = prices.drop_duplicates().reset_index(drop=True)
	return prices


def merge_prices_with_weather(prices: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
	# Keep one weather mapping per price row: select the first mapped region only.
	prices_expanded = prices.copy()
	prices_expanded["region_norm"] = prices_expanded.apply(
		_first_region_for_price_row,
		axis=1,
	)

	merged = prices_expanded.merge(
		weather,
		on=["sale_id", "region_norm"],
		how="left",
	)
	return merged


def apply_missing_value_policy(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Apply publication-friendly NA handling by data type.

	Policy:
	- Text/categorical fields -> "unknown"
	- Weather numeric fields   -> region-level median, then global median, then 0
	- Core price columns       -> keep as-is (do not force-fill to avoid distorting prices)
	"""
	out = df.copy()

	# 1) Fill text-like columns with an explicit unknown tag.
	text_cols = [
		"region",
		"region_label",
		"text_crop_change",
		"text_keywords",
		"segment",
		"grade",
		"tier",
		"category",
		"category_type",
		"elevation",
	]
	for col in text_cols:
		if col in out.columns:
			out[col] = out[col].fillna("unknown")

	# 2) Impute weather numeric columns without touching core price values.
	weather_numeric_cols = [
		"temperature_2m_mean_mean",
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

	for col in weather_numeric_cols:
		if col not in out.columns:
			continue

		# Ensure numeric dtype first.
		out[col] = pd.to_numeric(out[col], errors="coerce")

		if "region_norm" in out.columns:
			out[col] = out[col].fillna(
				out.groupby("region_norm", dropna=False)[col].transform("median")
			)

		# Global median fallback, then 0 as final safety net.
		global_median = out[col].median(skipna=True)
		if pd.isna(global_median):
			global_median = 0
		out[col] = out[col].fillna(global_median).fillna(0)

	return out


def build_outputs() -> None:
	df_sales = _read_csv(INTERIM_DIR / "01_sales_index.csv")
	df_hg = _read_csv(INTERIM_DIR / "04_high_grown_prices.csv")
	df_lg = _read_csv(INTERIM_DIR / "05_low_grown_prices.csv")
	df_od = _read_csv(INTERIM_DIR / "06_offgrade_dust_prices.csv")
	df_weather = _read_csv(INTERIM_DIR / "09_weather_features.csv")

	# Remove unknown sale IDs from all sources before feature construction and joins.
	df_sales = remove_unknown_sale_ids(df_sales, "01_sales_index")
	df_hg = remove_unknown_sale_ids(df_hg, "04_high_grown_prices")
	df_lg = remove_unknown_sale_ids(df_lg, "05_low_grown_prices")
	df_od = remove_unknown_sale_ids(df_od, "06_offgrade_dust_prices")
	df_weather = remove_unknown_sale_ids(df_weather, "09_weather_features")

	sale_index = build_sale_index(df_sales)
	sale_index = consolidate_fx_usd(sale_index)
	weather = build_weather(df_weather)
	prices = build_price_table(df_hg, df_lg, df_od)

	sale_weather = sale_index.merge(weather, on="sale_id", how="left")
	final_long = merge_prices_with_weather(prices, weather)
	final_long = final_long.merge(
		sale_index,
		on="sale_id",
		how="left",
		suffixes=("", "_sale"),
	)
	final_long = apply_missing_value_policy(final_long)

	# Keep long output clean for publication: remove internal/auxiliary region columns.
	cols_to_drop_from_long = [
		col
		for col in ["region_norm", "region_label", "price_lo_lkr", "price_hi_lkr", "source_table"]
		if col in final_long.columns
	]
	if cols_to_drop_from_long:
		final_long = final_long.drop(columns=cols_to_drop_from_long)

	# Final naming for publication.
	if "category_type" in final_long.columns:
		final_long = final_long.rename(columns={"category_type": "catalogue"})

	sort_cols_prices = [
		col for col in ["sale_id", "source_table", "category_type", "category", "tier"] if col in prices.columns
	]
	sort_cols_sale_weather = [
		col for col in ["sale_id", "region_norm", "region"] if col in sale_weather.columns
	]
	sort_cols_final = [
		col
		for col in ["sale_id", "source_table", "category_type", "category", "region_norm", "region"]
		if col in final_long.columns
	]

	if sort_cols_prices:
		prices = prices.sort_values(sort_cols_prices).reset_index(drop=True)
	if sort_cols_sale_weather:
		sale_weather = sale_weather.sort_values(sort_cols_sale_weather).reset_index(drop=True)
	if sort_cols_final:
		final_long = final_long.sort_values(sort_cols_final).reset_index(drop=True)

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	price_out = OUTPUT_DIR / "final_clean_price_table.csv"
	sale_weather_out = OUTPUT_DIR / "final_clean_sale_weather.csv"
	final_out = OUTPUT_DIR / "final_clean_dataset_long.csv"

	prices.to_csv(price_out, index=False)
	sale_weather.to_csv(sale_weather_out, index=False)
	final_long.to_csv(final_out, index=False)

	print("=== Final Clean Dataset Build Complete ===")
	print(f"Sales index rows: {len(sale_index):,}")
	print(f"Weather rows: {len(weather):,}")
	print(f"Unified price rows: {len(prices):,}")
	print(f"Sale+weather rows: {len(sale_weather):,}")
	print(f"Final long rows: {len(final_long):,}")
	print(f"Unique sale_id in final long: {final_long['sale_id'].nunique():,}")
	print(f"Rows missing matched weather region: {final_long['region'].isna().sum():,}")
	print("\nOutputs:")
	print(f"- {price_out}")
	print(f"- {sale_weather_out}")
	print(f"- {final_out}")


if __name__ == "__main__":
	build_outputs()
