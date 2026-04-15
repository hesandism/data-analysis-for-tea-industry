from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


# Minimum columns needed from each required CSV.
SALES_COLS: List[str] = [
    "sale_id",
    "sale_number",
    "sale_year",
    "sale_month",
    "total_kgs",
    "total_sold_weekly_2026",
    "total_sold_weekly_2025",
    "sentiment_overall",
    "fx_usd_2026",
    "sl_production_mkgs",
    "gross_lkr_weekly_western_high_2026",
    "gross_lkr_weekly_uva_high_2026",
    "gross_lkr_weekly_western_medium_2026",
    "gross_lkr_weekly_uva_medium_2026",
    "gross_lkr_weekly_ctc_medium_2026",
    "gross_lkr_weekly_orthodox_low_2026",
]

OFFERINGS_COLS: List[str] = [
    "sale_id",
    "category",
    "demand_score",
    "kgs",
]

HIGH_GROWN_COLS: List[str] = [
    "sale_id",
    "elevation",
    "segment",
    "grade",
    "price_lo_lkr",
    "price_hi_lkr",
]

LOW_GROWN_COLS: List[str] = [
    "sale_id",
    "elevation",
    "grade",
    "tier",
    "price_lo_lkr",
    "price_hi_lkr",
]


def _is_project_root(path: Path) -> bool:
    return (path / "tea_output").is_dir()


def _walk_up_to_project_root(start_path: Path) -> Path | None:
    for candidate in [start_path, *start_path.parents]:
        if _is_project_root(candidate):
            return candidate
    return None


def _resolve_project_root(base_dir: str | Path | None = None) -> Path:
    # Priority 1: explicit base_dir from caller.
    if base_dir is not None:
        explicit = Path(base_dir).expanduser().resolve()
        resolved = _walk_up_to_project_root(explicit)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(
            f"Could not find project root above provided path: {explicit}"
        )

    # Priority 2: directory of this module file.
    module_dir = Path(__file__).resolve().parent
    resolved_from_module = _walk_up_to_project_root(module_dir)
    if resolved_from_module is not None:
        return resolved_from_module

    # Priority 3: current working directory.
    cwd = Path.cwd().resolve()
    resolved_from_cwd = _walk_up_to_project_root(cwd)
    if resolved_from_cwd is not None:
        return resolved_from_cwd

    raise FileNotFoundError(
        "Could not auto-locate project root. Expected a parent directory containing 'tea_output/'."
    )


def load_minimal_tables(base_dir: str | Path | None = None) -> Dict[str, pd.DataFrame]:
    """Load only the required files and columns for grade-focused analysis."""
    base = _resolve_project_root(base_dir)
    tea_output = base / "tea_output"

    sales = pd.read_csv(tea_output / "01_sales_index.csv", usecols=SALES_COLS)
    offerings = pd.read_csv(tea_output / "02_auction_offerings.csv", usecols=OFFERINGS_COLS)
    high_grown = pd.read_csv(tea_output / "04_high_grown_prices.csv", usecols=HIGH_GROWN_COLS)
    low_grown = pd.read_csv(tea_output / "05_low_grown_prices.csv", usecols=LOW_GROWN_COLS)

    return {
        "sales": sales,
        "offerings": offerings,
        "high_grown": high_grown,
        "low_grown": low_grown,
    }


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _add_common_features(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = df.copy()
    out = _to_numeric(out, ["price_lo_lkr", "price_hi_lkr"])
    out["price_mid_lkr"] = (out["price_lo_lkr"] + out["price_hi_lkr"]) / 2.0
    out["price_range_lkr"] = out["price_hi_lkr"] - out["price_lo_lkr"]
    out = out.sort_values([group_col, "sale_number"]).reset_index(drop=True)
    out["wow_price_mid_change"] = out.groupby(group_col)["price_mid_lkr"].diff()
    return out


def _build_demand_features(offerings: pd.DataFrame) -> pd.DataFrame:
    off = offerings.copy()
    off = _to_numeric(off, ["demand_score", "kgs"])

    category_map = {
        "demand_bopf": ["ex_estate", "high_medium"],
        "demand_fbop": ["high_medium"],
        "demand_op1": ["leafy", "semi_leafy"],
    }

    agg = {}
    for feature_name, categories in category_map.items():
        tmp = off.loc[off["category"].isin(categories), ["sale_id", "demand_score"]]
        agg[feature_name] = (
            tmp.groupby("sale_id", as_index=False)["demand_score"]
            .mean()
            .rename(columns={"demand_score": feature_name})
        )

    demand_features = agg["demand_bopf"]
    demand_features = demand_features.merge(agg["demand_fbop"], on="sale_id", how="outer")
    demand_features = demand_features.merge(agg["demand_op1"], on="sale_id", how="outer")
    return demand_features


def build_grade_dataframes(base_dir: str | Path | None = None) -> Dict[str, pd.DataFrame]:
    """
    Build clear, separate dataframes for the selected grades:
    - High Grown BOPF
    - High Grown FBOP (grade label: pekoe_fbop)
    - Low Grown OP1
    """
    tables = load_minimal_tables(base_dir)
    sales = tables["sales"].copy()
    offerings = tables["offerings"].copy()
    high_grown = tables["high_grown"].copy()
    low_grown = tables["low_grown"].copy()

    sales = _to_numeric(
        sales,
        [
            "sale_number",
            "sale_year",
            "total_kgs",
            "total_sold_weekly_2026",
            "total_sold_weekly_2025",
            "sentiment_overall",
            "fx_usd_2026",
            "sl_production_mkgs",
            "gross_lkr_weekly_western_high_2026",
            "gross_lkr_weekly_uva_high_2026",
            "gross_lkr_weekly_western_medium_2026",
            "gross_lkr_weekly_uva_medium_2026",
            "gross_lkr_weekly_ctc_medium_2026",
            "gross_lkr_weekly_orthodox_low_2026",
        ],
    )

    sales["sold_to_offered_ratio"] = sales["total_sold_weekly_2026"] / sales["total_kgs"]
    sales["yoy_volume_change_pct"] = (
        (sales["total_sold_weekly_2026"] - sales["total_sold_weekly_2025"])
        / sales["total_sold_weekly_2025"]
        * 100.0
    )

    demand_features = _build_demand_features(offerings)
    sales_base = sales.merge(demand_features, on="sale_id", how="left")

    hg_focus = high_grown.loc[
        (high_grown["elevation"] == "high_grown")
        & (high_grown["grade"].isin(["bopf", "pekoe_fbop"]))
    ].copy()

    op1_focus = low_grown.loc[
        (low_grown["elevation"] == "low_grown") & (low_grown["grade"] == "op1")
    ].copy()

    bopf = hg_focus.loc[hg_focus["grade"] == "bopf"].copy()
    fbop = hg_focus.loc[hg_focus["grade"] == "pekoe_fbop"].copy()

    bopf = bopf.merge(sales_base, on="sale_id", how="left")
    fbop = fbop.merge(sales_base, on="sale_id", how="left")
    op1 = op1_focus.merge(sales_base, on="sale_id", how="left")

    bopf = _add_common_features(bopf, group_col="segment")
    fbop = _add_common_features(fbop, group_col="segment")
    op1 = _add_common_features(op1, group_col="tier")

    bopf_summary = (
        bopf.groupby("sale_id", as_index=False)["price_mid_lkr"]
        .mean()
        .rename(columns={"price_mid_lkr": "bopf_mid_mean_lkr"})
    )
    fbop_summary = (
        fbop.groupby("sale_id", as_index=False)["price_mid_lkr"]
        .mean()
        .rename(columns={"price_mid_lkr": "fbop_mid_mean_lkr"})
    )
    op1_summary = (
        op1.groupby("sale_id", as_index=False)["price_mid_lkr"]
        .mean()
        .rename(columns={"price_mid_lkr": "op1_mid_mean_lkr"})
    )

    grade_summary = sales_base[["sale_id", "sale_number", "sale_month"]].drop_duplicates()
    grade_summary = grade_summary.merge(bopf_summary, on="sale_id", how="left")
    grade_summary = grade_summary.merge(fbop_summary, on="sale_id", how="left")
    grade_summary = grade_summary.merge(op1_summary, on="sale_id", how="left")

    return {
        "sales_base": sales_base,
        "bopf_df": bopf,
        "fbop_df": fbop,
        "op1_df": op1,
        "grade_summary_df": grade_summary,
    }


if __name__ == "__main__":
    dfs = build_grade_dataframes()
    for name, df in dfs.items():
        print(f"{name}: {df.shape}")