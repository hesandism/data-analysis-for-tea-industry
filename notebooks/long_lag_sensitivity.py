"""
Long-lag Granger sensitivity analysis.

Extends the original Granger analysis from 1-3 weeks to 1-8 weeks
using the same sale-level aggregation, stationarity procedure, and
weather-series reconstruction as notebooks/granger_causality.ipynb.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DATA_PATH = ROOT / "data" / "colombo_tea_auction_dataset.csv"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LAG = 8
LEGACY_MAX_LAG = 3

CATALOGUES = ["high_grown", "low_grown", "off_grade", "dust"]

DISPLAY = {
    "high_grown": "High Grown",
    "low_grown": "Low Grown",
    "off_grade": "Off-Grade",
    "dust": "Dust",
}

WEATHER_VARS = {
    "Precipitation": "precip_current",
    "Temperature": "temperature_current",
    "Sunshine": "sunshine_current",
}


def adf_result(series, alpha=0.05):
    """Return stationarity decision and ADF p-value."""
    clean = pd.Series(series).dropna()

    if len(clean) < 12 or clean.nunique() < 2:
        return False, np.nan

    try:
        _, p_value, *_ = adfuller(clean, autolag="AIC")
        return p_value < alpha, float(p_value)
    except Exception:
        return False, np.nan


def run_granger(price_series, weather_series, max_lag):
    """Run weather -> price Granger tests separately for each maximum lag."""
    data = pd.concat(
        [
            pd.Series(price_series, name="price"),
            pd.Series(weather_series, name="weather"),
        ],
        axis=1,
    ).dropna()

    results = {}

    if len(data) < max_lag + 12:
        for lag in range(1, max_lag + 1):
            results[lag] = (
                np.nan,
                np.nan,
                len(data),
                "insufficient observations",
            )
        return results

    for lag in range(1, max_lag + 1):
        try:
            test_result = grangercausalitytests(
                data[["price", "weather"]],
                maxlag=lag,
                verbose=False,
            )

            f_stat, p_value, _, _ = test_result[lag][0]["ssr_ftest"]

            results[lag] = (
                float(f_stat),
                float(p_value),
                len(data),
                "",
            )

        except Exception as exc:
            results[lag] = (
                np.nan,
                np.nan,
                len(data),
                f"{type(exc).__name__}: {exc}",
            )

    return results


def get_best_result(block):
    """Return the row with the smallest available raw p-value."""
    valid = block.dropna(subset=["p_value"])

    if valid.empty:
        return None

    return valid.loc[valid["p_value"].idxmin()]


def main():
    print("=" * 82)
    print(" LONG-LAG GRANGER SENSITIVITY: 1-8 WEEKS")
    print("=" * 82)

    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["price_mid_lkr", "catalogue", "sale_year", "sale_number"]
    ).copy()

    required_columns = [
        "temperature_2m_mean_mean",
        "precipitation_sum_total_lag1",
        "sunshine_duration_total_lag1",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Stable sort preserves the original within-sale row order.
    # This reproduces the original notebook's use of groupby(...).first().
    df = df.sort_values(
        ["sale_year", "sale_number"],
        kind="stable",
    ).reset_index(drop=True)

    # Same sale x catalogue aggregation as the original Granger notebook.
    sale_ts_all = (
        df.groupby(["catalogue", "sale_year", "sale_number"], as_index=False)
        .agg(
            price_mid_lkr=("price_mid_lkr", "mean"),
            temperature_current=("temperature_2m_mean_mean", "first"),
            precipitation_lag1=("precipitation_sum_total_lag1", "first"),
            sunshine_lag1=("sunshine_duration_total_lag1", "first"),
        )
        .sort_values(
            ["catalogue", "sale_year", "sale_number"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # Same reconstruction used by the original notebook:
    # lag1[t+1] equals the immediately preceding weather week for sale t.
    for catalogue in CATALOGUES:
        mask = sale_ts_all["catalogue"] == catalogue

        sale_ts_all.loc[mask, "precip_current"] = (
            sale_ts_all.loc[mask, "precipitation_lag1"]
            .shift(-1)
            .to_numpy()
        )

        sale_ts_all.loc[mask, "sunshine_current"] = (
            sale_ts_all.loc[mask, "sunshine_lag1"]
            .shift(-1)
            .to_numpy()
        )

    catalogue_ts = {
        catalogue: sale_ts_all[
            sale_ts_all["catalogue"] == catalogue
        ].reset_index(drop=True)
        for catalogue in CATALOGUES
    }

    print("\nSale-level series:")
    for catalogue in CATALOGUES:
        print(
            f"  {DISPLAY[catalogue]:<12} "
            f"{len(catalogue_ts[catalogue]):>3} auction weeks"
        )

    adf_rows = []
    decisions = {}

    print("\nADF stationarity decisions:")
    for catalogue in CATALOGUES:
        ts = catalogue_ts[catalogue]
        decisions[catalogue] = {}

        price_stationary, price_p = adf_result(ts["price_mid_lkr"])
        decisions[catalogue]["price_diff"] = not price_stationary

        adf_rows.append(
            {
                "catalogue": DISPLAY[catalogue],
                "series": "Price",
                "adf_p_value": price_p,
                "stationary_at_level": price_stationary,
                "action": "levels" if price_stationary else "first_difference",
            }
        )

        print(
            f"  {DISPLAY[catalogue]:<12} Price: "
            f"{'levels' if price_stationary else 'first difference'}"
        )

        for weather_name, weather_column in WEATHER_VARS.items():
            stationary, p_value = adf_result(ts[weather_column])

            decisions[catalogue][f"{weather_name}_diff"] = not stationary

            adf_rows.append(
                {
                    "catalogue": DISPLAY[catalogue],
                    "series": weather_name,
                    "adf_p_value": p_value,
                    "stationary_at_level": stationary,
                    "action": "levels" if stationary else "first_difference",
                }
            )

    records = []

    for catalogue in CATALOGUES:
        ts = catalogue_ts[catalogue]
        decs = decisions[catalogue]

        price_series = (
            ts["price_mid_lkr"].diff()
            if decs["price_diff"]
            else ts["price_mid_lkr"]
        )

        for weather_name, weather_column in WEATHER_VARS.items():
            weather_series = ts[weather_column].copy()

            if decs[f"{weather_name}_diff"]:
                weather_series = weather_series.diff()

            lag_results = run_granger(
                price_series=price_series,
                weather_series=weather_series,
                max_lag=MAX_LAG,
            )

            for lag, (f_stat, p_value, n_obs, error_reason) in lag_results.items():
                records.append(
                    {
                        "catalogue": catalogue,
                        "catalogue_display": DISPLAY[catalogue],
                        "weather_variable": weather_name,
                        "lag_weeks": lag,
                        "F_statistic": f_stat,
                        "p_value": p_value,
                        "n_observations": n_obs,
                        "price_first_differenced": decs["price_diff"],
                        "weather_first_differenced": decs[
                            f"{weather_name}_diff"
                        ],
                        "error_reason": error_reason if error_reason else np.nan,
                    }
                )

    results_df = pd.DataFrame(records)

    # Benjamini-Hochberg correction across the full 4 x 3 x 8 = 96-test family.
    results_df["p_value_fdr_bh"] = np.nan
    valid = results_df["p_value"].notna()

    if valid.any():
        _, adjusted_p, _, _ = multipletests(
            results_df.loc[valid, "p_value"],
            alpha=0.05,
            method="fdr_bh",
        )
        results_df.loc[valid, "p_value_fdr_bh"] = adjusted_p

    results_df["nominal_p_lt_0_05"] = results_df["p_value"] < 0.05
    results_df["fdr_significant_0_05"] = (
        results_df["p_value_fdr_bh"] < 0.05
    )

    summary_rows = []

    for catalogue in CATALOGUES:
        for weather_name in WEATHER_VARS:
            subset = results_df[
                (results_df["catalogue"] == catalogue)
                & (results_df["weather_variable"] == weather_name)
            ].copy()

            early = subset[subset["lag_weeks"] <= LEGACY_MAX_LAG]
            extended = subset[subset["lag_weeks"] > LEGACY_MAX_LAG]

            early_best = get_best_result(early)
            extended_best = get_best_result(extended)

            summary_rows.append(
                {
                    "Catalogue": DISPLAY[catalogue],
                    "Weather_variable": weather_name,
                    "Best_lag_1_to_3": (
                        int(early_best["lag_weeks"])
                        if early_best is not None
                        else np.nan
                    ),
                    "Best_raw_p_1_to_3": (
                        early_best["p_value"]
                        if early_best is not None
                        else np.nan
                    ),
                    "Best_F_1_to_3": (
                        early_best["F_statistic"]
                        if early_best is not None
                        else np.nan
                    ),
                    "FDR_p_1_to_3": (
                        early_best["p_value_fdr_bh"]
                        if early_best is not None
                        else np.nan
                    ),
                    "Best_lag_4_to_8": (
                        int(extended_best["lag_weeks"])
                        if extended_best is not None
                        else np.nan
                    ),
                    "Best_raw_p_4_to_8": (
                        extended_best["p_value"]
                        if extended_best is not None
                        else np.nan
                    ),
                    "Best_F_4_to_8": (
                        extended_best["F_statistic"]
                        if extended_best is not None
                        else np.nan
                    ),
                    "FDR_p_4_to_8": (
                        extended_best["p_value_fdr_bh"]
                        if extended_best is not None
                        else np.nan
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    results_df.to_csv(
        OUTPUT_DIR / "long_lag_granger_all_tests.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR / "long_lag_granger_summary.csv",
        index=False,
    )

    pd.DataFrame(adf_rows).to_csv(
        OUTPUT_DIR / "long_lag_granger_adf.csv",
        index=False,
    )

    print("\n" + "=" * 82)
    print(" RESULTS")
    print("=" * 82)
    print(f"Total tests: {len(results_df)}")
    print(f"Nominal p < 0.05: {int(results_df['nominal_p_lt_0_05'].sum())}")
    print(
        "FDR-significant p < 0.05: "
        f"{int(results_df['fdr_significant_0_05'].sum())}"
    )

    print("\nBest raw p-values: lags 1-3 versus lags 4-8")
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved:")
    print(OUTPUT_DIR / "long_lag_granger_all_tests.csv")
    print(OUTPUT_DIR / "long_lag_granger_summary.csv")
    print(OUTPUT_DIR / "long_lag_granger_adf.csv")


if __name__ == "__main__":
    main()