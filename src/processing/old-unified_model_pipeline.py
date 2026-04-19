from pathlib import Path

import pandas as pd

from modeling_common import (
    SEED,
    TARGET,
    build_model_registry,
    get_segment_data,
    load_preprocessed_df,
    resolve_project_root,
    run_timeseries_cv,
)


def run_unified_pipeline(k_splits=5):
    root = resolve_project_root(Path(__file__).resolve())
    df, data_path = load_preprocessed_df(root, keep_target_only=True)

    segment_data = get_segment_data(df, target=TARGET)
    models = build_model_registry(seed=SEED)

    print(f"Loaded: {data_path}")
    print(f"Rows: {len(df)}")

    all_folds = []
    for seg_name, (sdf, feature_cols) in segment_data.items():
        print(f"\nRunning segment: {seg_name} | rows={len(sdf)} | features={len(feature_cols)}")

        if len(sdf) < 20:
            print(f"  Skipped {seg_name}: not enough rows for stable {k_splits}-fold CV.")
            continue

        for model_name, model_obj in models.items():
            try:
                fold_df = run_timeseries_cv(
                    sdf=sdf,
                    feature_cols=feature_cols,
                    model_name=model_name,
                    model_obj=model_obj,
                    target=TARGET,
                    k=k_splits,
                )
                fold_df.insert(0, "Segment", seg_name)
                all_folds.append(fold_df)
                print(f"  Done: {model_name}")
            except Exception as ex:
                print(f"  Failed: {model_name} -> {ex}")

    results_folds = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()

    if results_folds.empty:
        print("No CV results generated.")
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        results_folds
        .groupby(["Segment", "Model"], as_index=False)
        .agg(
            RMSE=("RMSE", "mean"),
            MAE=("MAE", "mean"),
            MAPE=("MAPE", "mean"),
            R2=("R2", "mean"),
        )
        .sort_values(["Segment", "RMSE"])
        .reset_index(drop=True)
    )

    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "unified_model_cv_results.csv"
    summary.to_csv(out_path, index=False)

    print(f"\nSaved summary to: {out_path}")
    return results_folds, summary


if __name__ == "__main__":
    run_unified_pipeline(k_splits=5)
