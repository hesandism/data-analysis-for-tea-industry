import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from modeling_common import (
    SEED,
    TARGET,
    build_model_registry,
    evaluate_estimator_timeseries,
    get_param_grids,
    get_segment_data,
    load_preprocessed_df,
    resolve_project_root,
    resolve_unified_summary_path,
)


LEAK_COLS = ["price_lo_lkr", "price_hi_lkr", "price_range_lkr", "tier"]


def run_hyperparameter_tuning(k_splits=5, top_k=3):
    root = resolve_project_root(Path(__file__).resolve())
    df, preprocessed_path = load_preprocessed_df(root, keep_target_only=True)
    baseline_path = resolve_unified_summary_path(root)
    baseline_summary = pd.read_csv(baseline_path)

    print(f"Loaded preprocessed data: {preprocessed_path}")
    print(f"Loaded baseline summary: {baseline_path}")
    print(f"Data shape: {df.shape}")

    segment_data = get_segment_data(df, target=TARGET)
    for seg, (sdf, feature_cols) in segment_data.items():
        print(f"{seg:<10} rows={len(sdf):4d} features={len(feature_cols):3d}")

    # Leakage audit for critical columns.
    audit_rows = []
    for seg, (sdf, feature_cols) in segment_data.items():
        feature_set = set(feature_cols)
        leaked_cols = [c for c in LEAK_COLS if c in feature_set]
        audit_rows.append(
            {
                "Segment": seg,
                "leak_cols_in_training_features": ", ".join(leaked_cols) if leaked_cols else "NONE",
                "leakage_pass": len(leaked_cols) == 0,
            }
        )
    leakage_audit_df = pd.DataFrame(audit_rows)
    print("\nLeakage audit:")
    print(leakage_audit_df.to_string(index=False))

    top_models = (
        baseline_summary.groupby("Model", as_index=False)["RMSE"]
        .mean()
        .sort_values("RMSE")
        .head(top_k)["Model"]
        .tolist()
    )
    print(f"\nTop-{top_k} models from baseline:", top_models)

    base_models = build_model_registry(seed=SEED)
    param_grids = get_param_grids()
    inner_cv = TimeSeriesSplit(n_splits=k_splits)

    all_grid_logs = []
    best_rows = []

    for seg_name, (sdf, feature_cols) in segment_data.items():
        print(f"\nTuning segment: {seg_name} | rows={len(sdf)} | features={len(feature_cols)}")
        if len(sdf) < 20:
            print(f"  Skipped {seg_name}: not enough rows for stable {k_splits}-fold tuning.")
            continue

        X = sdf[feature_cols].copy()
        y = sdf[TARGET].copy()
        segment_best = None

        for model_name in top_models:
            if model_name not in param_grids:
                print(f"  Skipped {model_name}: no grid defined.")
                continue

            gscv = GridSearchCV(
                estimator=base_models[model_name],
                param_grid=param_grids[model_name],
                scoring="neg_root_mean_squared_error",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
                verbose=0,
                return_train_score=False,
            )
            gscv.fit(X, y)

            grid_log = pd.DataFrame(gscv.cv_results_)[
                ["params", "mean_test_score", "std_test_score", "rank_test_score"]
            ].copy()
            grid_log["Segment"] = seg_name
            grid_log["Model"] = model_name
            grid_log["mean_test_RMSE_log"] = -grid_log["mean_test_score"]
            all_grid_logs.append(grid_log)

            fold_eval = evaluate_estimator_timeseries(
                sdf=sdf,
                feature_cols=feature_cols,
                estimator=gscv.best_estimator_,
                target=TARGET,
                k=k_splits,
            )

            row = {
                "Segment": seg_name,
                "Model": model_name,
                "RMSE": fold_eval["RMSE"].mean(),
                "MAE": fold_eval["MAE"].mean(),
                "MAPE": fold_eval["MAPE"].mean(),
                "R2": fold_eval["R2"].mean(),
                "Selected_Hyperparameters": json.dumps(gscv.best_params_, sort_keys=True),
            }

            if segment_best is None or row["RMSE"] < segment_best["RMSE"]:
                segment_best = row

            print(f"  Done: {model_name} | best RMSE={row['RMSE']:.2f}")

        if segment_best is not None:
            best_rows.append(segment_best)

    grid_logs_df = pd.concat(all_grid_logs, ignore_index=True) if all_grid_logs else pd.DataFrame()
    tuned_best_df = pd.DataFrame(best_rows)

    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    logs_path = out_dir / "hyperparam_gridsearch_all_results.csv"
    compact_path = out_dir / "hyperparam_best_per_segment_compact.csv"

    if not grid_logs_df.empty:
        grid_logs_df.to_csv(logs_path, index=False)
    if not tuned_best_df.empty:
        compact_table = (
            tuned_best_df[["Segment", "Model", "RMSE", "MAE", "Selected_Hyperparameters"]]
            .sort_values("RMSE")
            .reset_index(drop=True)
        )
        compact_table.to_csv(compact_path, index=False)
    else:
        compact_table = pd.DataFrame()

    print(f"\nAll grid-search logs shape: {grid_logs_df.shape}")
    print(f"Best-per-segment table shape: {tuned_best_df.shape}")
    print(f"Saved full tuning log to: {logs_path}")
    print(f"Saved compact paper table to: {compact_path}")

    return leakage_audit_df, grid_logs_df, tuned_best_df


if __name__ == "__main__":
    run_hyperparameter_tuning(k_splits=5, top_k=3)
