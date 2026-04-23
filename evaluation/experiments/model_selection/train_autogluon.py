"""
Rolling-origin AutoGluon training

- rolling time splits
- train only on open_future == 1
- closed-day prediction rule (force yhat = 0 when open_future == 0)
- micro + macro metrics
- horizons: 1, 7, 14

Output schema:
model,horizon,split,fold,agg,MAE,RMSE,WAPE,Bias,N,training_time_seconds,stores,WAPE_p90
"""

import os
import json
import time
import shutil
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

from training_helper import (
    NUM_COLS,
    CAT_COLS,
    add_features_per_store,
    build_target_cols,
    filter_issue_window,
    filter_issue_ranges,
    fill_missing_values,
    make_eval_frame_from_open_predictions,
    compute_micro,
    compute_macro,
    split_train_tune_by_time,
)

DATA_PATH = "evaluation/data/processed/panel_train_clean.csv"
HOLDOUT_PATH = "evaluation/data/splits/holdout_stores.csv"
SPLITS_PATH = "evaluation/data/splits/time_splits_purged_kfold.json"

OUT_RESULTS_DIR = "evaluation/experiments/model_selection/results/model_metrics"
OUT_ARTIFACTS_DIR = "evaluation/experiments/model_selection/artifacts"
os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUT_ARTIFACTS_DIR, exist_ok=True)

HORIZONS = [1, 7, 14]
TARGET = "Customers"
SEED = 42
GAP_DAYS = 28
TUNE_DAYS = 42

RUN_BOTH_LONG_AND_QUICK = True

# Full vs quick budgets per horizon run (seconds).
LONG_TIME_BUDGET_PER_RUN = 1200
QUICK_TIME_BUDGET_PER_RUN = 300

PRESETS = "medium_quality"
NUM_BAG_FOLDS = 0
NUM_STACK_LEVELS = 0


def get_run_configs() -> List[Dict[str, Any]]:
    long_cfg = {
        "tag": "long",
        "time_budget": LONG_TIME_BUDGET_PER_RUN,
        "metrics_filename": "autogluon_metrics.csv",
        "artifact_prefix": "autogluon",
    }
    quick_cfg = {
        "tag": "quick",
        "time_budget": QUICK_TIME_BUDGET_PER_RUN,
        "metrics_filename": "autogluon_metrics_quick.csv",
        "artifact_prefix": "autogluon_quick",
    }
    if RUN_BOTH_LONG_AND_QUICK:
        return [long_cfg, quick_cfg]
    return [long_cfg]


def train_and_evaluate_fold(
    d: pd.DataFrame,
    h: int,
    fold_info: Dict[str, Any],
    num_cols: List[str],
    cat_cols: List[str],
    time_budget: int,
    gap_days: int,
    tune_days: int,
    run_tag: str,
) -> List[Dict[str, Any]]:
    
    fold_idx = fold_info.get("fold", "")
    val_start = pd.to_datetime(fold_info["val"]["start"])
    val_end = pd.to_datetime(fold_info["val"]["end"])

    train_ranges = fold_info.get("train_ranges", None)
    if train_ranges is not None:
        d_train = filter_issue_ranges(d, train_ranges)
    else:
        train_start = pd.to_datetime(fold_info["train"]["start"])
        train_end = pd.to_datetime(fold_info["train"]["end"])
        d_train = filter_issue_window(d, train_start, train_end)

    d_val = filter_issue_window(d, val_start, val_end)

    needed_cols = ["Date"] + num_cols + cat_cols + ["y", "open_future"]
    d_train = d_train[needed_cols].copy().dropna(subset=["y"])
    d_val = d_val[needed_cols].copy().dropna(subset=["y"])

    d_train = fill_missing_values(d_train, num_cols, cat_cols)
    d_val = fill_missing_values(d_val, num_cols, cat_cols)

    d_inner, d_tune = split_train_tune_by_time(d_train, tune_days, gap_days)

    d_train_open = d_inner[d_inner["open_future"] == 1].copy()
    d_tune_open = d_tune[d_tune["open_future"] == 1].copy()
    d_val_open = d_val[d_val["open_future"] == 1].copy()

    feature_cols = num_cols + cat_cols

    train_df = d_train_open[feature_cols + ["y"]].copy()
    train_df["y_log"] = np.log1p(train_df["y"].to_numpy(dtype=float))
    train_df = train_df.drop(columns=["y"])

    tune_df = d_tune_open[feature_cols + ["y"]].copy()
    tune_df["y_log"] = np.log1p(tune_df["y"].to_numpy(dtype=float))
    tune_df = tune_df.drop(columns=["y"])

    fold_path = os.path.join(OUT_ARTIFACTS_DIR, f"_autogluon_tmp_{run_tag}_h{h}_fold{fold_idx}")
    shutil.rmtree(fold_path, ignore_errors=True)

    predictor = TabularPredictor(
        label="y_log",
        problem_type="regression",
        eval_metric="mae",
        path=fold_path,
        learner_kwargs={"random_state": SEED},
    )

    t0 = time.perf_counter()
    predictor.fit(
        train_data=train_df,
        tuning_data=tune_df,
        time_limit=time_budget,
        presets=PRESETS,
        num_bag_folds=NUM_BAG_FOLDS,
        num_stack_levels=NUM_STACK_LEVELS,
        excluded_model_types=["NN_TORCH", "FASTAI"],
        verbosity=0,
    )
    train_time = float(time.perf_counter() - t0)

    X_train_open = d_train_open[feature_cols]
    yhat_train_log = predictor.predict(X_train_open, as_pandas=False)
    yhat_train = np.expm1(np.asarray(yhat_train_log, dtype=float))
    yhat_train = np.maximum(0.0, yhat_train)
    train_eval = make_eval_frame_from_open_predictions(d_inner, yhat_train)
    train_micro_metrics = compute_micro(train_eval)
    train_macro_metrics = compute_macro(train_eval)

    X_val_open = d_val_open[feature_cols]
    yhat_open_log = predictor.predict(X_val_open, as_pandas=False)
    yhat_open = np.expm1(np.asarray(yhat_open_log, dtype=float))
    yhat_open = np.maximum(0.0, yhat_open)

    val_eval = make_eval_frame_from_open_predictions(d_val, yhat_open)
    micro_metrics = compute_micro(val_eval)
    macro_metrics = compute_macro(val_eval)

    shutil.rmtree(fold_path, ignore_errors=True)

    records: List[Dict[str, Any]] = []
    records.append({
        "model": "autogluon_log",
        "horizon": h,
        "split": "train",
        "fold": fold_idx,
        "agg": "micro",
        "MAE": train_micro_metrics["MAE"],
        "RMSE": train_micro_metrics["RMSE"],
        "WAPE": train_micro_metrics["WAPE"],
        "Bias": train_micro_metrics["Bias"],
        "N": train_micro_metrics["N"],
        "training_time_seconds": train_time,
        "stores": train_micro_metrics.get("stores", float("nan")),
        "WAPE_p90": train_micro_metrics.get("WAPE_p90", float("nan")),
    })
    records.append({
        "model": "autogluon_log",
        "horizon": h,
        "split": "train",
        "fold": fold_idx,
        "agg": "macro",
        "MAE": train_macro_metrics["MAE"],
        "RMSE": train_macro_metrics["RMSE"],
        "WAPE": train_macro_metrics["WAPE"],
        "Bias": train_macro_metrics["Bias"],
        "N": train_macro_metrics["N"],
        "training_time_seconds": train_time,
        "stores": train_macro_metrics["stores"],
        "WAPE_p90": train_macro_metrics["WAPE_p90"],
    })
    records.append({
        "model": "autogluon_log",
        "horizon": h,
        "split": "val",
        "fold": fold_idx,
        "agg": "micro",
        "MAE": micro_metrics["MAE"],
        "RMSE": micro_metrics["RMSE"],
        "WAPE": micro_metrics["WAPE"],
        "Bias": micro_metrics["Bias"],
        "N": micro_metrics["N"],
        "training_time_seconds": train_time,
        "stores": micro_metrics.get("stores", float("nan")),
        "WAPE_p90": micro_metrics.get("WAPE_p90", float("nan")),
    })
    records.append({
        "model": "autogluon_log",
        "horizon": h,
        "split": "val",
        "fold": fold_idx,
        "agg": "macro",
        "MAE": macro_metrics["MAE"],
        "RMSE": macro_metrics["RMSE"],
        "WAPE": macro_metrics["WAPE"],
        "Bias": macro_metrics["Bias"],
        "N": macro_metrics["N"],
        "training_time_seconds": train_time,
        "stores": macro_metrics["stores"],
        "WAPE_p90": macro_metrics["WAPE_p90"],
    })
    return records


def train_and_evaluate_final(
    d: pd.DataFrame,
    h: int,
    num_cols: List[str],
    cat_cols: List[str],
    time_budget: int,
    artifact_prefix: str,
    train_global_start: pd.Timestamp,
    train_global_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> List[Dict[str, Any]]:
    
    d_train_global = filter_issue_window(d, train_global_start, train_global_end)
    d_test = filter_issue_window(d, test_start, test_end)

    needed_cols = ["Date"] + num_cols + cat_cols + ["y", "open_future"]
    d_train_global = d_train_global[needed_cols].copy().dropna(subset=["y"])
    d_test = d_test[needed_cols].copy().dropna(subset=["y"])

    d_train_global = fill_missing_values(d_train_global, num_cols, cat_cols)
    d_test = fill_missing_values(d_test, num_cols, cat_cols)

    d_train_open = d_train_global[d_train_global["open_future"] == 1].copy()

    feature_cols = num_cols + cat_cols

    train_df = d_train_open[feature_cols + ["y"]].copy()
    train_df["y_log"] = np.log1p(train_df["y"].to_numpy(dtype=float))
    train_df = train_df.drop(columns=["y"])

    model_path = os.path.join(OUT_ARTIFACTS_DIR, f"{artifact_prefix}_h{h}")
    shutil.rmtree(model_path, ignore_errors=True)

    predictor = TabularPredictor(
        label="y_log",
        problem_type="regression",
        eval_metric="mae",
        path=model_path,
        learner_kwargs={"random_state": SEED},
    )

    t0 = time.perf_counter()
    predictor.fit(
        train_data=train_df,
        time_limit=time_budget,
        presets=PRESETS,
        num_bag_folds=NUM_BAG_FOLDS,
        num_stack_levels=NUM_STACK_LEVELS,
        excluded_model_types=["NN_TORCH", "FASTAI"],
        verbosity=0,
    )
    train_time = float(time.perf_counter() - t0)

    open_mask_train = (d_train_global["open_future"] == 1)
    X_train_open = d_train_global.loc[open_mask_train, feature_cols]
    yhat_train_log = predictor.predict(X_train_open, as_pandas=False) if len(X_train_open) else np.array([])
    yhat_train = np.expm1(np.asarray(yhat_train_log, dtype=float)) if len(X_train_open) else np.array([])
    yhat_train = np.maximum(0.0, yhat_train) if len(X_train_open) else yhat_train
    train_eval = make_eval_frame_from_open_predictions(d_train_global, yhat_train)
    micro_train = compute_micro(train_eval)
    macro_train = compute_macro(train_eval)

    open_mask_test = (d_test["open_future"] == 1)
    X_test_open = d_test.loc[open_mask_test, feature_cols]
    yhat_open_log = predictor.predict(X_test_open, as_pandas=False) if len(X_test_open) else np.array([])
    yhat_open = np.expm1(np.asarray(yhat_open_log, dtype=float)) if len(X_test_open) else np.array([])
    yhat_open = np.maximum(0.0, yhat_open) if len(X_test_open) else yhat_open

    test_eval = make_eval_frame_from_open_predictions(d_test, yhat_open)
    micro_test = compute_micro(test_eval)
    macro_test = compute_macro(test_eval)

    records: List[Dict[str, Any]] = []
    records.append({
        "model": "autogluon_log",
        "horizon": h,
        "split": "train",
        "fold": "",
        "agg": "micro",
        "MAE": micro_train["MAE"],
        "RMSE": micro_train["RMSE"],
        "WAPE": micro_train["WAPE"],
        "Bias": micro_train["Bias"],
        "N": micro_train["N"],
        "training_time_seconds": train_time,
        "stores": micro_train.get("stores", float("nan")),
        "WAPE_p90": micro_train.get("WAPE_p90", float("nan")),
    })
    records.append({
        "model": "autogluon_log",
        "horizon": h,
        "split": "train",
        "fold": "",
        "agg": "macro",
        "MAE": macro_train["MAE"],
        "RMSE": macro_train["RMSE"],
        "WAPE": macro_train["WAPE"],
        "Bias": macro_train["Bias"],
        "N": macro_train["N"],
        "training_time_seconds": train_time,
        "stores": macro_train["stores"],
        "WAPE_p90": macro_train["WAPE_p90"],
    })
    records.append({
        "model": "autogluon_log",
        "horizon": h,
        "split": "test",
        "fold": "",
        "agg": "micro",
        "MAE": micro_test["MAE"],
        "RMSE": micro_test["RMSE"],
        "WAPE": micro_test["WAPE"],
        "Bias": micro_test["Bias"],
        "N": micro_test["N"],
        "training_time_seconds": train_time,
        "stores": micro_test.get("stores", float("nan")),
        "WAPE_p90": micro_test.get("WAPE_p90", float("nan")),
    })
    records.append({
        "model": "autogluon_log",
        "horizon": h,
        "split": "test",
        "fold": "",
        "agg": "macro",
        "MAE": macro_test["MAE"],
        "RMSE": macro_test["RMSE"],
        "WAPE": macro_test["WAPE"],
        "Bias": macro_test["Bias"],
        "N": macro_test["N"],
        "training_time_seconds": train_time,
        "stores": macro_test["stores"],
        "WAPE_p90": macro_test["WAPE_p90"],
    })
    return records


def main() -> None:
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    holdout = pd.read_csv(HOLDOUT_PATH)
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    holdout_stores = set(holdout["Store"].tolist())
    dev = df[~df["Store"].isin(holdout_stores)].copy()

    print("Dev rows:", dev.shape[0], "Dev stores:", dev["Store"].nunique())

    dev = dev.sort_values(["Store", "Date"]).reset_index(drop=True)
    dev = dev.groupby("Store", group_keys=False).apply(add_features_per_store)

    val_folds = splits.get("val_folds", [])
    train_global_start = pd.to_datetime(splits["train_global"]["start"])
    train_global_end = pd.to_datetime(splits["train_global"]["end"])
    test_start = pd.to_datetime(splits["test"]["start"])
    test_end = pd.to_datetime(splits["test"]["end"])

    gap_days = int(splits.get("meta", {}).get("GAP_DAYS", 28))

    for cfg in get_run_configs():
        run_tag = str(cfg["tag"])
        time_budget = int(cfg["time_budget"])
        metrics_filename = str(cfg["metrics_filename"])
        artifact_prefix = str(cfg["artifact_prefix"])

        print("\n" + "=" * 72)
        print(f"Run profile: {run_tag} (time_budget={time_budget}s)")
        print("=" * 72)

        results: List[Dict[str, Any]] = []

        for h in HORIZONS:
            print(f"\nTraining horizon: {h}")
            d = build_target_cols(dev, h, target_col=TARGET)

            fold_wape_scores: List[float] = []
            for fold_info in val_folds:
                fold_records = train_and_evaluate_fold(
                    d=d,
                    h=h,
                    fold_info=fold_info,
                    num_cols=NUM_COLS,
                    cat_cols=CAT_COLS,
                    time_budget=time_budget,
                    gap_days=gap_days,
                    tune_days=TUNE_DAYS,
                    run_tag=run_tag,
                )
                for rec in fold_records:
                    results.append(rec)
                    if rec["agg"] == "macro" and rec["split"] == "val":
                        fold_wape_scores.append(rec["WAPE"])

            if fold_wape_scores:
                cv_mean_macro_wape = float(np.mean(fold_wape_scores))
                print(f"Mean val macro WAPE for h={h}: {cv_mean_macro_wape:.4f}%")
            else:
                print(f"No validation folds provided for h={h}")

            final_records = train_and_evaluate_final(
                d=d,
                h=h,
                num_cols=NUM_COLS,
                cat_cols=CAT_COLS,
                time_budget=time_budget,
                artifact_prefix=artifact_prefix,
                train_global_start=train_global_start,
                train_global_end=train_global_end,
                test_start=test_start,
                test_end=test_end,
            )
            for rec in final_records:
                results.append(rec)

            print(f"Saved model: {os.path.join(OUT_ARTIFACTS_DIR, f'{artifact_prefix}_h{h}')}")

        out = (
            pd.DataFrame(results)
            .sort_values(["split", "agg", "horizon", "fold"])
            .reset_index(drop=True)
        )
        out_path = os.path.join(OUT_RESULTS_DIR, metrics_filename)
        out.to_csv(out_path, index=False)
        print("Saved:", out_path)

        test_macro_df = out[(out["split"] == "test") & (out["agg"] == "macro")]
        if not test_macro_df.empty:
            test_mean = float(test_macro_df["WAPE"].mean())
            print(f"Test mean macro WAPE across horizons: {test_mean:.4f}%")

        val_macro_df = out[(out["split"] == "val") & (out["agg"] == "macro")]
        if not val_macro_df.empty:
            grouped = val_macro_df.groupby(["fold", "horizon"])["WAPE"].mean()
            val_mean = float(grouped.mean()) if not grouped.empty else float("nan")
            print(f"Validation mean macro WAPE across folds: {val_mean:.4f}%")


if __name__ == "__main__":
    main()
