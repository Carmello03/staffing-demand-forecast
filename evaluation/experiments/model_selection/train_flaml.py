"""
Rolling-origin Flaml training

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
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from flaml import AutoML

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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
)

DATA_PATH = "evaluation/data/processed/panel_train_clean.csv"
HOLDOUT_PATH = "evaluation/data/splits/holdout_stores.csv"
SPLITS_PATH = "evaluation/data/splits/time_splits_purged_kfold.json"

OUT_RESULTS_DIR = "evaluation/experiments/model_selection/results"
OUT_ARTIFACTS_DIR = "evaluation/experiments/model_selection/artifacts"
os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUT_ARTIFACTS_DIR, exist_ok=True)

HORIZONS: List[int] = [1, 7, 14]
TARGET = "Customers"
SEED = 42

# Fixed time budget (in seconds) for each AutoML run
TIME_BUDGET_PER_RUN: int = 3600

TUNE_DAYS: int = 42
GAP_DAYS: int = 28

def split_train_tune_by_time(d_train_all: pd.DataFrame, tune_days: int, gap_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = d_train_all["Date"].max()
    tune_end = max_date
    tune_start = tune_end - pd.Timedelta(days=tune_days - 1)

    inner_train_end = tune_start - pd.Timedelta(days=gap_days + 1)

    d_tune = d_train_all[(d_train_all["Date"] >= tune_start) & (d_train_all["Date"] <= tune_end)].copy()
    d_inner = d_train_all[d_train_all["Date"] <= inner_train_end].copy()

    if d_inner.empty:
        inner_train_end = tune_start - pd.Timedelta(days=1)
        d_inner = d_train_all[d_train_all["Date"] <= inner_train_end].copy()

    if d_tune.empty:
        d_tune = d_train_all.copy()

    return d_inner, d_tune


def train_and_evaluate_fold(
    d: pd.DataFrame,
    h: int,
    fold_info: Dict[str, Any],
    num_cols: List[str],
    cat_cols: List[str],
    time_budget: int,
) -> List[Dict[str, Any]]:
    
    # Train Flaml on one fold and return micro+macro metric rows
    fold_idx = fold_info.get("fold")

    # Slice the training and validation windows
    val_start = pd.to_datetime(fold_info["val"]["start"])
    val_end = pd.to_datetime(fold_info["val"]["end"])
    if "train_ranges" in fold_info:
        d_train = filter_issue_ranges(d, fold_info["train_ranges"])
    else:
        train_start = pd.to_datetime(fold_info["train"]["start"])
        train_end = pd.to_datetime(fold_info["train"]["end"])
        d_train = filter_issue_window(d, train_start, train_end)
    d_val = filter_issue_window(d, val_start, val_end)

    # Select relevant columns and drop rows with missing targets
    needed_cols = ["Date"] + num_cols + cat_cols + ["y", "open_future"]
    d_train = d_train[needed_cols].copy().dropna(subset=["y"])
    d_val = d_val[needed_cols].copy().dropna(subset=["y"])

    d_train = fill_missing_values(d_train, num_cols, cat_cols)
    d_val = fill_missing_values(d_val, num_cols, cat_cols)

    # Train only on open target days
    d_inner, d_tune = split_train_tune_by_time(d_train, TUNE_DAYS, GAP_DAYS)

    d_train_open = d_inner[d_inner["open_future"] == 1].copy()
    d_tune_open = d_tune[d_tune["open_future"] == 1].copy()
    d_val_open = d_val[d_val["open_future"] == 1].copy()

    if len(d_tune_open) == 0:
        d_tune_open = d_train_open.copy()

    # Prepare feature matrices
    X_train = d_train_open[num_cols + cat_cols]
    X_val = d_tune_open[num_cols + cat_cols]
    y_train = d_train_open["y"].to_numpy(dtype=float)
    y_val = d_tune_open["y"].to_numpy(dtype=float)


    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)

    # Configure and fit AutoML
    automl = AutoML()
    start_time = time.perf_counter()
    automl.fit(
        X_train=X_train_t,
        y_train=y_train_log,
        X_val=X_val_t,
        y_val=y_val_log,
        task="regression",
        metric="mae",
        time_budget=time_budget,
        estimator_list="auto",
        seed=SEED,
        verbose=0,
    )
    elapsed = time.perf_counter() - start_time

    # Predict on open rows in validation set
    X_val_eval = d_val_open[num_cols + cat_cols]
    X_val_eval_t = pre.transform(X_val_eval)
    yhat_open_log = automl.predict(X_val_eval_t)
    yhat_open = np.expm1(yhat_open_log)
    yhat_open = np.maximum(0.0, yhat_open)

    val_eval = make_eval_frame_from_open_predictions(d_val, yhat_open)

    micro_metrics = compute_micro(val_eval)
    macro_metrics = compute_macro(val_eval)

    records: List[Dict[str, Any]] = []

    # Micro row 
    records.append({
        "model": "flaml_log",
        "horizon": h,
        "split": "val",
        "fold": fold_idx,
        "agg": "micro",
        "MAE": micro_metrics["MAE"],
        "RMSE": micro_metrics["RMSE"],
        "WAPE": micro_metrics["WAPE"],
        "Bias": micro_metrics["Bias"],
        "N": micro_metrics["N"],
        "training_time_seconds": elapsed,
        "stores": micro_metrics.get("stores", float("nan")),
        "WAPE_p90": micro_metrics.get("WAPE_p90", float("nan")),
    })

    # Macro row 
    records.append({
        "model": "flaml_log",
        "horizon": h,
        "split": "val",
        "fold": fold_idx,
        "agg": "macro",
        "MAE": macro_metrics["MAE"],
        "RMSE": macro_metrics["RMSE"],
        "WAPE": macro_metrics["WAPE"],
        "Bias": macro_metrics["Bias"],
        "N": macro_metrics["N"],
        "training_time_seconds": elapsed,
        "stores": macro_metrics["stores"],
        "WAPE_p90": macro_metrics["WAPE_p90"],
    })
    return records


def train_and_evaluate_final(
    d: pd.DataFrame,
    h: int,
    best_params: Optional[Dict[str, Any]],  # ignored for FLAML
    num_cols: List[str],
    cat_cols: List[str],
    time_budget: int,
    train_global_start: pd.Timestamp,
    train_global_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> List[Dict[str, Any]]:
    # Train AutoGluon on global train window and evaluate on fixed test window
    d_train_global = filter_issue_window(d, train_global_start, train_global_end)
    d_test = filter_issue_window(d, test_start, test_end)
    needed_cols = num_cols + cat_cols + ["y", "open_future"]
    d_train_global = d_train_global[needed_cols].copy().dropna(subset=["y"])
    d_test = d_test[needed_cols].copy().dropna(subset=["y"])
    d_train_global = fill_missing_values(d_train_global, num_cols, cat_cols)
    d_test = fill_missing_values(d_test, num_cols, cat_cols)

    # Train only on open target days
    d_train_open = d_train_global[d_train_global["open_future"] == 1].copy()
    X_train_g = d_train_open[num_cols + cat_cols]
    y_train_g = d_train_open["y"].to_numpy(dtype=float)
    y_train_g_log = np.log1p(y_train_g)

    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    X_train_g_t = pre.fit_transform(X_train_g)

    # Fit AutoML on global training
    automl = AutoML()
    start_time = time.perf_counter()
    automl.fit(
        X_train=X_train_g_t,
        y_train=y_train_g_log,
        task="regression",
        metric="mae",
        time_budget=time_budget,
        estimator_list="auto",
        seed=SEED,
        verbose=0,
    )
    elapsed = time.perf_counter() - start_time

    # Evaluate on test set
    test_eval = d_test[["Store", "y", "open_future"]].copy()
    test_eval["yhat"] = 0.0  # default for closed days
    open_mask_test = test_eval["open_future"] == 1

    # Transform open rows and predict
    if open_mask_test.sum() > 0:
        X_test_open = d_test.loc[open_mask_test, num_cols + cat_cols]
        X_test_open_t = pre.transform(X_test_open)
        yhat_open_log = automl.predict(X_test_open_t)
        yhat_open = np.expm1(yhat_open_log)
        yhat_open = np.maximum(0.0, yhat_open)
        test_eval.loc[open_mask_test, "yhat"] = yhat_open

    micro_test = compute_micro(test_eval)
    macro_test = compute_macro(test_eval)

    records: List[Dict[str, Any]] = []
    records.append({
        "model": "flaml_log",
        "horizon": h,
        "split": "test",
        "fold": "",
        "agg": "micro",
        "MAE": micro_test["MAE"],
        "RMSE": micro_test["RMSE"],
        "WAPE": micro_test["WAPE"],
        "Bias": micro_test["Bias"],
        "N": micro_test["N"],
        "training_time_seconds": elapsed,
        "stores": micro_test.get("stores", float("nan")),
        "WAPE_p90": micro_test.get("WAPE_p90", float("nan")),
    })

    records.append({
        "model": "flaml_log",
        "horizon": h,
        "split": "test",
        "fold": "",
        "agg": "macro",
        "MAE": macro_test["MAE"],
        "RMSE": macro_test["RMSE"],
        "WAPE": macro_test["WAPE"],
        "Bias": macro_test["Bias"],
        "N": macro_test["N"],
        "training_time_seconds": elapsed,
        "stores": macro_test["stores"],
        "WAPE_p90": macro_test["WAPE_p90"],
    })

    # Save model artifact
    model_path = os.path.join(OUT_ARTIFACTS_DIR, f"flaml_h{h}.joblib")
    joblib.dump({"pre": pre, "automl": automl}, model_path)
    return records


def main() -> None:
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    holdout = pd.read_csv(HOLDOUT_PATH)
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    global GAP_DAYS
    GAP_DAYS = int(splits.get('meta', {}).get('GAP_DAYS', GAP_DAYS))

    # Exclude holdout stores from development data
    holdout_stores = set(holdout["Store"].tolist())
    dev = df[~df["Store"].isin(holdout_stores)].copy()
    print("Dev rows:", dev.shape[0], "Dev stores:", dev["Store"].nunique())

    # Feature engineering per store
    dev = dev.sort_values(["Store", "Date"]).reset_index(drop=True)
    dev = dev.groupby("Store", group_keys=False).apply(add_features_per_store)

    # Prepare fold ranges
    val_folds = splits.get("val_folds", [])
    train_global_start = pd.to_datetime(splits["train_global"]["start"])
    train_global_end = pd.to_datetime(splits["train_global"]["end"])
    test_start = pd.to_datetime(splits["test"]["start"])
    test_end = pd.to_datetime(splits["test"]["end"])
    results: List[Dict[str, Any]] = []

    # Iterate over horizons
    for h in HORIZONS:
        print(f"\nTraining horizon: {h}")
        d = build_target_cols(dev, h, target_col=TARGET)

        # Evaluate each validation fold
        fold_wape_scores: List[float] = []
        for fold_info in val_folds:
            # Train and evaluate on this fold
            fold_records = train_and_evaluate_fold(
                d=d,
                h=h,
                fold_info=fold_info,
                num_cols=NUM_COLS,
                cat_cols=CAT_COLS,
                time_budget=TIME_BUDGET_PER_RUN,
            )
            # Append results and collect WAPE for summary
            for rec in fold_records:
                results.append(rec)
                if rec["agg"] == "macro" and rec["split"] == "val":
                    fold_wape_scores.append(rec["WAPE"])

        if fold_wape_scores:
            cv_mean_macro_wape = float(np.mean(fold_wape_scores))
            print(f"Mean val macro WAPE for h={h}: {cv_mean_macro_wape:.4f}%")
        else:
            print(f"No validation folds provided for h={h}")

        # Final training on global train window and evaluation on test
        final_records = train_and_evaluate_final(
            d=d,
            h=h,
            best_params=None,  # FLAML does it automatically
            num_cols=NUM_COLS,
            cat_cols=CAT_COLS,
            time_budget=TIME_BUDGET_PER_RUN,
            train_global_start=train_global_start,
            train_global_end=train_global_end,
            test_start=test_start,
            test_end=test_end,
        )
        for rec in final_records:
            results.append(rec)
        print(f"Saved model: {os.path.join(OUT_ARTIFACTS_DIR, f'flaml_h{h}.joblib')}")

    out = pd.DataFrame(results)
    out = out.sort_values(["split", "fold", "horizon", "agg"]).reset_index(drop=True)
    out_path = os.path.join(OUT_RESULTS_DIR, "flaml_metrics.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # Test mean macro WAPE across horizons
    test_macro_df = out[(out["split"] == "test") & (out["agg"] == "macro")]
    if not test_macro_df.empty:
        test_mean = float(test_macro_df["WAPE"].mean())
        print(f"Test mean macro WAPE across horizons: {test_mean:.4f}%")

    # Validation mean macro WAPE across folds
    val_macro_df = out[(out["split"] == "val") & (out["agg"] == "macro")]
    if not val_macro_df.empty:
        grouped = val_macro_df.groupby(["fold", "horizon"])["WAPE"].mean()
        val_mean = float(grouped.mean()) if not grouped.empty else float("nan")
        print(f"Validation mean macro WAPE across folds: {val_mean:.4f}%")


if __name__ == "__main__":
    main()