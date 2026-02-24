"""
Rolling-origin PyCaret training

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
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from training_helper import (
    NUM_COLS,
    CAT_COLS,
    add_features_per_store,
    build_target_cols,
    filter_issue_window,
    fill_missing_values,
    filter_issue_ranges,
    make_eval_frame_from_open_predictions,
    compute_micro,
    compute_macro,
)

from pycaret.regression import setup, compare_models, predict_model, save_model, finalize_model


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

TIME_BUDGET_PER_RUN: int = 20
TUNE_DAYS: int = 42
GAP_DAYS: int = 28

def clean_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype(str).str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
    return df

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
    fold_idx = fold_info.get("fold", "")

    val_start = pd.to_datetime(fold_info["val"]["start"])
    val_end = pd.to_datetime(fold_info["val"]["end"])

    if "train_ranges" in fold_info:
        d_train = filter_issue_ranges(d, fold_info["train_ranges"])
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

    d_train = clean_categoricals(d_train, cat_cols)
    d_val = clean_categoricals(d_val, cat_cols)

    d_inner, d_tune = split_train_tune_by_time(d_train, TUNE_DAYS, GAP_DAYS)

    d_train_open = d_inner[d_inner["open_future"] == 1].copy()
    d_tune_open = d_tune[d_tune["open_future"] == 1].copy()
    d_val_open = d_val[d_val["open_future"] == 1].copy()

    feature_cols = num_cols + cat_cols

    if (len(d_train_open) > 0) and (len(d_tune_open) > 0):
        inner_df = d_train_open[["Date"] + feature_cols + ["y"]].copy()
        tune_df = d_tune_open[["Date"] + feature_cols + ["y"]].copy()
        df_all = pd.concat([inner_df, tune_df], axis=0, ignore_index=True).sort_values("Date").reset_index(drop=True)
        train_size = float(len(inner_df) / len(df_all)) if len(df_all) else 0.8
        train_size = max(0.5, min(0.95, train_size))
    else:
        df_all = d_train_open[["Date"] + feature_cols + ["y"]].copy().sort_values("Date").reset_index(drop=True)
        train_size = 0.8

    df_all["y_log"] = np.log1p(df_all["y"].to_numpy(dtype=float))
    df_all = df_all.drop(columns=["y", "Date"])

    t0 = time.time()
    setup(
        data=df_all,
        target="y_log",
        session_id=SEED,
        html=False,
        verbose=False,
        n_jobs=-1,
        data_split_shuffle=False,
        train_size=train_size,
    )

    best = compare_models(
        sort="MAE",
        n_select=1,
        cross_validation=False,
        budget_time=time_budget,
        turbo=True,
        errors="ignore",
        verbose=False,
    )
    best = finalize_model(best)
    train_time = float(time.time() - t0)

    if len(d_val_open) > 0:
        X_val_open = d_val_open[feature_cols].copy()
        preds_df = predict_model(best, data=X_val_open)
        yhat_open_log = preds_df["prediction_label"].to_numpy(dtype=float)
        yhat_open = np.maximum(0.0, np.expm1(yhat_open_log))
    else:
        yhat_open = np.array([], dtype=float)

    val_eval = make_eval_frame_from_open_predictions(d_val, yhat_open)
    micro = compute_micro(val_eval)
    macro = compute_macro(val_eval)

    records: List[Dict[str, Any]] = []
    records.append({
        "model": "pycaret_log",
        "horizon": h,
        "split": "val",
        "fold": fold_idx,
        "agg": "micro",
        "MAE": micro["MAE"],
        "RMSE": micro["RMSE"],
        "WAPE": micro["WAPE"],
        "Bias": micro["Bias"],
        "N": micro["N"],
        "training_time_seconds": train_time,
        "stores": micro.get("stores", float("nan")),
        "WAPE_p90": micro.get("WAPE_p90", float("nan")),
    })
    records.append({
        "model": "pycaret_log",
        "horizon": h,
        "split": "val",
        "fold": fold_idx,
        "agg": "macro",
        "MAE": macro["MAE"],
        "RMSE": macro["RMSE"],
        "WAPE": macro["WAPE"],
        "Bias": macro["Bias"],
        "N": macro["N"],
        "training_time_seconds": train_time,
        "stores": macro["stores"],
        "WAPE_p90": macro["WAPE_p90"],
    })
    return records


def train_and_evaluate_final(
    d: pd.DataFrame,
    h: int,
    num_cols: List[str],
    cat_cols: List[str],
    time_budget: int,
    train_global_start: pd.Timestamp,
    train_global_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> List[Dict[str, Any]]:
    d_train_g = filter_issue_window(d, train_global_start, train_global_end)
    d_test = filter_issue_window(d, test_start, test_end)

    needed_cols = ["Date"] + num_cols + cat_cols + ["y", "open_future"]
    d_train_g = d_train_g[needed_cols].copy().dropna(subset=["y"])
    d_test = d_test[needed_cols].copy().dropna(subset=["y"])

    d_train_g = fill_missing_values(d_train_g, num_cols, cat_cols)
    d_test = fill_missing_values(d_test, num_cols, cat_cols)

    d_train_g = clean_categoricals(d_train_g, cat_cols)
    d_test = clean_categoricals(d_test, cat_cols)

    d_inner, d_tune = split_train_tune_by_time(d_train_g, TUNE_DAYS, GAP_DAYS)

    d_train_open = d_inner[d_inner["open_future"] == 1].copy()
    d_tune_open = d_tune[d_tune["open_future"] == 1].copy()
    feature_cols = num_cols + cat_cols

    if (len(d_train_open) > 0) and (len(d_tune_open) > 0):
        inner_df = d_train_open[["Date"] + feature_cols + ["y"]].copy()
        tune_df = d_tune_open[["Date"] + feature_cols + ["y"]].copy()
        df_all = pd.concat([inner_df, tune_df], axis=0, ignore_index=True).sort_values("Date").reset_index(drop=True)
        train_size = float(len(inner_df) / len(df_all)) if len(df_all) else 0.8
        train_size = max(0.5, min(0.95, train_size))
    else:
        df_all = d_train_open[["Date"] + feature_cols + ["y"]].copy().sort_values("Date").reset_index(drop=True)
        train_size = 0.8

    df_all["y_log"] = np.log1p(df_all["y"].to_numpy(dtype=float))
    df_all = df_all.drop(columns=["y", "Date"])

    t0 = time.time()
    setup(
        data=df_all,
        target="y_log",
        session_id=SEED,
        html=False,
        verbose=False,
        n_jobs=-1,
        data_split_shuffle=False,
        train_size=train_size,
    )

    best = compare_models(
        sort="MAE",
        n_select=1,
        cross_validation=False,
        budget_time=time_budget,
        turbo=True,
        errors="ignore",
        verbose=False,
    )
    best = finalize_model(best)
    train_time = float(time.time() - t0)

    model_base = os.path.join(OUT_ARTIFACTS_DIR, f"pycaret_h{h}")
    save_model(best, model_base)

    open_mask_test = (d_test["open_future"] == 1)
    if open_mask_test.sum() > 0:
        X_test_open = d_test.loc[open_mask_test, feature_cols].copy()
        preds_df = predict_model(best, data=X_test_open)
        yhat_open_log = preds_df["prediction_label"].to_numpy(dtype=float)
        yhat_open = np.maximum(0.0, np.expm1(yhat_open_log))
    else:
        yhat_open = np.array([], dtype=float)

    test_eval = make_eval_frame_from_open_predictions(d_test, yhat_open)
    micro = compute_micro(test_eval)
    macro = compute_macro(test_eval)

    records: List[Dict[str, Any]] = []
    records.append({
        "model": "pycaret_log",
        "horizon": h,
        "split": "test",
        "fold": "",
        "agg": "micro",
        "MAE": micro["MAE"],
        "RMSE": micro["RMSE"],
        "WAPE": micro["WAPE"],
        "Bias": micro["Bias"],
        "N": micro["N"],
        "training_time_seconds": train_time,
        "stores": micro.get("stores", float("nan")),
        "WAPE_p90": micro.get("WAPE_p90", float("nan")),
    })
    records.append({
        "model": "pycaret_log",
        "horizon": h,
        "split": "test",
        "fold": "",
        "agg": "macro",
        "MAE": macro["MAE"],
        "RMSE": macro["RMSE"],
        "WAPE": macro["WAPE"],
        "Bias": macro["Bias"],
        "N": macro["N"],
        "training_time_seconds": train_time,
        "stores": macro["stores"],
        "WAPE_p90": macro["WAPE_p90"],
    })
    return records


def main() -> None:
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    holdout = pd.read_csv(HOLDOUT_PATH)
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    global GAP_DAYS
    GAP_DAYS = int(splits.get('meta', {}).get('GAP_DAYS', GAP_DAYS))

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
                time_budget=TIME_BUDGET_PER_RUN,
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
            time_budget=TIME_BUDGET_PER_RUN,
            train_global_start=train_global_start,
            train_global_end=train_global_end,
            test_start=test_start,
            test_end=test_end,
        )
        for rec in final_records:
            results.append(rec)

        print(f"Saved model: {os.path.join(OUT_ARTIFACTS_DIR, f'pycaret_h{h}.pkl')}")

    out = pd.DataFrame(results)
    out = out.sort_values(["split", "fold", "horizon", "agg"]).reset_index(drop=True)
    out_path = os.path.join(OUT_RESULTS_DIR, "pycaret_metrics.csv")
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
