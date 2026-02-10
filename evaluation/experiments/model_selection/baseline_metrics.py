"""
Baselines:
- last_value: predict Customers
- seasonal_naive: horizon-dependent lag

Evaluation:
- Train/eval windows from data/splits/time_splits_rolling.json
- Metrics MAE/RMSE/WAPE computed on open_future==1
- Bias computed on all rows
- Macro metrics computed per-store then averaged

Output schema:
model,horizon,split,fold,agg,MAE,RMSE,WAPE,Bias,N,training_time_seconds,stores,WAPE_p90

"""

import os
import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from training_helper import (
    add_features_per_store,
    filter_issue_window,
    compute_micro,
    compute_macro,
)

DATA_PATH = "data/processed/panel_train_clean.csv"
HOLDOUT_PATH = "data/splits/holdout_stores.csv"
SPLITS_PATH = "data/splits/time_splits_rolling.json"

OUT_RESULTS_DIR = "experiments/model_selection/results"
os.makedirs(OUT_RESULTS_DIR, exist_ok=True)

HORIZONS = [1, 7, 14]
TARGET = "Customers"


def build_targets(df: pd.DataFrame, h: int, target_col: str = TARGET) -> pd.DataFrame:
    # Add horizon target and open_future flag to data
    d = df.copy()
    d[f"y_tplus{h}"] = d.groupby("Store")[target_col].shift(-h)
    d[f"open_future_{h}"] = d.groupby("Store")["Open"].shift(-h)
    return d


def seasonal_reference_shift(h: int) -> int:
    # Lag used for the seasonal naive baseline
    if h == 1:
        return 6
    if h == 7:
        return 0
    if h == 14:
        return 7
    return 7


def evaluate_split(d: pd.DataFrame, h: int, split_name: str, fold: str, start: pd.Timestamp, end: pd.Timestamp) -> List[Dict]:
    # Evaluate both baselines on a given stary and end window
    out_rows: List[Dict] = []
    d_win = filter_issue_window(d, start, end).copy()

    y_col = f"y_tplus{h}"
    open_col = f"open_future_{h}"
    d_win = d_win.dropna(subset=[y_col]).copy()

    # last_value baseline: Customers
    for model_name, yhat_open in [
        ("last_value", d_win[TARGET].to_numpy(dtype=float)),
        ("seasonal_naive", d_win.groupby("Store")[TARGET].shift(seasonal_reference_shift(h)).to_numpy(dtype=float)),
    ]:
        # default yhat to 0 for closed days
        eval_df = pd.DataFrame({
            "Store": d_win["Store"].to_numpy(),
            "y": d_win[y_col].to_numpy(dtype=float),
            "open_future": d_win[open_col].to_numpy(dtype=float),
        })
        eval_df["yhat"] = 0.0
        open_mask = eval_df["open_future"] == 1

        # guard against NaNs in baseline
        yhat_open = np.nan_to_num(yhat_open, nan=0.0)
        eval_df.loc[open_mask, "yhat"] = yhat_open[open_mask.to_numpy()]

        micro = compute_micro(eval_df)
        macro = compute_macro(eval_df)

        out_rows.append({
            "model": model_name,
            "horizon": h,
            "split": split_name,
            "fold": fold,
            "agg": "micro",
            "MAE": micro["MAE"],
            "RMSE": micro["RMSE"],
            "WAPE": micro["WAPE"],
            "Bias": micro["Bias"],
            "N": micro["N"],
            "training_time_seconds": 0.0,
        })
        out_rows.append({
            "model": model_name,
            "horizon": h,
            "split": split_name,
            "fold": fold,
            "agg": "macro",
            "MAE": macro["MAE"],
            "RMSE": macro["RMSE"],
            "WAPE": macro["WAPE"],
            "Bias": macro["Bias"],
            "N": int(len(eval_df)),
            "training_time_seconds": 0.0,
            "stores": macro["stores"],
            "WAPE_p90": macro["WAPE_p90"],
        })

    return out_rows


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

    results: List[Dict] = []
    val_folds = splits.get("val_folds", [])
    train_global_start = pd.to_datetime(splits["train_global"]["start"])
    train_global_end = pd.to_datetime(splits["train_global"]["end"])
    test_start = pd.to_datetime(splits["test"]["start"])
    test_end = pd.to_datetime(splits["test"]["end"])

    for h in HORIZONS:
        d = build_targets(dev, h, target_col=TARGET)

        # val folds
        for fold_idx, fold in enumerate(val_folds, start=1):
            val_start = pd.to_datetime(fold["val"]["start"])
            val_end = pd.to_datetime(fold["val"]["end"])
            results.extend(evaluate_split(d, h, "val", str(fold_idx), val_start, val_end))

        # fixed test window
        results.extend(evaluate_split(d, h, "test", "", test_start, test_end))

    out = pd.DataFrame(results).sort_values(["model", "split", "fold", "horizon", "agg"]).reset_index(drop=True)
    out_path = os.path.join(OUT_RESULTS_DIR, "baseline_metrics.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # Test mean macro WAPE across horizons
    test_macro_df = out[(out["split"] == "test") & (out["agg"] == "macro")]
    if not test_macro_df.empty:
        models = sorted(test_macro_df["model"].unique().tolist())
        for m in models:
            m_df = test_macro_df[test_macro_df["model"] == m]
            test_mean = float(m_df["WAPE"].mean())
            print(f"Test mean macro WAPE across horizons ({m}): {test_mean:.4f}%")

    # Validation mean macro WAPE across folds
    val_macro_df = out[(out["split"] == "val") & (out["agg"] == "macro")]
    if not val_macro_df.empty:
        models = sorted(val_macro_df["model"].unique().tolist())
        for m in models:
            m_df = val_macro_df[val_macro_df["model"] == m].copy()
            grouped = m_df.groupby(["fold", "horizon"])["WAPE"].mean()
            if not grouped.empty:
                val_mean = float(grouped.mean())
                print(f"Validation mean macro WAPE across folds ({m}): {val_mean:.4f}%")


if __name__ == "__main__":
    main()
