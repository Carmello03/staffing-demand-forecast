"""
Seed Variance Analysis
======================
Re-trains LightGBM and/or AutoGluon on the global train window across multiple
random seeds to check how stable the test macro WAPE is.

Edit MODELS_TO_RUN and SEEDS at the bottom of this file to configure what runs.
"""

import json
import os
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from training_helper import (
    NUM_COLS, CAT_COLS,
    add_features_per_store, build_target_cols,
    fill_missing_values, filter_issue_window,
    make_eval_frame_from_open_predictions, compute_macro,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH     = "evaluation/data/processed/panel_train_clean.csv"
HOLDOUT_PATH  = "evaluation/data/splits/holdout_stores.csv"
SPLITS_PATH   = "evaluation/data/splits/time_splits_purged_kfold.json"
ARTIFACTS_DIR = "evaluation/experiments/model_selection/artifacts"
OUT_DIR       = "evaluation/experiments/model_selection/results/seed_variance"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [1, 7, 14]
TARGET   = "Customers"

# ── Hardcoded best params from the main LightGBM training run) ────────────────
LIGHTGBM_PARAMS = {
    1:  {"num_leaves": 63, "learning_rate": 0.10},
    7:  {"num_leaves": 31, "learning_rate": 0.05},
    14: {"num_leaves": 31, "learning_rate": 0.05},
}

# AutoGluon time budget per horizon per seed (seconds)
AUTOGLUON_TIME_BUDGET = 1200


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_dev_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    holdout = pd.read_csv(HOLDOUT_PATH)
    holdout_stores = set(holdout["Store"].tolist())
    dev = df[~df["Store"].isin(holdout_stores)].copy()
    dev = dev.sort_values(["Store", "Date"]).reset_index(drop=True)
    dev = dev.groupby("Store", group_keys=False).apply(add_features_per_store)
    print(f"Dev rows: {dev.shape[0]}, stores: {dev['Store'].nunique()}")
    return dev


def load_splits():
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    return (
        pd.to_datetime(splits["train_global"]["start"]),
        pd.to_datetime(splits["train_global"]["end"]),
        pd.to_datetime(splits["test"]["start"]),
        pd.to_datetime(splits["test"]["end"]),
    )


def get_train_test(dev, h, train_start, train_end, test_start, test_end):
    needed = ["Date"] + NUM_COLS + CAT_COLS + ["y", "open_future"]
    d = build_target_cols(dev, h, target_col=TARGET)
    d_train = filter_issue_window(d, train_start, train_end)[needed].copy().dropna(subset=["y"])
    d_test  = filter_issue_window(d, test_start,  test_end) [needed].copy().dropna(subset=["y"])
    d_train = fill_missing_values(d_train, NUM_COLS, CAT_COLS)
    d_test  = fill_missing_values(d_test,  NUM_COLS, CAT_COLS)
    return d_train, d_test


# ── LightGBM ───────────────────────────────────────────────────────────────────

def run_lightgbm_seed(dev, train_start, train_end, test_start, test_end, seed):
    import lightgbm as lgb
    rows = []

    for h in HORIZONS:
        p = LIGHTGBM_PARAMS[h]
        d_train, d_test = get_train_test(dev, h, train_start, train_end, test_start, test_end)

        d_train_open = d_train[d_train["open_future"] == 1].copy()
        X_train = d_train_open[NUM_COLS + CAT_COLS]
        y_train = d_train_open["y"].to_numpy(dtype=float)

        pipe = Pipeline(steps=[
            ("pre", ColumnTransformer(transformers=[
                ("num", "passthrough", NUM_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ])),
            ("model", lgb.LGBMRegressor(
                n_estimators=1200,
                learning_rate=p["learning_rate"],
                num_leaves=p["num_leaves"],
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
            )),
        ])

        t0 = time.time()
        pipe.fit(X_train, np.log1p(y_train))
        train_time = time.time() - t0

        open_mask = d_test["open_future"] == 1
        yhat = np.maximum(0.0, np.expm1(pipe.predict(d_test.loc[open_mask, NUM_COLS + CAT_COLS])))
        macro = compute_macro(make_eval_frame_from_open_predictions(d_test, yhat))

        print(f"  [lightgbm seed={seed} h={h}]  macro WAPE={macro['WAPE']:.4f}%  MAE={macro['MAE']:.2f}  Bias={macro['Bias']:.2f}  ({train_time:.1f}s)")
        rows.append({
            "model": "lightgbm", "horizon": h, "seed": seed,
            "test_macro_wape": macro["WAPE"],
            "test_macro_mae":  macro["MAE"],
            "test_macro_rmse": macro["RMSE"],
            "test_macro_bias": macro["Bias"],
            "test_macro_wape_p90": macro["WAPE_p90"],
            "n_stores": macro["stores"],
            "train_time_seconds": round(train_time, 1),
        })
    return rows


# ── AutoGluon ──────────────────────────────────────────────────────────────────

def run_autogluon_seed(dev, train_start, train_end, test_start, test_end, seed):
    from autogluon.tabular import TabularPredictor
    feature_cols = NUM_COLS + CAT_COLS
    rows = []

    for h in HORIZONS:
        d_train, d_test = get_train_test(dev, h, train_start, train_end, test_start, test_end)

        d_train_open = d_train[d_train["open_future"] == 1].copy()
        train_df = d_train_open[feature_cols].copy()
        train_df["y_log"] = np.log1p(d_train_open["y"].to_numpy(dtype=float))

        tmp_path = os.path.join(ARTIFACTS_DIR, f"_ag_seed_tmp_h{h}_s{seed}")
        shutil.rmtree(tmp_path, ignore_errors=True)

        predictor = TabularPredictor(
            label="y_log", problem_type="regression", eval_metric="mae",
            path=tmp_path, learner_kwargs={"random_state": seed},
        )

        t0 = time.time()
        predictor.fit(
            train_data=train_df,
            time_limit=AUTOGLUON_TIME_BUDGET,
            presets="medium_quality",
            num_bag_folds=0,
            num_stack_levels=0,
            excluded_model_types=["NN_TORCH", "FASTAI"],
            verbosity=0,
        )
        train_time = time.time() - t0

        open_mask = d_test["open_future"] == 1
        yhat = np.maximum(0.0, np.expm1(
            np.asarray(predictor.predict(d_test.loc[open_mask, feature_cols], as_pandas=False), dtype=float)
        ))
        macro = compute_macro(make_eval_frame_from_open_predictions(d_test, yhat))

        shutil.rmtree(tmp_path, ignore_errors=True)

        print(f"  [autogluon seed={seed} h={h}]  macro WAPE={macro['WAPE']:.4f}%  MAE={macro['MAE']:.2f}  Bias={macro['Bias']:.2f}  ({train_time:.1f}s)")
        rows.append({
            "model": "autogluon", "horizon": h, "seed": seed,
            "test_macro_wape": macro["WAPE"],
            "test_macro_mae":  macro["MAE"],
            "test_macro_rmse": macro["RMSE"],
            "test_macro_bias": macro["Bias"],
            "test_macro_wape_p90": macro["WAPE_p90"],
            "n_stores": macro["stores"],
            "train_time_seconds": round(train_time, 1),
        })
    return rows


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(df):
    print("\n" + "=" * 74)
    print("SEED VARIANCE SUMMARY")
    print("=" * 74)
    for model_name in df["model"].unique():
        print(f"\nModel: {model_name.upper()}")
        m = df[df["model"] == model_name]

        # ── WAPE ──────────────────────────────────────────────────────────────
        print(f"\n  WAPE %")
        print(f"  {'Horizon':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        horizon_means = []
        for h in HORIZONS:
            vals = m[m["horizon"] == h]["test_macro_wape"]
            if vals.empty:
                continue
            mean = vals.mean()
            std  = vals.std(ddof=1) if len(vals) > 1 else 0.0
            horizon_means.append(mean)
            seeds = ", ".join(str(s) for s in m[m["horizon"] == h]["seed"].tolist())
            print(f"  t+{h:<10} {mean:>8.4f} {std:>8.4f} {vals.min():>8.4f} {vals.max():>8.4f}  seeds=[{seeds}]")
        if horizon_means:
            print(f"  {'Mean (1,7,14)':<12} {np.mean(horizon_means):>8.4f}")

        # ── MAE ───────────────────────────────────────────────────────────────
        print(f"\n  MAE (customers)")
        print(f"  {'Horizon':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        for h in HORIZONS:
            vals = m[m["horizon"] == h]["test_macro_mae"]
            if vals.empty:
                continue
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            print(f"  t+{h:<10} {vals.mean():>8.2f} {std:>8.2f} {vals.min():>8.2f} {vals.max():>8.2f}")

        # ── Bias ──────────────────────────────────────────────────────────────
        print(f"\n  Bias (customers, +ve = over-forecast)")
        print(f"  {'Horizon':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        for h in HORIZONS:
            vals = m[m["horizon"] == h]["test_macro_bias"]
            if vals.empty:
                continue
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            print(f"  t+{h:<10} {vals.mean():>8.2f} {std:>8.2f} {vals.min():>8.2f} {vals.max():>8.2f}")

    print("\n" + "=" * 74)


# ── Config -- change these to run different models/seeds ──────────────────────
MODELS_TO_RUN = ["lightgbm", "autogluon"]   # options: "lightgbm", "autogluon"
SEEDS         = [0, 42, 123]                


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print(f"Models : {MODELS_TO_RUN}")
    print(f"Seeds  : {SEEDS}\n")

    dev = load_dev_data()
    train_start, train_end, test_start, test_end = load_splits()

    all_rows = []
    for model_name in MODELS_TO_RUN:
        for seed in SEEDS:
            print(f"\n--- {model_name.upper()}  seed={seed} ---")
            if model_name == "lightgbm":
                rows = run_lightgbm_seed(dev, train_start, train_end, test_start, test_end, seed)
            else:
                rows = run_autogluon_seed(dev, train_start, train_end, test_start, test_end, seed)
            all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "seed_variance_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nRaw results saved to: {out_path}")
    print_summary(results_df)


if __name__ == "__main__":
    main()
