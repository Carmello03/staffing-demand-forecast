"""
Rolling-origin LightGBM training

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

import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from training_helper import (
    NUM_COLS,
    CAT_COLS,
    add_features_per_store,
    build_target_cols,
    filter_issue_window,
    fill_missing_values,
    make_eval_frame_from_open_predictions,
    compute_micro,
    compute_macro,
)

DATA_PATH = "data/processed/panel_train_clean.csv"
HOLDOUT_PATH = "data/splits/holdout_stores.csv"
SPLITS_PATH = "data/splits/time_splits_rolling.json"

OUT_RESULTS_DIR = "experiments/model_selection/results"
OUT_ARTIFACTS_DIR = "experiments/model_selection/artifacts"
os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUT_ARTIFACTS_DIR, exist_ok=True)

HORIZONS = [1, 7, 14]
TARGET = "Customers"
SEED = 42


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

    param_grid = []
    for num_leaves in [31, 63]:
        for learning_rate in [0.05, 0.1]:
            param_grid.append({"num_leaves": num_leaves, "learning_rate": learning_rate})

    results = []

    val_folds = splits.get("val_folds", [])
    train_global_start = pd.to_datetime(splits["train_global"]["start"])
    train_global_end = pd.to_datetime(splits["train_global"]["end"])
    test_start = pd.to_datetime(splits["test"]["start"])
    test_end = pd.to_datetime(splits["test"]["end"])

    for h in HORIZONS:
        print(f"\nTraining horizon: {h}")
        d = build_target_cols(dev, h, target_col=TARGET)
        needed_cols = NUM_COLS + CAT_COLS + ["y", "open_future"]

        # Hyperparameter search across folds (macro WAPE)
        best_param = None
        best_score = None
        for p in param_grid:
            fold_scores = []
            for fold in val_folds:
                train_start = pd.to_datetime(fold["train"]["start"])
                train_end = pd.to_datetime(fold["train"]["end"])
                val_start = pd.to_datetime(fold["val"]["start"])
                val_end = pd.to_datetime(fold["val"]["end"])

                d_train = filter_issue_window(d, train_start, train_end)[needed_cols].copy().dropna(subset=["y"])
                d_val = filter_issue_window(d, val_start, val_end)[needed_cols].copy().dropna(subset=["y"])

                d_train = fill_missing_values(d_train, NUM_COLS, CAT_COLS)
                d_val = fill_missing_values(d_val, NUM_COLS, CAT_COLS)

                d_train_open = d_train[d_train["open_future"] == 1].copy()
                d_val_open = d_val[d_val["open_future"] == 1].copy()

                X_train = d_train_open[NUM_COLS + CAT_COLS]
                y_train = d_train_open["y"].to_numpy(dtype=float)
                X_val_open = d_val_open[NUM_COLS + CAT_COLS]

                pre = ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", NUM_COLS),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                    ]
                )

                model = lgb.LGBMRegressor(
                    n_estimators=1200,
                    learning_rate=p["learning_rate"],
                    num_leaves=p["num_leaves"],
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=SEED,
                    n_jobs=-1,
                )

                pipe = Pipeline(steps=[("pre", pre), ("model", model)])

                pipe.fit(X_train, np.log1p(y_train))

                yhat_open = np.expm1(pipe.predict(X_val_open))
                yhat_open = np.maximum(0.0, yhat_open)

                val_eval = make_eval_frame_from_open_predictions(d_val, yhat_open)
                macro_metrics = compute_macro(val_eval)
                fold_scores.append(macro_metrics["WAPE"])

            mean_score = float(np.mean(fold_scores)) if fold_scores else float("inf")
            if best_score is None or mean_score < best_score:
                best_score = mean_score
                best_param = p

        print(f"Best params for h={h}: {best_param}, mean val macro WAPE: {best_score:.4f}%")

        # Cross-validation evaluation using best params
        fold_macro_wapes = []
        for fold_idx, fold in enumerate(val_folds, start=1):
            train_start = pd.to_datetime(fold["train"]["start"])
            train_end = pd.to_datetime(fold["train"]["end"])
            val_start = pd.to_datetime(fold["val"]["start"])
            val_end = pd.to_datetime(fold["val"]["end"])

            d_train = filter_issue_window(d, train_start, train_end)[needed_cols].copy().dropna(subset=["y"])
            d_val = filter_issue_window(d, val_start, val_end)[needed_cols].copy().dropna(subset=["y"])

            d_train = fill_missing_values(d_train, NUM_COLS, CAT_COLS)
            d_val = fill_missing_values(d_val, NUM_COLS, CAT_COLS)

            d_train_open = d_train[d_train["open_future"] == 1].copy()
            d_val_open = d_val[d_val["open_future"] == 1].copy()

            X_train = d_train_open[NUM_COLS + CAT_COLS]
            y_train = d_train_open["y"].to_numpy(dtype=float)
            X_val_open = d_val_open[NUM_COLS + CAT_COLS]

            pre_cv = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", NUM_COLS),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                ]
            )
            model_cv = lgb.LGBMRegressor(
                n_estimators=1200,
                learning_rate=best_param["learning_rate"],
                num_leaves=best_param["num_leaves"],
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=-1,
            )
            pipe_cv = Pipeline(steps=[("pre", pre_cv), ("model", model_cv)])

            t0 = time.time()
            pipe_cv.fit(X_train, np.log1p(y_train))
            train_time = float(time.time() - t0)

            yhat_open = np.expm1(pipe_cv.predict(X_val_open))
            yhat_open = np.maximum(0.0, yhat_open)

            val_eval = make_eval_frame_from_open_predictions(d_val, yhat_open)

            micro_metrics = compute_micro(val_eval)
            macro_metrics = compute_macro(val_eval)

            fold_macro_wapes.append(macro_metrics["WAPE"])

            results.append({
                "model": "lightgbm_log",
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
            })
            results.append({
                "model": "lightgbm_log",
                "horizon": h,
                "split": "val",
                "fold": fold_idx,
                "agg": "macro",
                "MAE": macro_metrics["MAE"],
                "RMSE": macro_metrics["RMSE"],
                "WAPE": macro_metrics["WAPE"],
                "Bias": macro_metrics["Bias"],
                "N": int(len(val_eval)),
                "training_time_seconds": train_time,
                "stores": macro_metrics["stores"],
                "WAPE_p90": macro_metrics["WAPE_p90"],
            })

        cv_mean_macro_wape = float(np.mean(fold_macro_wapes)) if fold_macro_wapes else float("nan")
        print(f"Mean val macro WAPE for h={h}: {cv_mean_macro_wape}")

        # Final training on global train window + evaluation on fixed test window
        d_train_global = filter_issue_window(d, train_global_start, train_global_end)[needed_cols].copy().dropna(subset=["y"])
        d_test = filter_issue_window(d, test_start, test_end)[needed_cols].copy().dropna(subset=["y"])

        d_train_global = fill_missing_values(d_train_global, NUM_COLS, CAT_COLS)
        d_test = fill_missing_values(d_test, NUM_COLS, CAT_COLS)

        d_train_global_open = d_train_global[d_train_global["open_future"] == 1].copy()

        X_train_g = d_train_global_open[NUM_COLS + CAT_COLS]
        y_train_g = d_train_global_open["y"].to_numpy(dtype=float)

        pre_final = ColumnTransformer(
            transformers=[
                ("num", "passthrough", NUM_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ]
        )
        model_final = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=best_param["learning_rate"],
            num_leaves=best_param["num_leaves"],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
        )
        pipe_final = Pipeline(steps=[("pre", pre_final), ("model", model_final)])

        t0 = time.time()
        pipe_final.fit(X_train_g, np.log1p(y_train_g))
        final_train_time = float(time.time() - t0)

        open_mask_test = (d_test["open_future"] == 1)
        X_test_open = d_test.loc[open_mask_test, NUM_COLS + CAT_COLS]
        yhat_test_open = np.expm1(pipe_final.predict(X_test_open))
        yhat_test_open = np.maximum(0.0, yhat_test_open)

        test_eval = make_eval_frame_from_open_predictions(d_test, yhat_test_open)

        micro_test = compute_micro(test_eval)
        macro_test = compute_macro(test_eval)

        results.append({
            "model": "lightgbm_log",
            "horizon": h,
            "split": "test",
            "fold": "",
            "agg": "micro",
            "MAE": micro_test["MAE"],
            "RMSE": micro_test["RMSE"],
            "WAPE": micro_test["WAPE"],
            "Bias": micro_test["Bias"],
            "N": micro_test["N"],
            "training_time_seconds": final_train_time,
        })
        results.append({
            "model": "lightgbm_log",
            "horizon": h,
            "split": "test",
            "fold": "",
            "agg": "macro",
            "MAE": macro_test["MAE"],
            "RMSE": macro_test["RMSE"],
            "WAPE": macro_test["WAPE"],
            "Bias": macro_test["Bias"],
            "N": int(len(test_eval)),
            "training_time_seconds": final_train_time,
            "stores": macro_test["stores"],
            "WAPE_p90": macro_test["WAPE_p90"],
        })

        model_path = os.path.join(OUT_ARTIFACTS_DIR, f"lightgbm_h{h}.joblib")
        joblib.dump(pipe_final, model_path)
        params_path = os.path.join(OUT_ARTIFACTS_DIR, f"lightgbm_h{h}_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_param, f, indent=2)
        print("Saved model:", model_path)
        print("Saved params:", params_path)

    out = pd.DataFrame(results).sort_values(["split", "fold", "horizon", "agg"]).reset_index(drop=True)
    out_path = os.path.join(OUT_RESULTS_DIR, "lightgbm_metrics.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    test_macro = out[(out["split"] == "test") & (out["agg"] == "macro")]
    primary_score = float(test_macro["WAPE"].mean()) if not test_macro.empty else float("nan")
    val_macro = out[(out["split"] == "val") & (out["agg"] == "macro")]
    val_cv_score = float(val_macro["WAPE"].mean()) if not val_macro.empty else float("nan")

    print(f"Test mean macro WAPE across horizons: {primary_score:.4f}%")
    print(f"Validation mean macro WAPE across folds: {val_cv_score:.4f}%")


if __name__ == "__main__":
    main()
