# evaluation/experiments/model_selection/train_lightgbm_holdout_local_vs_global.py

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn.compose._column_transformer as sklearn_column_transformer

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

DATA_PATH = "evaluation/data/processed/panel_train_clean.csv"
HOLDOUT_PATH = "evaluation/data/splits/holdout_stores.csv"
SPLITS_PATH = "evaluation/data/splits/time_splits_purged_kfold.json"

GLOBAL_ARTIFACTS_DIR = "evaluation/experiments/model_selection/artifacts"
OUT_RESULTS_DIR = "evaluation/experiments/model_selection/results"
OUT_LOCAL_ARTIFACTS_DIR = os.path.join(GLOBAL_ARTIFACTS_DIR, "lightgbm_local_holdouts")

HORIZONS = [1, 7, 14]
TARGET = "Customers"
SEED = 42
N_ESTIMATORS = 1200

SAVE_LOCAL_MODELS = True


def add_sklearn_pickle_compat() -> None:
    # Compatibility shim for artifacts saved with a newer sklearn version.
    if hasattr(sklearn_column_transformer, "_RemainderColsList"):
        return

    class _RemainderColsList(list):
        pass

    sklearn_column_transformer._RemainderColsList = _RemainderColsList


def build_pipe(best_param: dict) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=float(best_param.get("learning_rate", 0.05)),
        num_leaves=int(best_param.get("num_leaves", 31)),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
    )

    return Pipeline(steps=[("pre", pre), ("model", model)])


def load_best_params(h: int) -> dict:
    params_path = os.path.join(GLOBAL_ARTIFACTS_DIR, f"lightgbm_h{h}_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"num_leaves": 31, "learning_rate": 0.05}


def predict_with_pipe(pipe: Pipeline, df_window: pd.DataFrame) -> pd.DataFrame:
    open_mask = df_window["open_future"] == 1
    X_open = df_window.loc[open_mask, NUM_COLS + CAT_COLS]
    yhat_open = np.expm1(pipe.predict(X_open))
    yhat_open = np.maximum(0.0, yhat_open)
    return make_eval_frame_from_open_predictions(df_window, yhat_open)


def pick_metric_row(agg_rows: list, scope: str, horizon: int, agg: str) -> dict:
    for r in agg_rows:
        if r["scope"] == scope and r["horizon"] == horizon and r["agg"] == agg:
            return r
    return {}


def main() -> None:
    os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
    os.makedirs(OUT_LOCAL_ARTIFACTS_DIR, exist_ok=True)
    add_sklearn_pickle_compat()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    holdout_df = pd.read_csv(HOLDOUT_PATH)

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    df["Store"] = df["Store"].astype(int)
    holdout_df["Store"] = holdout_df["Store"].astype(int)

    holdout_ids = sorted(holdout_df["Store"].tolist())
    print("Holdout stores:", len(holdout_ids))

    train_start = pd.to_datetime(splits["train_global"]["start"])
    train_end = pd.to_datetime(splits["train_global"]["end"])
    test_start = pd.to_datetime(splits["test"]["start"])
    test_end = pd.to_datetime(splits["test"]["end"])

    hold = df[df["Store"].isin(holdout_ids)].copy()
    hold = hold.sort_values(["Store", "Date"]).reset_index(drop=True)
    hold = hold.groupby("Store", group_keys=False).apply(add_features_per_store)

    store_rows = []
    agg_rows = []

    for h in HORIZONS:
        print(f"\nHorizon {h}")
        best_param = load_best_params(h)
        d = build_target_cols(hold, h, target_col=TARGET)

        needed_cols = ["Date"] + NUM_COLS + CAT_COLS + ["y", "open_future"]

        global_model_path = os.path.join(GLOBAL_ARTIFACTS_DIR, f"lightgbm_h{h}.joblib")
        pipe_global = None
        if os.path.exists(global_model_path):
            pipe_global = joblib.load(global_model_path)
            print("Loaded global model")

        eval_all_global = []
        eval_all_local = []

        for sid in holdout_ids:
            g = d[d["Store"] == sid][needed_cols].copy().dropna(subset=["y"])
            if g.empty:
                continue

            g_train = filter_issue_window(g, train_start, train_end).copy()
            g_test = filter_issue_window(g, test_start, test_end).copy()
            if g_train.empty or g_test.empty:
                continue

            g_train = fill_missing_values(g_train, NUM_COLS, CAT_COLS)
            g_test = fill_missing_values(g_test, NUM_COLS, CAT_COLS)

            g_train_open = g_train[g_train["open_future"] == 1].copy()
            if g_train_open.empty:
                continue

            X_train = g_train_open[NUM_COLS + CAT_COLS]
            y_train = g_train_open["y"].to_numpy(dtype=float)

            pipe_local = build_pipe(best_param)

            t0 = time.time()
            pipe_local.fit(X_train, np.log1p(y_train))
            local_train_time = float(time.time() - t0)

            eval_local = predict_with_pipe(pipe_local, g_test)
            eval_all_local.append(eval_local)

            m_local = compute_micro(eval_local)
            store_rows.append(
                {
                    "scope": "local",
                    "Store": int(sid),
                    "horizon": int(h),
                    "WAPE": m_local["WAPE"],
                    "MAE": m_local["MAE"],
                    "RMSE": m_local["RMSE"],
                    "Bias": m_local["Bias"],
                    "N": m_local["N"],
                    "training_time_seconds": local_train_time,
                }
            )

            if SAVE_LOCAL_MODELS:
                local_path = os.path.join(OUT_LOCAL_ARTIFACTS_DIR, f"store_{sid}_h{h}.joblib")
                joblib.dump(pipe_local, local_path)

            if pipe_global is not None:
                eval_global = predict_with_pipe(pipe_global, g_test)
                eval_all_global.append(eval_global)

                m_global = compute_micro(eval_global)
                store_rows.append(
                    {
                        "scope": "global",
                        "Store": int(sid),
                        "horizon": int(h),
                        "WAPE": m_global["WAPE"],
                        "MAE": m_global["MAE"],
                        "RMSE": m_global["RMSE"],
                        "Bias": m_global["Bias"],
                        "N": m_global["N"],
                        "training_time_seconds": 0.0,
                    }
                )

        if eval_all_local:
            all_local = pd.concat(eval_all_local, ignore_index=True)
            agg_rows.append({"scope": "local", "horizon": h, "agg": "micro", **compute_micro(all_local)})
            agg_rows.append({"scope": "local", "horizon": h, "agg": "macro", **compute_macro(all_local)})

        if eval_all_global:
            all_global = pd.concat(eval_all_global, ignore_index=True)
            agg_rows.append({"scope": "global", "horizon": h, "agg": "micro", **compute_micro(all_global)})
            agg_rows.append({"scope": "global", "horizon": h, "agg": "macro", **compute_macro(all_global)})

    store_df = pd.DataFrame(store_rows)
    store_out = os.path.join(OUT_RESULTS_DIR, "holdout_local_vs_global_store_metrics.csv")
    store_df.to_csv(store_out, index=False)
    print("\nSaved store metrics")

    agg_df = pd.DataFrame(agg_rows)
    agg_out = os.path.join(OUT_RESULTS_DIR, "holdout_local_vs_global_agg_metrics.csv")
    agg_df.to_csv(agg_out, index=False)
    print("Saved agg metrics")

    delta_out = os.path.join(OUT_RESULTS_DIR, "holdout_local_vs_global_deltas.csv")
    summary_out = os.path.join(OUT_RESULTS_DIR, "holdout_local_vs_global_summary.csv")

    if not store_df.empty:
        piv = (
            store_df.pivot_table(index=["Store", "horizon"], columns="scope", values="WAPE", aggfunc="mean")
            .reset_index()
        )
        if "global" in piv.columns and "local" in piv.columns:
            piv["delta_WAPE_global_minus_local"] = piv["global"] - piv["local"]
        piv.to_csv(delta_out, index=False)
        print("Saved deltas")

        summary_rows = []
        for h in HORIZONS:
            d_h = piv[piv["horizon"] == h]
            local_wins = int((d_h["delta_WAPE_global_minus_local"] > 0).sum())
            global_wins = int((d_h["delta_WAPE_global_minus_local"] < 0).sum())
            ties = int((d_h["delta_WAPE_global_minus_local"] == 0).sum())

            gm = pick_metric_row(agg_rows, "global", h, "macro")
            lm = pick_metric_row(agg_rows, "local", h, "macro")
            gi = pick_metric_row(agg_rows, "global", h, "micro")
            li = pick_metric_row(agg_rows, "local", h, "micro")

            summary_rows.append(
                {
                    "horizon": int(h),
                    "global_macro_WAPE_%": float(gm.get("WAPE", np.nan)),
                    "local_macro_WAPE_%": float(lm.get("WAPE", np.nan)),
                    "global_micro_WAPE_%": float(gi.get("WAPE", np.nan)),
                    "local_micro_WAPE_%": float(li.get("WAPE", np.nan)),
                    "local_wins_stores": local_wins,
                    "global_wins_stores": global_wins,
                    "ties": ties,
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values("horizon")
        summary_df.to_csv(summary_out, index=False)
        print("Saved simple summary")

    print("\nDone.")


if __name__ == "__main__":
    main()
