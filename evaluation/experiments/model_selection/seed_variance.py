"""
Seed Variance Analysis (All Models)
==================================
Re-trains configured models across multiple random seeds on the global train
window, evaluates on the fixed test window, and reports stability on test macro
metrics.

Resumability:
- If results/seed_variance/seed_variance_results.csv already exists, completed
  model-seed-horizon rows are reused and skipped (configurable by status).
- Results are checkpointed during the run so interrupted overnight jobs can
  resume instead of restarting from scratch.

Outputs:
1) Detailed per-seed rows:
   - results/seed_variance/seed_variance_results.csv
2) Per-model per-horizon summary (mean/std/min/max):
   - results/seed_variance/seed_variance_model_horizon_summary.csv
3) Compact model matrix (mean WAPE like model_horizon_wape_matrix):
   - results/seed_variance/seed_variance_model_wape_matrix.csv
4) Optional charts (controlled by ENABLE_CHARTS):
   - results/seed_variance/charts/
"""

from __future__ import annotations

import json
import os
import shutil
import time
import inspect
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn.utils as sklearn_utils
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training_helper import (
    NUM_COLS,
    CAT_COLS,
    add_features_per_store,
    build_target_cols,
    fill_missing_values,
    filter_issue_window,
    make_eval_frame_from_open_predictions,
    compute_macro,
)

warnings.filterwarnings("ignore")

DATA_PATH = "evaluation/data/processed/panel_train_clean.csv"
HOLDOUT_PATH = "evaluation/data/splits/holdout_stores.csv"
SPLITS_PATH = "evaluation/data/splits/time_splits_purged_kfold.json"
ARTIFACTS_DIR = "evaluation/experiments/model_selection/artifacts"
OUT_DIR = "evaluation/experiments/model_selection/results/seed_variance"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_RESULTS_CSV = os.path.join(OUT_DIR, "seed_variance_results.csv")
OUT_HORIZON_SUMMARY_CSV = os.path.join(OUT_DIR, "seed_variance_model_horizon_summary.csv")
OUT_WAPE_MATRIX_CSV = os.path.join(OUT_DIR, "seed_variance_model_wape_matrix.csv")
OUT_RUN_CONFIG_JSON = os.path.join(OUT_DIR, "seed_variance_run_config.json")
OUT_CHARTS_DIR = os.path.join(OUT_DIR, "charts")

HORIZONS = [1, 7, 14]
TARGET = "Customers"
SEEDS = [0, 42, 123]
ENABLE_CHARTS = True
CONTINUE_ON_ERROR = True
RESUME_FROM_EXISTING = True
SKIP_EXISTING_STATUSES = {"ok"}
SAVE_AFTER_EACH_ROW = True

# Keep ordering close to final model matrix outputs.
MODELS_TO_RUN = [
    "lightgbm",
    "autogluon",
    "autogluon_quick",
    "xgboost",
    "flaml_quick",
    "pycaret",
    "pycaret_quick",
    "flaml",
    "linear_regression",
    "last_value",
]

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "lightgbm": {
        "kind": "lightgbm",
        "params_by_h": {
            1: {"num_leaves": 63, "learning_rate": 0.10},
            7: {"num_leaves": 31, "learning_rate": 0.05},
            14: {"num_leaves": 31, "learning_rate": 0.05},
        },
    },
    "xgboost": {
        "kind": "xgboost",
        "params_by_h": {
            1: {"max_depth": 10, "learning_rate": 0.10},
            7: {"max_depth": 6, "learning_rate": 0.05},
            14: {"max_depth": 6, "learning_rate": 0.05},
        },
    },
    "linear_regression": {"kind": "linear_regression"},
    "last_value": {"kind": "last_value"},
    "autogluon": {"kind": "autogluon", "time_limit_seconds": 1200, "artifact_tag": "autogluon"},
    "autogluon_quick": {"kind": "autogluon", "time_limit_seconds": 300, "artifact_tag": "autogluon_quick"},
    "flaml": {"kind": "flaml", "time_limit_seconds": 1200},
    "flaml_quick": {"kind": "flaml", "time_limit_seconds": 300},
    # PyCaret compare_models budget_time is in minutes.
    "pycaret": {"kind": "pycaret", "budget_time_minutes": 20},
    "pycaret_quick": {"kind": "pycaret", "budget_time_minutes": 5},
}

RESULT_COLUMNS = [
    "model",
    "horizon",
    "seed",
    "status",
    "error",
    "test_macro_wape",
    "test_macro_mae",
    "test_macro_rmse",
    "test_macro_bias",
    "test_macro_wape_p90",
    "n_stores",
    "train_time_seconds",
]


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for chart generation. Install in your active env:\n"
            "  pip install matplotlib"
        ) from exc
    return plt


def ensure_pycaret_compatibility() -> None:
    # Compatibility shim for PyCaret on newer scikit-learn versions where
    # sklearn.utils._print_elapsed_time is no longer exported.
    if not hasattr(sklearn_utils, "_print_elapsed_time"):
        @contextmanager
        def _print_elapsed_time(*args, **kwargs):
            yield
        sklearn_utils._print_elapsed_time = _print_elapsed_time

    # Compatibility shim for PyCaret expecting old joblib internals.
    try:
        from joblib.memory import MemorizedFunc
        from joblib._store_backends import StoreBackendMixin

        if not getattr(StoreBackendMixin.load_item, "_pycaret_compat_wrapped", False):
            _orig_load_item = StoreBackendMixin.load_item

            def _load_item_compat(self, call_id, *args, **kwargs):
                kwargs.pop("msg", None)
                return _orig_load_item(self, call_id, *args, **kwargs)

            _load_item_compat._pycaret_compat_wrapped = True
            StoreBackendMixin.load_item = _load_item_compat

        if not hasattr(MemorizedFunc, "_get_output_identifiers"):
            def _get_output_identifiers(self, *args, **kwargs):
                return self.func_id, self._get_args_id(*args, **kwargs)
            MemorizedFunc._get_output_identifiers = _get_output_identifiers

        if not getattr(MemorizedFunc._persist_input, "_pycaret_compat_wrapped", False):
            _orig_persist_input = MemorizedFunc._persist_input
            _persist_params = tuple(inspect.signature(_orig_persist_input).parameters.keys())
            _persist_uses_call_id = len(_persist_params) >= 3 and _persist_params[2] == "call_id"

            # Only patch when the installed joblib API requires call_id.
            # joblib 1.3.x already uses _persist_input(duration, args, kwargs)
            # and should not be rewritten.
            if _persist_uses_call_id:
                def _persist_input_compat(self, duration, *args, **kwargs):
                    if len(args) >= 3:
                        return _orig_persist_input(self, duration, *args, **kwargs)
                    if len(args) == 2:
                        call_args, call_kwargs = args
                        if hasattr(self, "_get_output_identifiers") and isinstance(call_kwargs, dict):
                            call_id = self._get_output_identifiers(*call_args, **call_kwargs)
                            return _orig_persist_input(self, duration, call_id, call_args, call_kwargs, **kwargs)
                        return _orig_persist_input(self, duration, *args, **kwargs)
                    return _orig_persist_input(self, duration, *args, **kwargs)

                _persist_input_compat._pycaret_compat_wrapped = True
                MemorizedFunc._persist_input = _persist_input_compat
    except Exception:
        pass


def clean_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype(str).str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
    return out


def load_dev_data() -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    holdout = pd.read_csv(HOLDOUT_PATH)
    holdout_stores = set(holdout["Store"].tolist())

    dev = df[~df["Store"].isin(holdout_stores)].copy()
    dev = dev.sort_values(["Store", "Date"]).reset_index(drop=True)
    dev = dev.groupby("Store", group_keys=False).apply(add_features_per_store)
    print(f"Dev rows: {dev.shape[0]}, stores: {dev['Store'].nunique()}")
    return dev


def load_splits() -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    return (
        pd.to_datetime(splits["train_global"]["start"]),
        pd.to_datetime(splits["train_global"]["end"]),
        pd.to_datetime(splits["test"]["start"]),
        pd.to_datetime(splits["test"]["end"]),
    )


def build_horizon_frames(dev: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    frames: Dict[int, pd.DataFrame] = {}
    for h in HORIZONS:
        frames[h] = build_target_cols(dev, h, target_col=TARGET)
    return frames


def get_train_test(d_h: pd.DataFrame, train_start, train_end, test_start, test_end) -> Tuple[pd.DataFrame, pd.DataFrame]:
    needed = ["Date"] + NUM_COLS + CAT_COLS + [TARGET, "y", "open_future"]
    needed = list(dict.fromkeys(needed))

    d_train = filter_issue_window(d_h, train_start, train_end)[needed].copy().dropna(subset=["y"])
    d_test = filter_issue_window(d_h, test_start, test_end)[needed].copy().dropna(subset=["y"])
    d_train = fill_missing_values(d_train, NUM_COLS, CAT_COLS)
    d_test = fill_missing_values(d_test, NUM_COLS, CAT_COLS)
    return d_train, d_test


def run_last_value_once(d_train: pd.DataFrame, d_test: pd.DataFrame, seed: int, h: int, cfg: Dict[str, Any]):
    del d_train, seed, h, cfg
    open_mask = d_test["open_future"] == 1
    yhat_open = d_test.loc[open_mask, TARGET].to_numpy(dtype=float)
    macro = compute_macro(make_eval_frame_from_open_predictions(d_test, yhat_open))
    return macro, 0.0


def run_linear_regression_once(d_train: pd.DataFrame, d_test: pd.DataFrame, seed: int, h: int, cfg: Dict[str, Any]):
    del seed, h, cfg
    d_train_open = d_train[d_train["open_future"] == 1].copy()
    feature_cols = NUM_COLS + CAT_COLS
    X_train = d_train_open[feature_cols]
    y_train = d_train_open["y"].to_numpy(dtype=float)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", Ridge(alpha=1.0))])

    t0 = time.time()
    pipe.fit(X_train, np.log1p(y_train))
    train_time = time.time() - t0

    open_mask = d_test["open_future"] == 1
    if int(open_mask.sum()) > 0:
        X_test_open = d_test.loc[open_mask, feature_cols]
        yhat_open = np.maximum(0.0, np.expm1(pipe.predict(X_test_open)))
    else:
        yhat_open = np.array([], dtype=float)
    macro = compute_macro(make_eval_frame_from_open_predictions(d_test, yhat_open))
    return macro, train_time


def run_lightgbm_once(d_train: pd.DataFrame, d_test: pd.DataFrame, seed: int, h: int, cfg: Dict[str, Any]):
    import lightgbm as lgb

    p = cfg["params_by_h"][h]
    d_train_open = d_train[d_train["open_future"] == 1].copy()
    feature_cols = NUM_COLS + CAT_COLS
    X_train = d_train_open[feature_cols]
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
    if int(open_mask.sum()) > 0:
        X_test_open = d_test.loc[open_mask, feature_cols]
        yhat_open = np.maximum(0.0, np.expm1(pipe.predict(X_test_open)))
    else:
        yhat_open = np.array([], dtype=float)
    macro = compute_macro(make_eval_frame_from_open_predictions(d_test, yhat_open))
    return macro, train_time


def run_xgboost_once(d_train: pd.DataFrame, d_test: pd.DataFrame, seed: int, h: int, cfg: Dict[str, Any]):
    import xgboost as xgb

    p = cfg["params_by_h"][h]
    d_train_open = d_train[d_train["open_future"] == 1].copy()
    feature_cols = NUM_COLS + CAT_COLS
    X_train = d_train_open[feature_cols]
    y_train = d_train_open["y"].to_numpy(dtype=float)

    pipe = Pipeline(steps=[
        ("pre", ColumnTransformer(transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ])),
        ("model", xgb.XGBRegressor(
            n_estimators=1200,
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            objective="reg:squarederror",
        )),
    ])

    t0 = time.time()
    pipe.fit(X_train, np.log1p(y_train))
    train_time = time.time() - t0

    open_mask = d_test["open_future"] == 1
    if int(open_mask.sum()) > 0:
        X_test_open = d_test.loc[open_mask, feature_cols]
        yhat_open = np.maximum(0.0, np.expm1(pipe.predict(X_test_open)))
    else:
        yhat_open = np.array([], dtype=float)
    macro = compute_macro(make_eval_frame_from_open_predictions(d_test, yhat_open))
    return macro, train_time


def run_flaml_once(d_train: pd.DataFrame, d_test: pd.DataFrame, seed: int, h: int, cfg: Dict[str, Any]):
    from flaml import AutoML

    d_train_open = d_train[d_train["open_future"] == 1].copy()
    feature_cols = NUM_COLS + CAT_COLS
    X_train = d_train_open[feature_cols]
    y_train = d_train_open["y"].to_numpy(dtype=float)

    pre = ColumnTransformer([
        ("num", "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    X_train_t = pre.fit_transform(X_train)

    automl = AutoML()
    t0 = time.time()
    automl.fit(
        X_train=X_train_t,
        y_train=np.log1p(y_train),
        task="regression",
        metric="mae",
        time_budget=int(cfg["time_limit_seconds"]),
        estimator_list="auto",
        seed=seed,
        verbose=0,
    )
    train_time = time.time() - t0

    open_mask = d_test["open_future"] == 1
    if int(open_mask.sum()) > 0:
        X_test_open_t = pre.transform(d_test.loc[open_mask, feature_cols])
        yhat_open = np.maximum(0.0, np.expm1(automl.predict(X_test_open_t)))
    else:
        yhat_open = np.array([], dtype=float)
    macro = compute_macro(make_eval_frame_from_open_predictions(d_test, yhat_open))
    return macro, train_time


def run_autogluon_once(d_train: pd.DataFrame, d_test: pd.DataFrame, seed: int, h: int, cfg: Dict[str, Any]):
    from autogluon.tabular import TabularPredictor

    d_train_open = d_train[d_train["open_future"] == 1].copy()
    feature_cols = NUM_COLS + CAT_COLS
    train_df = d_train_open[feature_cols].copy()
    train_df["y_log"] = np.log1p(d_train_open["y"].to_numpy(dtype=float))

    tmp_path = os.path.join(
        ARTIFACTS_DIR,
        f"_seed_variance_{cfg['artifact_tag']}_h{h}_s{seed}",
    )
    shutil.rmtree(tmp_path, ignore_errors=True)

    predictor = TabularPredictor(
        label="y_log",
        problem_type="regression",
        eval_metric="mae",
        path=tmp_path,
        learner_kwargs={"random_state": seed},
    )

    t0 = time.time()
    predictor.fit(
        train_data=train_df,
        time_limit=int(cfg["time_limit_seconds"]),
        presets="medium_quality",
        num_bag_folds=0,
        num_stack_levels=0,
        excluded_model_types=["NN_TORCH", "FASTAI"],
        verbosity=0,
    )
    train_time = time.time() - t0

    open_mask = d_test["open_future"] == 1
    if int(open_mask.sum()) > 0:
        yhat_open = np.maximum(
            0.0,
            np.expm1(
                np.asarray(
                    predictor.predict(d_test.loc[open_mask, feature_cols], as_pandas=False),
                    dtype=float,
                )
            ),
        )
    else:
        yhat_open = np.array([], dtype=float)
    macro = compute_macro(make_eval_frame_from_open_predictions(d_test, yhat_open))

    shutil.rmtree(tmp_path, ignore_errors=True)
    return macro, train_time


def run_pycaret_once(d_train: pd.DataFrame, d_test: pd.DataFrame, seed: int, h: int, cfg: Dict[str, Any]):
    ensure_pycaret_compatibility()
    from pycaret.regression import setup, compare_models, finalize_model, predict_model

    d_train_open = d_train[d_train["open_future"] == 1].copy()
    feature_cols = NUM_COLS + CAT_COLS

    train_df = d_train_open[feature_cols + ["y"]].copy()
    train_df = clean_categoricals(train_df, CAT_COLS)
    train_df["y_log"] = np.log1p(train_df["y"].to_numpy(dtype=float))
    train_df = train_df.drop(columns=["y"])

    t0 = time.time()
    setup(
        data=train_df,
        target="y_log",
        session_id=seed,
        html=False,
        verbose=False,
        n_jobs=-1,
        data_split_shuffle=False,
        train_size=0.8,
    )
    best = compare_models(
        sort="MAE",
        n_select=1,
        cross_validation=False,
        budget_time=int(cfg["budget_time_minutes"]),
        turbo=True,
        errors="ignore",
        verbose=False,
    )
    best = finalize_model(best)
    train_time = time.time() - t0

    open_mask = d_test["open_future"] == 1
    if int(open_mask.sum()) > 0:
        X_test_open = d_test.loc[open_mask, feature_cols].copy()
        X_test_open = clean_categoricals(X_test_open, CAT_COLS)
        preds_df = predict_model(best, data=X_test_open)
        yhat_open = np.maximum(0.0, np.expm1(preds_df["prediction_label"].to_numpy(dtype=float)))
    else:
        yhat_open = np.array([], dtype=float)
    macro = compute_macro(make_eval_frame_from_open_predictions(d_test, yhat_open))
    return macro, train_time


RUNNER_BY_KIND = {
    "last_value": run_last_value_once,
    "linear_regression": run_linear_regression_once,
    "lightgbm": run_lightgbm_once,
    "xgboost": run_xgboost_once,
    "flaml": run_flaml_once,
    "autogluon": run_autogluon_once,
    "pycaret": run_pycaret_once,
}


def ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults: Dict[str, Any] = {
        "model": "",
        "horizon": np.nan,
        "seed": np.nan,
        "status": "",
        "error": "",
        "test_macro_wape": np.nan,
        "test_macro_mae": np.nan,
        "test_macro_rmse": np.nan,
        "test_macro_bias": np.nan,
        "test_macro_wape_p90": np.nan,
        "n_stores": np.nan,
        "train_time_seconds": np.nan,
    }
    out = df.copy()
    for col in RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = defaults[col]
    return out[RESULT_COLUMNS].copy()


def load_existing_results() -> pd.DataFrame:
    if not RESUME_FROM_EXISTING or not os.path.exists(OUT_RESULTS_CSV):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    try:
        existing = pd.read_csv(OUT_RESULTS_CSV)
    except Exception as exc:
        print(f"Resume disabled for this run: could not read {OUT_RESULTS_CSV} ({type(exc).__name__}: {exc})")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    existing = ensure_result_columns(existing)
    existing["model"] = existing["model"].astype(str)
    existing["status"] = existing["status"].astype(str)
    existing["error"] = existing["error"].fillna("").astype(str)
    existing["horizon"] = pd.to_numeric(existing["horizon"], errors="coerce")
    existing["seed"] = pd.to_numeric(existing["seed"], errors="coerce")
    existing = existing.dropna(subset=["model", "horizon", "seed", "status"]).copy()
    if existing.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    existing["horizon"] = existing["horizon"].astype(int)
    existing["seed"] = existing["seed"].astype(int)
    existing = existing.drop_duplicates(subset=["model", "seed", "horizon"], keep="last")
    existing = existing.sort_values(["model", "seed", "horizon"]).reset_index(drop=True)
    print(f"Resume: loaded {len(existing)} rows from {OUT_RESULTS_CSV}")
    return existing


def persist_results_csv(results_df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_result_columns(results_df)
    out = out.sort_values(["model", "seed", "horizon"]).reset_index(drop=True)
    out.to_csv(OUT_RESULTS_CSV, index=False)
    return out


def persist_all_outputs(results_df: pd.DataFrame, emit_paths: bool = True):
    results_df = persist_results_csv(results_df)
    horizon_summary_df = build_model_horizon_summary(results_df)
    horizon_summary_df.to_csv(OUT_HORIZON_SUMMARY_CSV, index=False)
    wape_matrix_df = build_wape_matrix(horizon_summary_df)
    wape_matrix_df.to_csv(OUT_WAPE_MATRIX_CSV, index=False)
    if emit_paths:
        print(f"\nSaved: {OUT_RESULTS_CSV}")
        print(f"Saved: {OUT_HORIZON_SUMMARY_CSV}")
        print(f"Saved: {OUT_WAPE_MATRIX_CSV}")
    return results_df, horizon_summary_df, wape_matrix_df


def run_model_seed_horizon(
    model_name: str,
    cfg: Dict[str, Any],
    d_train: pd.DataFrame,
    d_test: pd.DataFrame,
    seed: int,
    h: int,
) -> Dict[str, Any]:
    runner = RUNNER_BY_KIND[cfg["kind"]]
    row: Dict[str, Any] = {
        "model": model_name,
        "horizon": h,
        "seed": seed,
        "status": "ok",
        "error": "",
        "test_macro_wape": np.nan,
        "test_macro_mae": np.nan,
        "test_macro_rmse": np.nan,
        "test_macro_bias": np.nan,
        "test_macro_wape_p90": np.nan,
        "n_stores": np.nan,
        "train_time_seconds": np.nan,
    }

    try:
        macro, train_time = runner(d_train=d_train, d_test=d_test, seed=seed, h=h, cfg=cfg)
        row.update({
            "test_macro_wape": float(macro["WAPE"]),
            "test_macro_mae": float(macro["MAE"]),
            "test_macro_rmse": float(macro["RMSE"]),
            "test_macro_bias": float(macro["Bias"]),
            "test_macro_wape_p90": float(macro["WAPE_p90"]),
            "n_stores": int(macro["stores"]),
            "train_time_seconds": float(train_time),
        })
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        if not CONTINUE_ON_ERROR:
            raise
    return row


def _std_or_zero(s: pd.Series) -> float:
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=1))


def build_model_horizon_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    ok = results_df[results_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby(["model", "horizon"], as_index=False)
        .agg(
            test_wape_mean=("test_macro_wape", "mean"),
            test_wape_std=("test_macro_wape", _std_or_zero),
            test_wape_min=("test_macro_wape", "min"),
            test_wape_max=("test_macro_wape", "max"),
            test_mae_mean=("test_macro_mae", "mean"),
            test_rmse_mean=("test_macro_rmse", "mean"),
            test_bias_mean=("test_macro_bias", "mean"),
            test_wape_p90_mean=("test_macro_wape_p90", "mean"),
            train_time_seconds_mean=("train_time_seconds", "mean"),
            n_seeds=("seed", "nunique"),
            n_rows=("seed", "size"),
        )
        .sort_values(["horizon", "test_wape_mean", "model"])
        .reset_index(drop=True)
    )
    summary["rank_test_wape_within_horizon"] = (
        summary.groupby("horizon")["test_wape_mean"].rank(method="dense", ascending=True)
    )
    return summary


def build_wape_matrix(model_horizon_summary: pd.DataFrame) -> pd.DataFrame:
    if model_horizon_summary.empty:
        return pd.DataFrame()

    wape = (
        model_horizon_summary.pivot(index="model", columns="horizon", values="test_wape_mean")
        .reset_index()
    )
    rename_map = {
        1: "h1_test_wape",
        7: "h7_test_wape",
        14: "h14_test_wape",
    }
    wape = wape.rename(columns=rename_map)
    for col in ["h1_test_wape", "h7_test_wape", "h14_test_wape"]:
        if col not in wape.columns:
            wape[col] = np.nan

    wape["mean_test_wape_1_7_14"] = wape[["h1_test_wape", "h7_test_wape", "h14_test_wape"]].mean(axis=1, skipna=True)
    wape["rank_mean_test_wape_1_7_14"] = wape["mean_test_wape_1_7_14"].rank(method="dense", ascending=True)
    wape = wape.sort_values(["rank_mean_test_wape_1_7_14", "model"]).reset_index(drop=True)
    return wape


def print_console_summary(results_df: pd.DataFrame, matrix_df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("SEED VARIANCE SUMMARY")
    print("=" * 74)
    ok = results_df[results_df["status"] == "ok"].copy()
    failed = results_df[results_df["status"] != "ok"].copy()

    if ok.empty:
        print("No successful rows.")
    else:
        for model_name in sorted(ok["model"].unique()):
            m = ok[ok["model"] == model_name]
            print(f"\nModel: {model_name}")
            print(f"{'Horizon':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Seeds':>6}")
            for h in HORIZONS:
                vals = m[m["horizon"] == h]["test_macro_wape"]
                if vals.empty:
                    continue
                print(
                    f"t+{h:<8} {vals.mean():>8.4f} {_std_or_zero(vals):>8.4f} "
                    f"{vals.min():>8.4f} {vals.max():>8.4f} {len(vals):>6}"
                )

    if not failed.empty:
        print("\nFailures:")
        fail_counts = failed.groupby("model", as_index=False).size().sort_values("size", ascending=False)
        for _, r in fail_counts.iterrows():
            print(f"  {r['model']}: {int(r['size'])} failed rows")

    if not matrix_df.empty:
        print("\nTop by mean_test_wape_1_7_14:")
        show = matrix_df[["model", "mean_test_wape_1_7_14", "rank_mean_test_wape_1_7_14"]].head(10)
        for _, r in show.iterrows():
            print(
                f"  rank={int(r['rank_mean_test_wape_1_7_14'])} "
                f"model={r['model']} mean={float(r['mean_test_wape_1_7_14']):.4f}%"
            )
    print("=" * 74)


def plot_seed_overlay_per_horizon(plt, df: pd.DataFrame, out_dir: Path) -> None:
    models = [m for m in MODELS_TO_RUN if m in set(df["model"].astype(str))]
    for h in HORIZONS:
        d_h = df[(df["horizon"] == h) & (df["model"].isin(models))].copy()
        if d_h.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        for model in models:
            d_m = d_h[d_h["model"] == model].sort_values("seed")
            if d_m.empty:
                continue
            ax.plot(
                d_m["seed"].astype(int).to_numpy(),
                d_m["test_macro_wape"].astype(float).to_numpy(),
                marker="o",
                linewidth=2,
                label=model,
            )

        ax.set_title(f"Seed Variance | Horizon t+{h} | test_macro_wape")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Test Macro WAPE (%)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        out_path = out_dir / f"seed_variance_overlay_h{h}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_mean_std_summary(plt, horizon_summary_df: pd.DataFrame, out_dir: Path) -> None:
    if horizon_summary_df.empty:
        return
    models = [m for m in MODELS_TO_RUN if m in set(horizon_summary_df["model"].astype(str))]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(HORIZONS), dtype=float)
    width = 0.8 / max(1, len(models))

    for i, model in enumerate(models):
        vals = horizon_summary_df[horizon_summary_df["model"] == model].set_index("horizon")
        means = [float(vals.loc[h, "test_wape_mean"]) if h in vals.index else np.nan for h in HORIZONS]
        stds = [float(vals.loc[h, "test_wape_std"]) if h in vals.index else 0.0 for h in HORIZONS]
        xpos = x - 0.4 + width / 2 + i * width
        ax.bar(xpos, means, width=width, label=model, alpha=0.9)
        ax.errorbar(xpos, means, yerr=stds, fmt="none", ecolor="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{h}" for h in HORIZONS])
    ax.set_title("Seed Variance Summary | test_macro_wape mean +/- std")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Test Macro WAPE (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = out_dir / "seed_variance_mean_std_summary.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_run_config() -> None:
    payload = {
        "horizons": HORIZONS,
        "seeds": SEEDS,
        "models_to_run": MODELS_TO_RUN,
        "model_configs": MODEL_CONFIGS,
        "enable_charts": ENABLE_CHARTS,
        "resume_from_existing": RESUME_FROM_EXISTING,
        "skip_existing_statuses": sorted(SKIP_EXISTING_STATUSES),
        "save_after_each_row": SAVE_AFTER_EACH_ROW,
    }
    with open(OUT_RUN_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {OUT_RUN_CONFIG_JSON}")


def main() -> None:
    print(f"Models : {MODELS_TO_RUN}")
    print(f"Seeds  : {SEEDS}")
    print(f"Charts : {ENABLE_CHARTS}\n")
    print(f"Resume : {RESUME_FROM_EXISTING} (skip statuses={sorted(SKIP_EXISTING_STATUSES)})")

    for m in MODELS_TO_RUN:
        if m not in MODEL_CONFIGS:
            raise ValueError(f"Model '{m}' is not in MODEL_CONFIGS")

    save_run_config()
    results_df = load_existing_results()
    if results_df.empty:
        results_df = pd.DataFrame(columns=RESULT_COLUMNS)

    completed_keys = set()
    if not results_df.empty:
        reusable = results_df[results_df["status"].isin(SKIP_EXISTING_STATUSES)].copy()
        completed_keys = set(
            (str(r["model"]), int(r["seed"]), int(r["horizon"]))
            for _, r in reusable.iterrows()
        )
        if completed_keys:
            print(f"Resume: skipping {len(completed_keys)} previously completed rows.")

    dev = load_dev_data()
    train_start, train_end, test_start, test_end = load_splits()
    horizon_frames = build_horizon_frames(dev)

    for model_name in MODELS_TO_RUN:
        total_model_rows = len(SEEDS) * len(HORIZONS)
        skipped_for_model = sum(
            1 for seed in SEEDS for h in HORIZONS if (model_name, seed, h) in completed_keys
        )
        if skipped_for_model == total_model_rows:
            print(f"\n--- {model_name.upper()} already complete in checkpoint; skipping model ---")
            continue

        cfg = MODEL_CONFIGS[model_name]
        for seed in SEEDS:
            print(f"\n--- {model_name.upper()} seed={seed} ---")
            for h in HORIZONS:
                key = (model_name, seed, h)
                if key in completed_keys:
                    print(f"  [h={h}] SKIP: already completed in checkpoint")
                    continue

                d_train, d_test = get_train_test(
                    d_h=horizon_frames[h],
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )

                row = run_model_seed_horizon(
                    model_name=model_name,
                    cfg=cfg,
                    d_train=d_train,
                    d_test=d_test,
                    seed=seed,
                    h=h,
                )

                key_mask = (
                    (results_df["model"].astype(str) == model_name)
                    & (pd.to_numeric(results_df["seed"], errors="coerce") == seed)
                    & (pd.to_numeric(results_df["horizon"], errors="coerce") == h)
                )
                results_df = results_df.loc[~key_mask].copy()
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                if SAVE_AFTER_EACH_ROW:
                    results_df = persist_results_csv(results_df)

                if row["status"] == "ok":
                    completed_keys.add(key)
                    print(
                        f"  [h={h}] WAPE={row['test_macro_wape']:.4f}% "
                        f"MAE={row['test_macro_mae']:.2f} Bias={row['test_macro_bias']:.2f} "
                        f"({row['train_time_seconds']:.1f}s)"
                    )
                else:
                    print(f"  [h={h}] FAILED: {row['error']}")

        results_df, _, _ = persist_all_outputs(results_df, emit_paths=False)
        print(f"Checkpoint saved after model '{model_name}'.")

    results_df, horizon_summary_df, wape_matrix_df = persist_all_outputs(results_df, emit_paths=True)

    print_console_summary(results_df, wape_matrix_df)

    if ENABLE_CHARTS:
        ok = results_df[results_df["status"] == "ok"].copy()
        if ok.empty:
            print("Charts skipped: no successful rows.")
            return
        Path(OUT_CHARTS_DIR).mkdir(parents=True, exist_ok=True)
        plt = _import_matplotlib()
        plot_seed_overlay_per_horizon(plt, ok, Path(OUT_CHARTS_DIR))
        plot_mean_std_summary(plt, horizon_summary_df, Path(OUT_CHARTS_DIR))


if __name__ == "__main__":
    main()
