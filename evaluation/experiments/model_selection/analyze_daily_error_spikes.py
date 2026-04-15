"""
Analyze per-target-date error spikes for direct horizon models.

Purpose:
- Find specific dates where forecast error spikes.
- Check whether spikes align with weekday, promo, school holiday, or state holiday.
- Support thesis discussion when aggregate WAPE hides a few difficult dates.
- Each row is one target date (forecasted date), aggregated across open stores.

Outputs:
- daily_error_readable_<prefix>.csv
- daily_error_spike_summary_<prefix>.csv
- daily_error_model_horizon_summary_<prefix>.csv

All outputs are saved into:
- evaluation/experiments/model_selection/results/daily_results/

Note:
- These artifacts were saved with scikit-learn 1.5.2 in this project setup.
- If loading fails with a pickle/sklearn private-class error, run this script in
  an environment that matches the artifact sklearn version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn.compose._column_transformer as sklearn_column_transformer

from training_helper import (
    NUM_COLS,
    CAT_COLS,
    add_features_per_store,
    build_target_cols,
    filter_issue_window,
    fill_missing_values,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "evaluation" / "data" / "processed" / "panel_train_clean.csv"
HOLDOUT_PATH = ROOT / "evaluation" / "data" / "splits" / "holdout_stores.csv"
SPLITS_PATH = ROOT / "evaluation" / "data" / "splits" / "time_splits_purged_kfold.json"
ARTIFACT_DIR = ROOT / "evaluation" / "experiments" / "model_selection" / "artifacts"
RESULTS_DIR = ROOT / "evaluation" / "experiments" / "model_selection" / "results"
DAILY_RESULTS_DIR = RESULTS_DIR / "daily_results"

TARGET = "Customers"
HORIZONS = [1, 7, 14]
# Keep this hardcoded and simple for FYP experiments.
# Change this single value when you want a different model.
# Examples: "flaml", "lightgbm", "xgboost", "linear_regression"
ARTIFACT_PREFIX = "autogluon"
TOP_SPIKE_DAYS = 15


def add_sklearn_pickle_compat() -> None:
    # Compatibility shim for artifacts saved with a newer sklearn version.
    if hasattr(sklearn_column_transformer, "_RemainderColsList"):
        return

    class _RemainderColsList(list):
        pass

    sklearn_column_transformer._RemainderColsList = _RemainderColsList


def load_artifacts(prefix: str) -> dict[int, Any]:
    add_sklearn_pickle_compat()
    loaded: dict[int, Any] = {}
    for horizon in HORIZONS:
        base = ARTIFACT_DIR / f"{prefix}_h{horizon}"
        joblib_path = base.with_suffix(".joblib")
        pkl_path = base.with_suffix(".pkl")

        try:
            if prefix == "autogluon":
                if not base.exists():
                    raise FileNotFoundError(f"Missing artifact for h={horizon}: {base}")
                from autogluon.tabular import TabularPredictor

                loaded[horizon] = TabularPredictor.load(str(base))
            elif joblib_path.exists():
                loaded[horizon] = joblib.load(joblib_path)
            elif pkl_path.exists():
                loaded[horizon] = joblib.load(pkl_path)
            else:
                raise FileNotFoundError(
                    f"Missing artifact for h={horizon}: expected one of "
                    f"{joblib_path} or {pkl_path}"
                )
        except ModuleNotFoundError as exc:
            if exc.name == "ray":
                raise ModuleNotFoundError(
                    "Loading this artifact requires 'ray', which is not installed in evaluation/.venv. "
                    "For a simple run with current eval requirements, use model prefixes like "
                    "'lightgbm', 'xgboost', or 'linear_regression'."
                ) from exc
            if exc.name and exc.name.startswith("autogluon"):
                raise ModuleNotFoundError(
                    "Loading AutoGluon artifacts requires 'autogluon.tabular' in the active environment."
                ) from exc
            raise
    return loaded


def predict_open_rows(model_obj: Any, x_open: pd.DataFrame) -> np.ndarray:
    if len(x_open) == 0:
        return np.array([], dtype=float)

    if isinstance(model_obj, dict) and "pre" in model_obj and "automl" in model_obj:
        x_t = model_obj["pre"].transform(x_open)
        y_log = model_obj["automl"].predict(x_t)
    elif model_obj.__class__.__module__.startswith("autogluon."):
        y_log = model_obj.predict(x_open, as_pandas=False)
    else:
        y_log = model_obj.predict(x_open)

    y_hat = np.expm1(np.asarray(y_log, dtype=float))
    return np.maximum(0.0, y_hat)


def prepare_dev_frame() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    holdout = pd.read_csv(HOLDOUT_PATH)
    with SPLITS_PATH.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)

    holdout_stores = set(holdout["Store"].tolist())
    dev = df[~df["Store"].isin(holdout_stores)].copy()
    dev = dev.sort_values(["Store", "Date"]).reset_index(drop=True)
    dev = dev.groupby("Store", group_keys=False).apply(add_features_per_store)
    return dev, splits


def collect_split_windows(splits: dict) -> list[dict]:
    windows: list[dict] = []
    for fold in splits.get("val_folds", []):
        windows.append(
            {
                "split": "val",
                "fold": str(fold.get("fold", "")),
                "issue_start": pd.to_datetime(fold["val"]["start"]),
                "issue_end": pd.to_datetime(fold["val"]["end"]),
            }
        )

    windows.append(
        {
            "split": "test",
            "fold": "",
            "issue_start": pd.to_datetime(splits["test"]["start"]),
            "issue_end": pd.to_datetime(splits["test"]["end"]),
        }
    )
    return windows


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def build_daily_error_rows(
    d: pd.DataFrame,
    model_obj: Any,
    split_name: str,
    fold: str,
    issue_start: pd.Timestamp,
    issue_end: pd.Timestamp,
    horizon: int,
    artifact_prefix: str,
) -> list[dict]:
    needed_cols = ["Date"] + NUM_COLS + CAT_COLS + ["y", "open_future"]
    d_window = (
        filter_issue_window(d, issue_start, issue_end)[needed_cols]
        .copy()
        .dropna(subset=["y"])
    )
    d_window = fill_missing_values(d_window, NUM_COLS, CAT_COLS)

    target_date = d_window["Date"] + pd.to_timedelta(horizon, unit="D")
    d_window["target_date"] = target_date
    d_window["target_day_name"] = target_date.dt.day_name()
    d_window["target_month"] = target_date.dt.month
    d_window["target_year"] = target_date.dt.year
    d_window["target_has_state_holiday"] = (d_window["StateHoliday"].astype(str) != "0").astype(int)

    d_open = d_window[d_window["open_future"] == 1].copy()
    x_open = d_open[NUM_COLS + CAT_COLS]
    y_hat_open = predict_open_rows(model_obj, x_open)

    d_eval = d_open[
        [
            "Store",
            "Date",
            "target_date",
            "target_day_name",
            "target_month",
            "target_year",
            "Promo",
            "SchoolHoliday",
            "StateHoliday",
            "target_has_state_holiday",
            "y",
        ]
    ].copy()
    d_eval["yhat"] = y_hat_open
    d_eval["abs_error"] = np.abs(d_eval["y"] - d_eval["yhat"])
    d_eval["sq_error"] = (d_eval["y"] - d_eval["yhat"]) ** 2
    d_eval["signed_error"] = d_eval["y"] - d_eval["yhat"]

    grouped = (
        d_eval.groupby("target_date", as_index=False)
        .agg(
            split=("Store", lambda _: split_name),
            fold=("Store", lambda _: fold),
            horizon=("Store", lambda _: horizon),
            target_day_name=("target_day_name", "first"),
            target_month=("target_month", "first"),
            target_year=("target_year", "first"),
            open_store_count=("Store", "nunique"),
            open_row_count=("Store", "size"),
            target_total_customers=("y", "sum"),
            mean_target_customers=("y", "mean"),
            total_abs_error=("abs_error", "sum"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            mean_signed_error=("signed_error", "mean"),
            promo_rate_target=("Promo", "mean"),
            schoolholiday_rate_target=("SchoolHoliday", "mean"),
            stateholiday_rate_target=("target_has_state_holiday", "mean"),
            stateholiday_code=("StateHoliday", lambda s: next((v for v in s.astype(str) if v != "0"), "0")),
        )
        .sort_values("target_date")
        .reset_index(drop=True)
    )

    grouped["daily_wape"] = grouped.apply(
        lambda row: _safe_div(float(row["total_abs_error"]), float(row["target_total_customers"])) * 100.0,
        axis=1,
    )
    grouped["artifact_prefix"] = artifact_prefix
    grouped["issue_start"] = str(issue_start.date())
    grouped["issue_end"] = str(issue_end.date())
    grouped["target_date"] = grouped["target_date"].dt.strftime("%Y-%m-%d")
    grouped["has_promo_heavy_target"] = (grouped["promo_rate_target"] >= 0.5).astype(int)
    grouped["has_schoolholiday_signal"] = (grouped["schoolholiday_rate_target"] > 0).astype(int)
    grouped["has_stateholiday_signal"] = (grouped["stateholiday_rate_target"] > 0).astype(int)

    ordered_cols = [
        "artifact_prefix",
        "split",
        "fold",
        "horizon",
        "issue_start",
        "issue_end",
        "target_date",
        "target_day_name",
        "target_month",
        "target_year",
        "open_store_count",
        "open_row_count",
        "target_total_customers",
        "mean_target_customers",
        "total_abs_error",
        "mae",
        "rmse",
        "mean_signed_error",
        "daily_wape",
        "promo_rate_target",
        "schoolholiday_rate_target",
        "stateholiday_rate_target",
        "stateholiday_code",
        "has_promo_heavy_target",
        "has_schoolholiday_signal",
        "has_stateholiday_signal",
    ]
    return grouped[ordered_cols].to_dict(orient="records")


def build_spike_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (split_name, fold, horizon), g in daily_df.groupby(["split", "fold", "horizon"], dropna=False):
        ranked = g.sort_values(["daily_wape", "total_abs_error"], ascending=[False, False]).copy()
        ranked["daily_wape_rank"] = range(1, len(ranked) + 1)
        ranked["daily_wape_p90_cutoff"] = float(ranked["daily_wape"].quantile(0.90))
        ranked["is_top_decile_spike"] = (ranked["daily_wape"] >= ranked["daily_wape_p90_cutoff"]).astype(int)
        parts.append(ranked.head(TOP_SPIKE_DAYS))

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_readable_daily_view(daily_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "artifact_prefix",
        "split",
        "fold",
        "horizon",
        "target_date",
        "target_day_name",
        "daily_wape",
        "total_abs_error",
        "target_total_customers",
        "mean_signed_error",
        "promo_rate_target",
        "schoolholiday_rate_target",
        "stateholiday_code",
    ]
    readable = daily_df[cols].copy()
    return readable.sort_values(["split", "fold", "horizon", "target_date"]).reset_index(drop=True)


def build_model_horizon_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()

    summary = (
        daily_df.groupby(["split", "horizon"], dropna=False)
        .agg(
            day_count=("target_date", "count"),
            mean_daily_wape=("daily_wape", "mean"),
            median_daily_wape=("daily_wape", "median"),
            p90_daily_wape=("daily_wape", lambda s: float(np.percentile(s, 90))),
            max_daily_wape=("daily_wape", "max"),
            mean_abs_bias=("mean_signed_error", lambda s: float(np.mean(np.abs(s)))),
        )
        .reset_index()
    )

    top_idx = daily_df.groupby(["split", "horizon"], dropna=False)["daily_wape"].idxmax()
    top_rows = daily_df.loc[top_idx, ["split", "horizon", "target_date", "daily_wape"]].copy()
    top_rows = top_rows.rename(
        columns={
            "target_date": "top_spike_target_date",
            "daily_wape": "top_spike_daily_wape",
        }
    )

    out = summary.merge(top_rows, on=["split", "horizon"], how="left")
    out["artifact_prefix"] = ARTIFACT_PREFIX
    ordered_cols = [
        "artifact_prefix",
        "split",
        "horizon",
        "day_count",
        "mean_daily_wape",
        "median_daily_wape",
        "p90_daily_wape",
        "max_daily_wape",
        "mean_abs_bias",
        "top_spike_target_date",
        "top_spike_daily_wape",
    ]
    return out[ordered_cols].sort_values(["split", "horizon"]).reset_index(drop=True)


def main() -> None:
    DAILY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_results_dir = DAILY_RESULTS_DIR / ARTIFACT_PREFIX
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and splits...")
    print(f"Using model: {ARTIFACT_PREFIX}")
    print("Per-day error is computed per target_date (forecast date), aggregated across open stores.")
    dev, splits = prepare_dev_frame()
    windows = collect_split_windows(splits)
    artifacts = load_artifacts(ARTIFACT_PREFIX)

    results: list[dict] = []
    for horizon in HORIZONS:
        print(f"Preparing horizon {horizon}...")
        d = build_target_cols(dev, horizon, target_col=TARGET)

        for window in windows:
            label = f"{window['split']}"
            if window["split"] == "val":
                label = f"{label} fold {window['fold']}"
            print(
                f"  Evaluating {label}: "
                f"issue {window['issue_start'].date()} to {window['issue_end'].date()}"
            )

            rows = build_daily_error_rows(
                d=d,
                model_obj=artifacts[horizon],
                split_name=window["split"],
                fold=window["fold"],
                issue_start=window["issue_start"],
                issue_end=window["issue_end"],
                horizon=horizon,
                artifact_prefix=ARTIFACT_PREFIX,
            )
            results.extend(rows)

    daily_df = pd.DataFrame(results)
    if daily_df.empty:
        print("No evaluation rows were generated.")
        return

    readable_df = build_readable_daily_view(daily_df)
    spike_df = build_spike_summary(daily_df)
    model_summary_df = build_model_horizon_summary(daily_df)

    readable_out = model_results_dir / f"daily_error_readable_{ARTIFACT_PREFIX}.csv"
    spike_out = model_results_dir / f"daily_error_spike_summary_{ARTIFACT_PREFIX}.csv"
    model_summary_out = model_results_dir / f"daily_error_model_horizon_summary_{ARTIFACT_PREFIX}.csv"

    readable_df.to_csv(readable_out, index=False)
    spike_df.to_csv(spike_out, index=False)
    model_summary_df.to_csv(model_summary_out, index=False)

    print()
    print(f"Saved: {readable_out}")
    print(f"Saved: {spike_out}")
    print(f"Saved: {model_summary_out}")

    if not model_summary_df.empty:
        print()
        print("Summary (mean daily WAPE by split/horizon):")
        for _, row in model_summary_df.iterrows():
            print(
                f"  split={row['split']}, h={int(row['horizon'])}: "
                f"mean={float(row['mean_daily_wape']):.2f}, "
                f"p90={float(row['p90_daily_wape']):.2f}, "
                f"top_spike={row['top_spike_target_date']} ({float(row['top_spike_daily_wape']):.2f})"
            )


if __name__ == "__main__":
    main()
