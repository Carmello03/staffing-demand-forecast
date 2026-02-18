
# Shared helper functions for rolling evaluation and model training

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

NUM_COLS: List[str] = [
    "DayOfWeek", "month", "year", "weekofyear", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "Promo", "SchoolHoliday",
    "CompetitionDistance", "Promo2",
    "lag1", "lag7", "lag14",
    "roll7_mean", "roll14_mean", "roll28_mean",
]

CAT_COLS: List[str] = [
    "Store", "StoreType", "Assortment", "StateHoliday", "PromoInterval",
]

FEATURE_COLS: List[str] = NUM_COLS + CAT_COLS


def dedupe_list(items: List[str]) -> List[str]:
    # Return a list with duplicates removed
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out

def add_features_per_store(g: pd.DataFrame, target_col: str = "Customers") -> pd.DataFrame:
    # Create time‑based and lag features for a single store
    g = g.sort_values("Date").copy()

    # Calendar and weekend features
    g["month"] = g["Date"].dt.month
    g["year"] = g["Date"].dt.year
    g["weekofyear"] = g["Date"].dt.isocalendar().week.astype(int)
    g["is_weekend"] = (g["DayOfWeek"] >= 6).astype(int)

    # Cyclical encodings
    g["dow_sin"] = np.sin(2 * np.pi * g["DayOfWeek"] / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * g["DayOfWeek"] / 7.0)
    g["month_sin"] = np.sin(2 * np.pi * g["month"] / 12.0)
    g["month_cos"] = np.cos(2 * np.pi * g["month"] / 12.0)

    # Lag features
    g["lag1"] = g[target_col].shift(1)
    g["lag7"] = g[target_col].shift(7)
    g["lag14"] = g[target_col].shift(14)

    # Rolling mean features 
    shifted = g[target_col].shift(1)
    g["roll7_mean"] = shifted.rolling(7).mean()
    g["roll14_mean"] = shifted.rolling(14).mean()
    g["roll28_mean"] = shifted.rolling(28).mean()

    return g


def build_target_cols(df: pd.DataFrame, h: int, target_col: str = "Customers") -> pd.DataFrame:
    # Add shifted target and open indicator columns for horizson = h
    out = df.copy()
    out["y"] = out.groupby("Store")[target_col].shift(-h)
    out["open_future"] = out.groupby("Store")["Open"].shift(-h)
    return out


def filter_issue_window(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    # Return a slice of df between issue dates start_date and end_date
    return df.loc[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

def filter_issue_ranges(df: pd.DataFrame, ranges: List[dict]) -> pd.DataFrame:
    # Return concatenated slices of df for multiple [start,end] issue-date ranges
    parts: List[pd.DataFrame] = []
    for r in ranges:
        start = pd.to_datetime(r["start"])
        end = pd.to_datetime(r["end"])
        parts.append(filter_issue_window(df, start, end))
    if len(parts) == 0:
        return df.iloc[0:0].copy()
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out

def fill_missing_values(df: pd.DataFrame, num_cols: List[str] | None = None, cat_cols: List[str] | None = None) -> pd.DataFrame:
    # Fill missing values in numeric and categorical columns
    if num_cols is None:
        num_cols = NUM_COLS
    if cat_cols is None:
        cat_cols = CAT_COLS
    df[num_cols] = df[num_cols].fillna(0)
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")
    return df


def make_eval_frame_from_open_predictions(df_window: pd.DataFrame, yhat_open: np.ndarray) -> pd.DataFrame:
    # Construct an evaluation frame with closed‑day prediction rule applied (yhat=0 when open_future=0)
    eval_df = df_window[["Store", "y", "open_future"]].copy()
    eval_df["yhat"] = 0.0
    open_mask = eval_df["open_future"] == 1
    n_open = int(open_mask.sum())
    if len(yhat_open) != n_open:
        raise ValueError(f"Length of yhat_open ({len(yhat_open)}) does not match number of open rows ({n_open}).")
    eval_df.loc[open_mask, "yhat"] = yhat_open
    return eval_df

def wape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Weighted absolute percentage error in percentage
    denom = float(np.sum(y_true))
    if denom == 0.0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Mean absolute error 
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Root mean squared error   
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias_mean_signed_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Mean signed error with positive values mean underprediction and negative values meaning overprediction
    return float(np.mean(y_true - y_pred))


def compute_micro(df_eval: pd.DataFrame) -> dict:
    # Compute thesis‑aligned micro metrics on an evaluation data
    if df_eval.empty:
        return {
            "MAE": float("nan"), "RMSE": float("nan"), "WAPE": float("nan"),
            "Bias": float("nan"), "N": 0, "stores": float("nan"), "WAPE_p90": float("nan"),
        }
    
    open_mask = df_eval["open_future"] == 1
    y_true_open = df_eval.loc[open_mask, "y"].to_numpy(dtype=float)
    y_pred_open = df_eval.loc[open_mask, "yhat"].to_numpy(dtype=float)
    y_true_all = df_eval["y"].to_numpy(dtype=float)
    y_pred_all = df_eval["yhat"].to_numpy(dtype=float)

    return {
        "MAE": mae(y_true_open, y_pred_open) if y_true_open.size else float("nan"),
        "RMSE": rmse(y_true_open, y_pred_open) if y_true_open.size else float("nan"),
        "WAPE": wape_percent(y_true_open, y_pred_open) if y_true_open.size else float("nan"),
        "Bias": bias_mean_signed_error(y_true_all, y_pred_all),
        "N": int(len(df_eval)),
        "stores": float("nan"),
        "WAPE_p90": float("nan"),
    }


def compute_macro(df_eval: pd.DataFrame) -> dict:
    # Compute thesis‑aligned macro metrics on an evaluation daat by averaging store‑level metrics
    if df_eval.empty:
        return {
            "MAE": float("nan"), "RMSE": float("nan"), "WAPE": float("nan"),
            "Bias": float("nan"), "N": 0, "stores": 0, "WAPE_p90": float("nan"),
        }
    
    store_ids = sorted(df_eval["Store"].unique())
    maes, rmses, wapes, biases = [], [], [], []
    for sid in store_ids:
        g = df_eval[df_eval["Store"] == sid]
        open_mask = g["open_future"] == 1
        y_true_open = g.loc[open_mask, "y"].to_numpy(dtype=float)
        y_pred_open = g.loc[open_mask, "yhat"].to_numpy(dtype=float)
        y_true_all = g["y"].to_numpy(dtype=float)
        y_pred_all = g["yhat"].to_numpy(dtype=float)
        if y_true_open.size:
            maes.append(mae(y_true_open, y_pred_open))
            rmses.append(rmse(y_true_open, y_pred_open))
            wapes.append(wape_percent(y_true_open, y_pred_open))
        biases.append(bias_mean_signed_error(y_true_all, y_pred_all))
        
    return {
        "MAE": float(np.mean(maes)) if maes else float("nan"),
        "RMSE": float(np.mean(rmses)) if rmses else float("nan"),
        "WAPE": float(np.mean(wapes)) if wapes else float("nan"),
        "Bias": float(np.mean(biases)) if biases else float("nan"),
        "N": int(len(df_eval)),
        "stores": int(len(store_ids)),
        "WAPE_p90": float(np.percentile(wapes, 90)) if wapes else float("nan"),
    }
