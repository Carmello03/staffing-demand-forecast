import pandas as pd
import os

RESULTS_DIR = r"evaluation\experiments\model_selection\results"
HORIZON_SUMMARY_CSV = os.path.join(RESULTS_DIR, "model_horizon_summary.csv")
HORIZON_WAPE_MATRIX_CSV = os.path.join(RESULTS_DIR, "model_horizon_wape_matrix.csv")

# plot the error accorss the horizon for holodout stores, possibilty that test set may be weird or there is a bug
# find why t+7 is not performing over t+1 and t+14, find error acorss the stores
# 

FILES = {
    "baseline": "baseline_metrics.csv",
    "linear_regression": "linear_regression_metrics.csv",
    "lightgbm": "lightgbm_metrics.csv",
    "xgboost": "xgboost_metrics.csv",
    "flaml": "flaml_metrics.csv",
    "autogluon": "autogluon_metrics.csv",
    "pycaret": "pycaret_metrics.csv",
    "autogluon_quick":"autogluon_metrics_quick.csv",
    "flaml_quick": "flaml_metrics_quick.csv",
}

# close threshold for primary metrsic Wape to consider tie-breakers
CLOSE_EPS = 0.30

def get_mean_test_macro_wape(df):
    d = df[
        (df["split"] == "test")
        & (df["agg"] == "macro")
        & (df["horizon"].isin([1, 7, 14]))
    ]
    if d.empty:
        return None
    return float(d["WAPE"].mean())


def get_mean_val_macro_wape(df):
    d = df[
        (df["split"] == "val")
        & (df["agg"] == "macro")
        & (df["horizon"].isin([1, 7, 14]))
    ]
    if d.empty:
        return None
    return float(d["WAPE"].mean())


def get_abs_test_macro_bias_t7(df):
    d = df[(df["split"] == "test") & (df["agg"] == "macro") & (df["horizon"] == 7)]
    if d.empty:
        return None
    return abs(float(d["Bias"].mean()))


def get_mean_test_training_time(df):
    d = df[df["split"] == "test"]
    if d.empty:
        return None
    return float(d["training_time_seconds"].mean())


def sort_by_primary(row):
    return row["test_wape"]


def sort_by_tie(row):
    # tie-break logic: 1) lower Bias at t+7 (TEST)  2) lower training time (TEST)
    return (row["abs_bias_t7"], row["train_time"])


def display_model_name(raw_name: str) -> str:
    name = str(raw_name)
    if name.endswith("_log"):
        return name[:-4]
    return name


def mapped_display_name(file_key: str, all_models_in_file: list[str], raw_model_name: str) -> str:
    # If a file has one model variant, use the FILES map key (e.g., autogluon_quick).
    # If a file has multiple model variants, keep the model names from the file.
    if len(all_models_in_file) == 1:
        return str(file_key)
    return display_model_name(raw_model_name)


def _mean_or_none(df: pd.DataFrame, col: str):
    if col not in df.columns or df.empty:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def _std_or_none(df: pd.DataFrame, col: str):
    if col not in df.columns or df.empty:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.std(ddof=1))


def build_horizon_summary(rows_by_model_h: list[dict]) -> pd.DataFrame:
    if not rows_by_model_h:
        return pd.DataFrame()

    out = pd.DataFrame(rows_by_model_h)

    # rank by test WAPE within each horizon (1 is best, lower WAPE is better)
    out["rank_test_wape_within_horizon"] = (
        out.groupby("horizon")["test_wape"]
        .rank(method="dense", ascending=True)
    )

    out = out.sort_values(
        ["horizon", "rank_test_wape_within_horizon", "model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return out


def build_wape_matrix(h_summary: pd.DataFrame) -> pd.DataFrame:
    if h_summary.empty:
        return pd.DataFrame()

    index_col = "model_display" if "model_display" in h_summary.columns else "model"
    piv = h_summary.pivot_table(index=index_col, columns="horizon", values="test_wape", aggfunc="mean")

    # Stable, easy-to-scan column names.
    rename_map = {}
    for c in piv.columns.tolist():
        rename_map[c] = f"h{int(c)}_test_wape"
    piv = piv.rename(columns=rename_map)

    wanted = ["h1_test_wape", "h7_test_wape", "h14_test_wape"]
    for col in wanted:
        if col not in piv.columns:
            piv[col] = float("nan")

    piv["mean_test_wape_1_7_14"] = piv[wanted].mean(axis=1, skipna=True)
    piv = piv.sort_values(["mean_test_wape_1_7_14"], ascending=[True]).reset_index()
    if index_col in piv.columns and index_col != "model":
        piv = piv.rename(columns={index_col: "model"})

    piv["rank_mean_test_wape_1_7_14"] = piv["mean_test_wape_1_7_14"].rank(method="dense", ascending=True)

    return piv[["model", "h1_test_wape", "h7_test_wape", "h14_test_wape", "mean_test_wape_1_7_14", "rank_mean_test_wape_1_7_14"]]


def main():
    rows = []
    horizon_rows = []

    print("\nMODEL SELECTION\n")
    print("Primary metric:  TEST mean macro WAPE across horizons (1, 7, 14)")
    print("Tie-breakers:    1) lower Bias at t+7 (TEST)  2) lower training time (TEST)")
    print(f"Close threshold: {CLOSE_EPS:.2f} WAPE points\n")

    for model, filename in FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        model_names = (
            sorted(df["model"].dropna().astype(str).unique().tolist())
            if "model" in df.columns
            else [model]
        )
        if not model_names:
            model_names = [model]

        for model_name in model_names:
            d_model = df[df["model"].astype(str) == model_name].copy() if "model" in df.columns else df
            model_label = mapped_display_name(model, model_names, model_name)

            test_wape = get_mean_test_macro_wape(d_model)
            val_wape = get_mean_val_macro_wape(d_model)
            abs_bias_t7 = get_abs_test_macro_bias_t7(d_model)
            train_time = get_mean_test_training_time(d_model)

            if test_wape is None:
                continue

            rows.append({
                "raw_model": model_name,
                "model": model_name,
                "model_display": model_label,
                "test_wape": test_wape,
                "val_wape": val_wape if val_wape is not None else float("inf"),
                "abs_bias_t7": abs_bias_t7 if abs_bias_t7 is not None else float("inf"),
                "train_time": train_time if train_time is not None else float("inf"),
            })

            # Per-horizon compact summary from macro rows.
            d_macro = d_model[d_model["agg"] == "macro"].copy() if "agg" in d_model.columns else d_model.copy()
            if d_macro.empty or "horizon" not in d_macro.columns:
                continue

            horizons = sorted(pd.to_numeric(d_macro["horizon"], errors="coerce").dropna().astype(int).unique().tolist())
            for h in horizons:
                d_h = d_macro[pd.to_numeric(d_macro["horizon"], errors="coerce") == h].copy()
                d_test = d_h[d_h["split"] == "test"].copy() if "split" in d_h.columns else pd.DataFrame()
                d_val = d_h[d_h["split"] == "val"].copy() if "split" in d_h.columns else pd.DataFrame()

                horizon_rows.append({
                    "raw_model": model_name,
                    "model": model_name,
                    "model_display": model_label,
                    "horizon": h,
                    "test_wape": _mean_or_none(d_test, "WAPE"),
                    "test_mae": _mean_or_none(d_test, "MAE"),
                    "test_rmse": _mean_or_none(d_test, "RMSE"),
                    "test_bias": _mean_or_none(d_test, "Bias"),
                    "test_n": _mean_or_none(d_test, "N"),
                    "val_wape_mean": _mean_or_none(d_val, "WAPE"),
                    "val_wape_std": _std_or_none(d_val, "WAPE"),
                    "val_mae_mean": _mean_or_none(d_val, "MAE"),
                    "val_rmse_mean": _mean_or_none(d_val, "RMSE"),
                    "val_bias_mean": _mean_or_none(d_val, "Bias"),
                    "val_n_mean": _mean_or_none(d_val, "N"),
                    "test_training_time_seconds": _mean_or_none(d_test, "training_time_seconds"),
                })

    if not rows:
        print("No results found.")
        return

    # Summary table
    print("Summary (lower is better):")
    print(f"{'Model':<18} {'TestWAPE':>9} {'ValWAPE':>9} {'Bias@t+7':>11} {'Train(s)':>9}")
    print("-" * 60)

    rows_for_print = sorted(rows, key=sort_by_primary)
    for r in rows_for_print:
        print(
            f"{r['model_display']:<18} "
            f"{r['test_wape']:>9.4f} "
            f"{r['val_wape']:>9.4f} "
            f"{r['abs_bias_t7']:>11.4f} "
            f"{r['train_time']:>9.1f}"
        )

    # Tie-breaking logic
    best_primary = rows_for_print[0]["test_wape"]

    close_set = []
    for r in rows_for_print:
        if (r["test_wape"] - best_primary) <= CLOSE_EPS:
            close_set.append(r)

    if len(close_set) == 1:
        winner = close_set[0]
        print(f"\nWinner selected on primary metric only because no model within {CLOSE_EPS:.2f} WAPE points")
    else:
        close_sorted = sorted(close_set, key=sort_by_tie)
        winner = close_sorted[0]

        print(f"\nModels within {CLOSE_EPS:.2f} WAPE points primary metric close enough to consider tie-breakers")
        for i, r in enumerate(close_sorted, 1):
            print(
                f"  {i}. {r['model_display']:<18} "
                f"TestWAPE={r['test_wape']:.4f}  "
                f"Bias@t+7={r['abs_bias_t7']:.4f}  "
                f"Train={r['train_time']:.1f}s"
            )

    # Final winner summary
    print("\nFINAL DECISION")
    print("-------------")
    print("Winner:", winner["model_display"].upper())
    print(f"Primary (Test mean macro WAPE): {winner['test_wape']:.4f}")
    print(f"Tie-break 1 (Bias at t+7):    {winner['abs_bias_t7']:.4f}")
    print(f"Tie-break 2 (Training time):    {winner['train_time']:.1f}s")
    print(f"Extra (Val mean macro WAPE):    {winner['val_wape']:.4f}")
    print()

    # Write compact horizon summary outputs.
    h_summary = build_horizon_summary(horizon_rows)
    wape_matrix = build_wape_matrix(h_summary)

    if not h_summary.empty:
        if "model_display" in h_summary.columns:
            h_summary["model"] = h_summary["model_display"]
            h_summary = h_summary.drop(columns=["model_display"], errors="ignore")
        h_summary = h_summary.drop(columns=["raw_model"], errors="ignore")
        h_summary.to_csv(HORIZON_SUMMARY_CSV, index=False)
        print(f"Saved: {HORIZON_SUMMARY_CSV}")

    if not wape_matrix.empty:
        if "model_display" in wape_matrix.columns:
            wape_matrix["model"] = wape_matrix["model_display"]
            wape_matrix = wape_matrix.drop(columns=["model_display"], errors="ignore")
        wape_matrix = wape_matrix.drop(columns=["raw_model"], errors="ignore")
        wape_matrix.to_csv(HORIZON_WAPE_MATRIX_CSV, index=False)
        print(f"Saved: {HORIZON_WAPE_MATRIX_CSV}")


if __name__ == "__main__":
    main()
