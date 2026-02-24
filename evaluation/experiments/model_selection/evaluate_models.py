import pandas as pd
import os

RESULTS_DIR = r"evaluation\experiments\model_selection\results"

FILES = {
    "baseline": "baseline_metrics.csv",
    "linear_regression": "linear_regression_metrics.csv",
    "lightgbm": "lightgbm_metrics.csv",
    "xgboost": "xgboost_metrics.csv",
    "flaml": "flaml_metrics.csv",
    "autogluon": "autogluon_metrics.csv",
    "pycaret": "pycaret_metrics.csv",
}

# close threshold for primary metric Wape to consider tie-breakers
CLOSE_EPS = 0.30


def get_mean_test_macro_wape(df):
    d = df[(df["split"] == "test") & (df["agg"] == "macro")]
    if d.empty:
        return None
    return float(d["WAPE"].mean())


def get_mean_val_macro_wape(df):
    d = df[(df["split"] == "val") & (df["agg"] == "macro")]
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


def main():
    rows = []

    print("\nMODEL SELECTION\n")
    print("Primary metric:  TEST mean macro WAPE across horizons (1, 7, 14)")
    print("Tie-breakers:    1) lower Bias at t+7 (TEST)  2) lower training time (TEST)")
    print(f"Close threshold: {CLOSE_EPS:.2f} WAPE points\n")

    for model, filename in FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        test_wape = get_mean_test_macro_wape(df)
        val_wape = get_mean_val_macro_wape(df)
        abs_bias_t7 = get_abs_test_macro_bias_t7(df)
        train_time = get_mean_test_training_time(df)

        if test_wape is None:
            continue

        rows.append({
            "model": model,
            "test_wape": test_wape,
            "val_wape": val_wape if val_wape is not None else float("inf"),
            "abs_bias_t7": abs_bias_t7 if abs_bias_t7 is not None else float("inf"),
            "train_time": train_time if train_time is not None else float("inf"),
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
            f"{r['model']:<18} "
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
                f"  {i}. {r['model']:<18} "
                f"TestWAPE={r['test_wape']:.4f}  "
                f"Bias@t+7={r['abs_bias_t7']:.4f}  "
                f"Train={r['train_time']:.1f}s"
            )

    # Final winner summary
    print("\nFINAL DECISION")
    print("-------------")
    print("Winner:", winner["model"].upper())
    print(f"Primary (Test mean macro WAPE): {winner['test_wape']:.4f}")
    print(f"Tie-break 1 (Bias at t+7):    {winner['abs_bias_t7']:.4f}")
    print(f"Tie-break 2 (Training time):    {winner['train_time']:.1f}s")
    print(f"Extra (Val mean macro WAPE):    {winner['val_wape']:.4f}")
    print()


if __name__ == "__main__":
    main()