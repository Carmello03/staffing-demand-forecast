import os
import json
import pandas as pd

"""
- fixed final TEST window
- rolling validation folds (same length as val window)
"""

DATA_PATH = "data/processed/panel_train_clean.csv"
HOLDOUT_PATH = "data/splits/holdout_stores.csv"
OUT_PATH = "data/splits/time_splits_rolling.json"

VAL_DAYS = 56
TEST_DAYS = 56
MAX_HORIZON_DAYS = 14

N_FOLDS = 3
FOLD_STRIDE_DAYS = 28  # rolling step with overlapping folds

def fmt(d: pd.Timestamp) -> str:
    return d.strftime("%Y-%m-%d")

print("Loading panel to get date range...")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
holdout_df = pd.read_csv(HOLDOUT_PATH)

holdout_stores = set(holdout_df["Store"].tolist())
dev = df[~df["Store"].isin(holdout_stores)].copy()

min_date = dev["Date"].min()
max_date = dev["Date"].max()

print("All stores:", df["Store"].nunique())
print("Holdout stores:", len(holdout_stores))
print("Development stores:", dev["Store"].nunique())
print("Dev date range:", min_date.date(), "to", max_date.date())

EFFECTIVE_VAL_DAYS = VAL_DAYS - MAX_HORIZON_DAYS
EFFECTIVE_TEST_DAYS = TEST_DAYS - MAX_HORIZON_DAYS

# Final test end is the last date where the largest horizon has a target available
test_end = max_date - pd.Timedelta(days=MAX_HORIZON_DAYS)
test_start = test_end - pd.Timedelta(days=EFFECTIVE_TEST_DAYS - 1)

# Last validation fold
last_val_end = test_start - pd.Timedelta(days=1)

# Validation windows of length VAL_DAYS
# Generate folds rolling backwards from last_val_end
val_folds = []
for i in range(N_FOLDS):
    val_end = last_val_end - pd.Timedelta(days=i * FOLD_STRIDE_DAYS)
    val_start = val_end - pd.Timedelta(days=EFFECTIVE_VAL_DAYS - 1)

    train_start = min_date
    train_end = val_start - pd.Timedelta(days=1)

    val_folds.append({
        "fold": i + 1,
        "train": {"start": fmt(train_start), "end": fmt(train_end)},
        "val": {"start": fmt(val_start), "end": fmt(val_end)},
    })

# Put folds in chronological order
val_folds = list(reversed(val_folds))

splits = {
    "train_global": {"start": fmt(min_date), "end": val_folds[-1]["train"]["end"]},
    "val_folds": val_folds,
    "test": {"start": fmt(test_start), "end": fmt(test_end)},
    "meta": {
        "VAL_DAYS": VAL_DAYS,
        "TEST_DAYS": TEST_DAYS,
        "MAX_HORIZON_DAYS": MAX_HORIZON_DAYS,
        "N_FOLDS": N_FOLDS,
        "FOLD_STRIDE_DAYS": FOLD_STRIDE_DAYS
    }
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(splits, f, indent=2)

print("Saved rolling splits to:", OUT_PATH)
print("Test issue-date range:", splits["test"]["start"], "to", splits["test"]["end"])
for fold in splits["val_folds"]:
    print(
        f'Fold {fold["fold"]}: train {fold["train"]["start"]}..{fold["train"]["end"]} | '
        f'val {fold["val"]["start"]}..{fold["val"]["end"]}'
    )