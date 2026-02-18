import os
import json
import pandas as pd

"""Purged k-fold time-series splits.

This replaces the earlier rolling/expanding split with *purged k-fold*
cross-validation as described by Lainder & Wolfinger (IJF 2022): each fold
holds out a contiguous validation block, while training uses all remaining
time periods **excluding an embargo (gap) of length MAX_HORIZON_DAYS** on both
sides of the validation block.

Why the embargo matters here:
  - Our targets are constructed as y(t+h). Without a gap, training labels from
    the last h days before validation would fall inside the validation window
    (leakage via overlapping forecasting intervals).
  - Our features include lags up to 28 days; training rows immediately after
    validation would use lag values from inside the validation window.

We keep a fixed final TEST window at the end of the dev timeline, and apply the
same embargo between TRAIN and TEST.
"""

DATA_PATH = "evaluation/data/processed/panel_train_clean.csv"
HOLDOUT_PATH = "evaluation/data/splits/holdout_stores.csv"
OUT_PATH = "evaluation/data/splits/time_splits_purged_kfold.json"

VAL_DAYS = 56
TEST_DAYS = 56
MAX_HORIZON_DAYS = 14
ROLLING_28_DAYS = 28

# Embargo / purge size. Paper recommends >= longest forecast horizon.
GAP_DAYS = ROLLING_28_DAYS

N_FOLDS = 3

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

VAL_ISSUE_DAYS = VAL_DAYS - MAX_HORIZON_DAYS
TEST_ISSUE_DAYS = TEST_DAYS - MAX_HORIZON_DAYS

# Final test end is the last date where the largest horizon has a target available
test_end = max_date - pd.Timedelta(days=MAX_HORIZON_DAYS)
test_start = test_end - pd.Timedelta(days=TEST_ISSUE_DAYS - 1)

train_global_end = test_start - pd.Timedelta(days=GAP_DAYS + 1)

# Validation folds live within the global training timeline.
last_val_end = train_global_end

val_folds = []

# Build non-overlapping validation blocks going backwards from the most recent
# pre-test date. Consecutive validation blocks are separated by GAP_DAYS.
cursor_end = last_val_end
for i in range(N_FOLDS):
    val_end = cursor_end
    val_start = val_end - pd.Timedelta(days=VAL_ISSUE_DAYS - 1)

    # Train ranges are the union of everything outside the validation block,
    # excluding an embargo on both sides.
    left_train_end = val_start - pd.Timedelta(days=GAP_DAYS + 1)
    right_train_start = val_end + pd.Timedelta(days=GAP_DAYS + 1)

    train_ranges = []
    if left_train_end >= min_date:
        train_ranges.append({"start": fmt(min_date), "end": fmt(left_train_end)})
    if right_train_start <= train_global_end:
        train_ranges.append({"start": fmt(right_train_start), "end": fmt(train_global_end)})

    # Store embargo ranges for transparency / thesis write-up.
    gap_ranges = []
    left_gap_start = val_start - pd.Timedelta(days=GAP_DAYS)
    left_gap_end = val_start - pd.Timedelta(days=1)
    if left_gap_end >= min_date:
        gap_ranges.append({"start": fmt(max(min_date, left_gap_start)), "end": fmt(left_gap_end)})

    right_gap_start = val_end + pd.Timedelta(days=1)
    right_gap_end = val_end + pd.Timedelta(days=GAP_DAYS)
    if right_gap_start <= train_global_end:
        gap_ranges.append({"start": fmt(right_gap_start), "end": fmt(min(train_global_end, right_gap_end))})

    val_folds.append({
        "fold": i + 1,
        "train_ranges": train_ranges,
        "gap_ranges": gap_ranges,
        "val": {"start": fmt(val_start), "end": fmt(val_end)},
    })

    # Move cursor earlier: leave a GAP_DAYS separation between validation blocks
    cursor_end = val_start - pd.Timedelta(days=GAP_DAYS + 1)

# Put folds in chronological order
val_folds = list(reversed(val_folds))

for j, fold in enumerate(val_folds, start=1):
    fold["fold"] = j

splits = {
    "train_global": {"start": fmt(min_date), "end": fmt(train_global_end)},
    "val_folds": val_folds,
    "test": {"start": fmt(test_start), "end": fmt(test_end)},
    "meta": {
        "VAL_DAYS": VAL_DAYS,
        "TEST_DAYS": TEST_DAYS,
        "GAP_DAYS": GAP_DAYS,
        "MAX_HORIZON_DAYS": MAX_HORIZON_DAYS,
        "N_FOLDS": N_FOLDS,
        "VAL_ISSUE_DAYS": VAL_ISSUE_DAYS,
        "TEST_ISSUE_DAYS": TEST_ISSUE_DAYS
    }
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(splits, f, indent=2)

print("Saved purged k-fold splits to:", OUT_PATH)
print("Test issue-date range:", splits["test"]["start"], "to", splits["test"]["end"])
for fold in splits["val_folds"]:
    tr = fold.get("train_ranges", [])
    tr_str = " + ".join([f"{r['start']}..{r['end']}" for r in tr]) if tr else "(empty)"
    print(f"Fold {fold['fold']}: train {tr_str} | val {fold['val']['start']}..{fold['val']['end']}")