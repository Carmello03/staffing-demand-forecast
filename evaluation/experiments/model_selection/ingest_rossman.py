import os
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

train_path = os.path.join(RAW_DIR, "train.csv")
store_path = os.path.join(RAW_DIR, "store.csv")

print("train.csv exists:", os.path.exists(train_path))
print("store.csv exists:", os.path.exists(store_path))

# Loads train.csv and parses Date as a real datetime
train = pd.read_csv(
    train_path,
    low_memory=False,
    dtype={"StateHoliday": "string"},
    parse_dates=["Date"]
)

# Loads store.csv containing store-level features
store = pd.read_csv(store_path, low_memory=False)

print("Train shape:", train.shape)
print("Store shape:", store.shape)

# Merges store info into each store-day row
panel = train.merge(store, on="Store", how="left")
panel = panel.sort_values(["Store", "Date"]).reset_index(drop=True)

print("Merged panel shape:", panel.shape)

print("Missing values (top 15 columns):")
print(panel.isna().sum().sort_values(ascending=False).head(15))

# Save the merged panel for all next steps (splits, baselines, features, models)
out_path = os.path.join(OUT_DIR, "panel_train.csv")
panel.to_csv(out_path, index=False)

print("Saved:", out_path)
