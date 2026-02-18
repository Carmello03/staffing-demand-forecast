import os
import pandas as pd

IN_PATH = "evaluation/data/processed/panel_train_clean.csv"  # Input data path
OUT_DIR = "evaluation/data/splits"  # Output directory
OUT_PATH = os.path.join(OUT_DIR, "holdout_stores.csv")  # Output file path

SEED = 42
N_PER_BAND = 5

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH, parse_dates=["Date"], dtype={"StateHoliday": "string"})

print("Loaded rows:", df.shape[0])
print("Unique stores:", df["Store"].nunique())

 # Get average customers per open day for each store
open_df = df[df["Open"] == 1]
mean_customers = open_df.groupby("Store")["Customers"].mean().reset_index()
mean_customers = mean_customers.rename(columns={"Customers": "mean_customers_open"})

 # Show min/median/max of mean customers
print("Mean Customers (open days) min/median/max:",
      mean_customers["mean_customers_open"].min(),
      mean_customers["mean_customers_open"].median(),
      mean_customers["mean_customers_open"].max())

 # Split stores into 3 groups by customer count
q1 = mean_customers["mean_customers_open"].quantile(1/3)
q2 = mean_customers["mean_customers_open"].quantile(2/3)

bands = []
for _, row in mean_customers.iterrows():
    val = row["mean_customers_open"]
    if val <= q1:
        bands.append("small")
    elif val <= q2:
        bands.append("medium")
    else:
        bands.append("large")

mean_customers["band"] = bands

 # Pick 5 stores from each group randomly
holdout = []
for band_name in ["small", "medium", "large"]:
    band_stores = mean_customers[mean_customers["band"] == band_name]["Store"]
    band_stores = band_stores.sort_values()

    n_take = N_PER_BAND
    if band_stores.shape[0] < n_take:
        n_take = band_stores.shape[0]

    sampled = band_stores.sample(n=n_take, random_state=SEED)
    for store_id in sampled.tolist():
        holdout.append({"Store": int(store_id), "band": band_name})

holdout_df = pd.DataFrame(holdout).sort_values(["band", "Store"]).reset_index(drop=True)

 # Show how many stores were picked from each group
print("Holdout stores selected:", holdout_df.shape[0])
print(holdout_df.groupby("band")["Store"].count())

holdout_df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
