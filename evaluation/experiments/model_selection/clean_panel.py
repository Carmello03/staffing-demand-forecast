import os
import pandas as pd


IN_PATH = "data/processed/panel_train.csv"  # Input file
OUT_PATH = "data/processed/panel_train_clean.csv"  # Output file


df = pd.read_csv(IN_PATH, parse_dates=["Date"], dtype={"StateHoliday": "string"})

print("Loaded panel shape:", df.shape)
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("Unique stores:", df["Store"].nunique())

# Fill missing values for store features
df["PromoInterval"] = df["PromoInterval"].fillna("None")
df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0).astype(int)
df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0).astype(int)
df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0).astype(int)
df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0).astype(int)

median_comp_dist = df["CompetitionDistance"].median()
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(median_comp_dist)

print("Filled missing store-feature values.")
print("CompetitionDistance median used:", median_comp_dist)

# Make sure every store has a row for every date
all_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
stores = sorted(df["Store"].unique())

print("Expected full rows if continuous:", len(stores) * len(all_dates))

parts = []
for store_id in stores:
    g = df[df["Store"] == store_id].copy()
    g = g.set_index("Date").sort_index()

    g = g.reindex(all_dates)

    g["Store"] = store_id

    # Set DayOfWeek e.g 1=Monday, 7=Sunday
    g["DayOfWeek"] = g.index.dayofweek + 1

    # Fill missing days as closed
    g["Open"] = g["Open"].fillna(0).astype(int)
    g["Sales"] = g["Sales"].fillna(0)
    g["Customers"] = g["Customers"].fillna(0)
    g["Promo"] = g["Promo"].fillna(0).astype(int)
    g["SchoolHoliday"] = g["SchoolHoliday"].fillna(0).astype(int)
    g["StateHoliday"] = g["StateHoliday"].fillna("0")

    # Fill other columns using values from the same store
    g = g.ffill().bfill()

    g = g.reset_index().rename(columns={"index": "Date"})
    parts.append(g)

clean = pd.concat(parts, ignore_index=True)
clean = clean.sort_values(["Store", "Date"]).reset_index(drop=True)
clean["PromoInterval"] = clean["PromoInterval"].fillna("None")

print("Clean continuous panel shape:", clean.shape)
print("Duplicate Store-Date rows:", clean.duplicated(subset=["Store", "Date"]).sum())

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
clean.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
