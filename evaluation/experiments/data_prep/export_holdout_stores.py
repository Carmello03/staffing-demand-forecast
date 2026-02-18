import os
import random
import pandas as pd

IN_PATH = "evaluation/data/processed/panel_train_clean.csv"
HOLDOUT_PATH = "evaluation/data/splits/holdout_stores.csv"
OUT_DIR = "evaluation/data/holdout_stores"
UPLOAD_DIR = os.path.join(OUT_DIR, "uploads")

OUT_HOLDOUT_DEMO = os.path.join(OUT_DIR, "holdout_stores_demo.csv")
OUT_META = os.path.join(OUT_DIR, "holdout_store_meta.csv")
OUT_REF = os.path.join(OUT_DIR, "holdout_ref_store_date.csv")

# rules
MIN_UP_TO_DATE = 5
GAP_CHOICES = [1, 7, 14] 
SEED = 42

# Manager uploads
UPLOAD_COLS = ["Date", "Customers", "Open", "Promo"]

# Static store metadata
META_COLS = [
    "Store", "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"
]

# Per-store per-date reference
REF_COLS = ["Date", "Store", "DayOfWeek","StateHoliday", "SchoolHoliday"]


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    df = pd.read_csv(IN_PATH, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    hold = pd.read_csv(HOLDOUT_PATH)

    hold["Store"] = hold["Store"].astype(int)
    df["Store"] = df["Store"].astype(int)

    store_ids = hold["Store"].tolist()

    # assign gap days
    gaps = [0] * min(MIN_UP_TO_DATE, len(store_ids))
    gaps += [random.choice(GAP_CHOICES) for _ in range(len(store_ids) - len(gaps))]

    hold_demo = hold.copy()
    hold_demo["gap_days"] = gaps

    meta_rows = []
    ref_rows = []
    cutoff_dates = []

    for sid, gap in zip(store_ids, gaps):
        g = df[df["Store"] == sid].sort_values("Date").copy()
        if g.empty:
            cutoff_dates.append("")
            continue

        last_date = g["Date"].max()
        cutoff = last_date - pd.Timedelta(days=int(gap))
        cutoff_dates.append(cutoff.date().isoformat())

        # manager upload file
        upload_cols = [c for c in UPLOAD_COLS if c in g.columns]
        upload = g[g["Date"] <= cutoff][upload_cols].copy()
        upload["Date"] = upload["Date"].dt.date.astype(str)

        upload_cols = [c for c in UPLOAD_COLS if c in g.columns]
        upload = g[g["Date"] <= cutoff][upload_cols].copy()
        upload["Date"] = upload["Date"].dt.date.astype(str)

        if "Open" in upload.columns:
            upload["Open"] = pd.to_numeric(upload["Open"], errors="coerce").fillna(1).astype(int)

        if "Promo" in upload.columns:
            upload["Promo"] = pd.to_numeric(upload["Promo"], errors="coerce").fillna(0).astype(int)

        if "Customers" in upload.columns:
            upload["Customers"] = pd.to_numeric(upload["Customers"], errors="coerce").fillna(0.0)

        upload.to_csv(os.path.join(UPLOAD_DIR, f"store_{sid}.csv"), index=False)


        # Static store metadata
        meta_cols = [c for c in META_COLS if c in g.columns]
        meta_row = g[meta_cols].dropna(how="all").iloc[0].to_dict()
        meta_rows.append(meta_row)

        ref_cols = [c for c in REF_COLS if c in g.columns]
        ref = g[ref_cols].copy()
        ref["Date"] = ref["Date"].dt.date.astype(str)
        ref_rows.append(ref)

    hold_demo["cutoff_date"] = cutoff_dates
    hold_demo["is_uptodate"] = hold_demo["gap_days"] == 0
    hold_demo.to_csv(OUT_HOLDOUT_DEMO, index=False)

    if meta_rows:
        pd.DataFrame(meta_rows).drop_duplicates("Store").to_csv(OUT_META, index=False)

    if ref_rows:
        pd.concat(ref_rows, ignore_index=True).to_csv(OUT_REF, index=False)

    print("Done.")
    print("Uploads:", UPLOAD_DIR)
    print("Holdout demo manifest:", OUT_HOLDOUT_DEMO)
    print("Store meta:", OUT_META)
    print("Store-date reference:", OUT_REF)


if __name__ == "__main__":
    main()
