import numpy as np
import pandas as pd

NUM_COLS = [
    "DayOfWeek", "month", "year", "weekofyear", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "Promo", "SchoolHoliday",
    "CompetitionDistance", "Promo2",
    "lag1", "lag7", "lag14",
    "roll7_mean", "roll14_mean", "roll28_mean",
]

CAT_COLS = [
    "Store", "StoreType", "Assortment", "StateHoliday", "PromoInterval",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Rossmann DayOfWeek is 1-7 (Mon=1)
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1

    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    # trig encodings (no lambda)
    two_pi = 2.0 * np.pi
    df["dow_sin"] = np.sin(two_pi * df["DayOfWeek"] / 7.0)
    df["dow_cos"] = np.cos(two_pi * df["DayOfWeek"] / 7.0)
    df["month_sin"] = np.sin(two_pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(two_pi * df["month"] / 12.0)

    return df


def add_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()

    df["lag1"] = df["Customers"].shift(1)
    df["lag7"] = df["Customers"].shift(7)
    df["lag14"] = df["Customers"].shift(14)

    # rolling means computed from past values only (shift first)
    past = df["Customers"].shift(1)
    df["roll7_mean"] = past.rolling(7).mean()
    df["roll14_mean"] = past.rolling(14).mean()
    df["roll28_mean"] = past.rolling(28).mean()

    return df