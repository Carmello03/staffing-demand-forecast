import math
from typing import Optional

import pandas as pd

from features import add_calendar_features, add_lag_roll_features, NUM_COLS, CAT_COLS
from model import predict_one, explain_one


def resolve_source_horizon(target_h: int) -> int:
    if target_h == 1:
        return 1
    if 2 <= target_h <= 7:
        return 7
    if 8 <= target_h <= 14:
        return 14
    raise ValueError("h must be between 1 and 14")


def build_forecast_feature_row(
    hist_full: pd.DataFrame,
    target_date: pd.Timestamp,
    source_issue_date: pd.Timestamp,
    store_doc: dict,
    meta: dict,
    open_value: int = 1,
    promo_value: int = 0,
    state_holiday: str = "0",
    school_holiday: int = 0,
) -> pd.DataFrame:
    hist = hist_full[hist_full["Date"] <= source_issue_date].copy()
    if len(hist) < 35:
        raise ValueError(
            f"Not enough history uploaded before source issue date {source_issue_date.date()} (need ~35+ days)."
        )

    target = pd.DataFrame([{
        "Date": target_date,
        "Customers": None,
        "Open": int(open_value),
        "Promo": int(promo_value),
    }])

    df = pd.concat([hist, target], ignore_index=True)
    df = add_calendar_features(df)
    df = add_lag_roll_features(df)

    row = df[df["Date"] == target_date].copy().fillna(0)
    if row.empty:
        raise ValueError("Unable to build forecast feature row.")

    store_number = store_doc.get("demo_store_number")
    if store_number is None:
        store_number = meta.get("Store")
    if store_number is None:
        raise ValueError("Missing Store number in store_meta (field: Store).")

    row["Store"] = int(store_number)
    row["StoreType"] = meta.get("StoreType", "d")
    row["Assortment"] = meta.get("Assortment", "a")
    row["StateHoliday"] = str(state_holiday or "0")
    row["PromoInterval"] = meta.get("PromoInterval", "")
    row["CompetitionDistance"] = float(meta.get("CompetitionDistance", 0.0) or 0.0)
    row["Promo2"] = float(meta.get("Promo2", 0.0) or 0.0)
    row["SchoolHoliday"] = int(school_holiday)
    return row


def predict_for_requested_horizon(
    store_id: str,
    requested_h: int,
    current_issue_date: pd.Timestamp,
    hist_full: pd.DataFrame,
    store_doc: dict,
    meta: dict,
    customers_per_staff: Optional[float] = None,
    day_overrides: Optional[dict[int, dict[str, int]]] = None,
    holiday_context_by_h: Optional[dict[int, dict]] = None,
    include_explanation: bool = False,
) -> dict:
    source_model_h = resolve_source_horizon(requested_h)
    backshift_days = source_model_h - requested_h
    source_issue_date = current_issue_date - pd.Timedelta(days=backshift_days)
    target_date = current_issue_date + pd.Timedelta(days=requested_h)
    override = (day_overrides or {}).get(requested_h, {})
    open_value = int(override.get("open", 1))
    promo_value = int(override.get("promo", 0))
    holiday_context = (holiday_context_by_h or {}).get(requested_h, {})
    state_holiday = str(holiday_context.get("state_holiday", "0"))
    school_holiday = int(holiday_context.get("school_holiday", 0))
    public_holiday_name = holiday_context.get("public_holiday_name")
    school_holiday_name = holiday_context.get("school_holiday_name")

    row = build_forecast_feature_row(
        hist_full=hist_full,
        target_date=target_date,
        source_issue_date=source_issue_date,
        store_doc=store_doc,
        meta=meta,
        open_value=open_value,
        promo_value=promo_value,
        state_holiday=state_holiday,
        school_holiday=school_holiday,
    )
    X = row[NUM_COLS + CAT_COLS]
    # Business rule: if store is planned closed, demand and staff must be zero.
    if open_value == 0:
        yhat = 0.0
        explanation = None
    else:
        yhat = predict_one(X, horizon=source_model_h)
        explanation = explain_one(X, source_model_h) if include_explanation else None

    suggested_staff = None
    if open_value == 0:
        suggested_staff = 0
    elif customers_per_staff is not None and customers_per_staff > 0:
        suggested_staff = int(math.ceil(max(float(yhat), 0.0) / customers_per_staff))

    return {
        "store_id": store_id,
        "horizon": requested_h,
        "issue_date": str(current_issue_date.date()),
        "target_date": str(target_date.date()),
        "weekday": str(target_date.day_name()),
        "planned_open": open_value,
        "planned_promo": promo_value,
        "prediction_customers": float(yhat),
        "customers_per_staff": customers_per_staff,
        "suggested_staff": suggested_staff,
        "state_holiday": state_holiday,
        "school_holiday": school_holiday,
        "public_holiday_name": public_holiday_name,
        "school_holiday_name": school_holiday_name,
        "explanation": explanation,
        "source_model_h": source_model_h,
        "source_issue_date": str(source_issue_date.date()),
    }
