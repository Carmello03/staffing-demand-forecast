# main.py
import os
import pandas as pd

from typing import Optional
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

from firebase_auth import get_uid_from_bearer
from db import (
    create_store, list_stores, upload_store_days_csv,
    get_store_status, get_store_days,
    get_store_doc, update_store_profile
)
from model import predict_one
from features import add_calendar_features, add_lag_roll_features, NUM_COLS, CAT_COLS


load_dotenv()
app = FastAPI()


class StoreIn(BaseModel):
    store_name: str


class ProfileIn(BaseModel):
    customers_per_staff: Optional[float] = None
    country: Optional[str] = None


def get_uid_or_dev(authorization: Optional[str]) -> str:
    if os.getenv("DEV_MODE") == "1":
        return os.getenv("DEV_UID", "dev-user")
    return get_uid_from_bearer(authorization)


def get_store_meta_from_doc(store_doc: dict) -> dict:
    meta = store_doc.get("store_meta")
    if isinstance(meta, dict) and meta:
        return meta
    
    for k in "store_meta":
        legacy = store_doc.get(k)
        if isinstance(legacy, dict) and legacy:
            return legacy

    return {}


@app.post("/stores")
def post_store(payload: StoreIn, authorization: Optional[str] = Header(default=None)):
    try:
        uid = get_uid_or_dev(authorization)
        return create_store(uid, payload.store_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stores")
def get_stores(authorization: Optional[str] = Header(default=None)):
    try:
        uid = get_uid_or_dev(authorization)
        return list_stores(uid)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/stores/{store_id}/upload")
def upload_store_csv(
    store_id: str,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    try:
        get_uid_or_dev(authorization)
        return upload_store_days_csv(store_id, file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stores/{store_id}/status")
def store_status(store_id: str, authorization: Optional[str] = Header(default=None)):
    try:
        get_uid_or_dev(authorization)
        current_date = os.getenv("CURRENT_DATE", "2015-07-31")
        return get_store_status(store_id, current_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.patch("/stores/{store_id}/profile")
def patch_store_profile(store_id: str, payload: ProfileIn, authorization: Optional[str] = Header(default=None)):
    try:
        get_uid_or_dev(authorization)
        updates = payload.dict(exclude_none=True)
        return update_store_profile(store_id, updates)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stores/{store_id}/forecast")
def forecast_store(store_id: str, h: int, authorization: Optional[str] = Header(default=None)):
    try:
        get_uid_or_dev(authorization)

        if h not in [1, 7, 14]:
            raise ValueError("h must be 1, 7, or 14")

        rows = get_store_days(store_id)
        if len(rows) < 35:
            raise ValueError("Not enough history uploaded (need ~35+ days).")

        hist = pd.DataFrame(rows)
        hist["Date"] = pd.to_datetime(hist["Date"])

        issue_date = pd.to_datetime(os.getenv("CURRENT_DATE", "2015-07-31"))
        target_date = issue_date + pd.Timedelta(days=h)

        target = pd.DataFrame([{
            "Date": target_date,
            "Customers": None,
            "Open": 1,
            "Promo": 0,
        }])

        df = pd.concat([hist, target], ignore_index=True)

        df = add_calendar_features(df)
        df = add_lag_roll_features(df)

        row = df[df["Date"] == target_date].copy().fillna(0)

        store_doc = get_store_doc(store_id)
        meta = get_store_meta_from_doc(store_doc)
        if not meta:
            raise ValueError("Missing store_meta (seed it first).")

        store_number = store_doc.get("demo_store_number")
        if store_number is None:
            store_number = meta.get("Store")
        if store_number is None:
            raise ValueError("Missing demo_store_number (and no Store in store_meta).")

        row["Store"] = int(store_number)
        row["StoreType"] = meta.get("StoreType", "d")
        row["Assortment"] = meta.get("Assortment", "a")
        row["StateHoliday"] = "0"
        row["PromoInterval"] = meta.get("PromoInterval", "")

        row["CompetitionDistance"] = float(meta.get("CompetitionDistance", 0.0) or 0.0)
        row["Promo2"] = float(meta.get("Promo2", 0.0) or 0.0)
        row["SchoolHoliday"] = 0

        X = row[NUM_COLS + CAT_COLS]
        yhat = predict_one(X, horizon=h)

        return {
            "store_id": store_id,
            "horizon": h,
            "issue_date": str(issue_date.date()),
            "target_date": str(target_date.date()),
            "prediction_customers": float(yhat),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))