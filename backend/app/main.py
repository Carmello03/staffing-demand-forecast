import csv
import hashlib
import io
import json
import os
from datetime import date, timedelta

import pandas as pd

from typing import Optional
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google.api_core import exceptions as gcloud_exceptions

from firebase_auth import get_uid_from_bearer
from db import (
    create_store, list_stores, upload_store_days_csv,
    get_store_status, get_store_days,
    get_store_doc, get_store_meta, get_store_meta_from_doc,
    get_store_forecast_cache, set_store_forecast_cache,
    update_store_profile, update_store_meta, get_store_profile,
    verify_store_owner, write_day_rows,
)
from forecast_service import predict_for_requested_horizon
from holiday_service import (
    build_holiday_context_by_horizon,
    normalize_country_iso,
    normalize_subdivision_code,
)

load_dotenv()
app = FastAPI()
FORECAST_HISTORY_DAYS = int(os.getenv("FORECAST_HISTORY_DAYS", "60"))
EXPLANATION_CACHE_VERSION = "exp3"
FORECAST_CACHE_VERSION = os.getenv("FORECAST_CACHE_VERSION", "fc3")

# Simple CORS setup for local frontend development.
cors_origins_env = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
)
cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoreIn(BaseModel):
    store_name: str

class CatchUpDayIn(BaseModel):
    date: str
    customers: float = Field(..., ge=0)
    open: int = Field(default=1, ge=0, le=1)
    promo: int = Field(default=0, ge=0, le=1)


class ProfileIn(BaseModel):
    customers_per_staff: Optional[float] = None
    country: Optional[str] = None
    holiday_subdivision: Optional[str] = None

class MetaIn(BaseModel):
    Store: Optional[int] = None
    StoreType: Optional[str] = None      # a|b|c|d
    Assortment: Optional[str] = None     # a=basic, b=extra, c=extended
    CompetitionDistance: Optional[float] = None
    CompetitionOpenSinceMonth: Optional[int] = None
    CompetitionOpenSinceYear: Optional[int] = None
    Promo2: Optional[int] = None         # 0 or 1
    Promo2SinceWeek: Optional[int] = None
    Promo2SinceYear: Optional[int] = None
    PromoInterval: Optional[str] = None  # e.g. "Feb,May,Aug,Nov" or None


class DayOverrideIn(BaseModel):
    horizon: int
    open: Optional[int] = None
    promo: Optional[int] = None


class ForecastRangeIn(BaseModel):
    k: int
    day_overrides: list[DayOverrideIn] = Field(default_factory=list)
    include_explanations: bool = False


def get_uid_or_dev(authorization: Optional[str]) -> str:
    # DEV_MODE=1 bypasses Firebase auth for local development only.
    # Never set this in production Cloud Run environment variables.
    if os.getenv("DEV_MODE") == "1":
        return os.getenv("DEV_UID", "dev-user")
    try:
        return get_uid_from_bearer(authorization)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def raise_api_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, (gcloud_exceptions.GoogleAPICallError, gcloud_exceptions.RetryError)):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail="Internal server error") from exc


def normalize_cached_forecasts(raw_forecasts: list) -> dict[int, dict]:
    by_horizon: dict[int, dict] = {}
    for item in raw_forecasts:
        if not isinstance(item, dict):
            continue
        horizon = item.get("horizon")
        if not isinstance(horizon, int):
            continue
        by_horizon[horizon] = item
    return by_horizon


def normalize_day_overrides(raw_overrides: list[DayOverrideIn] | None) -> dict[int, dict[str, int]]:
    normalized: dict[int, dict[str, int]] = {}
    if not raw_overrides:
        return normalized

    for item in raw_overrides:
        day = item.horizon
        if day < 1 or day > 14:
            raise ValueError("day_overrides.horizon must be between 1 and 14")

        day_override: dict[str, int] = {}
        if item.open is not None:
            if item.open not in [0, 1]:
                raise ValueError("day_overrides.open must be 0 or 1")
            day_override["open"] = int(item.open)

        if item.promo is not None:
            if item.promo not in [0, 1]:
                raise ValueError("day_overrides.promo must be 0 or 1")
            day_override["promo"] = int(item.promo)

        if day_override:
            normalized[day] = day_override

    return normalized


def canonical_day_overrides(day_overrides: dict[int, dict[str, int]]) -> list[dict]:
    canonical: list[dict] = []
    for day in sorted(day_overrides.keys()):
        item: dict = {"horizon": day}
        if "open" in day_overrides[day]:
            item["open"] = int(day_overrides[day]["open"])
        if "promo" in day_overrides[day]:
            item["promo"] = int(day_overrides[day]["promo"])
        canonical.append(item)
    return canonical


def hash_day_overrides(day_overrides: dict[int, dict[str, int]]) -> str:
    canonical = canonical_day_overrides(day_overrides)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_customers_per_staff(store_doc: dict) -> Optional[float]:
    profile = store_doc.get("profile")
    if not isinstance(profile, dict):
        return None

    raw_value = profile.get("customers_per_staff")
    if raw_value is None:
        return None

    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return None

    if parsed <= 0:
        return None
    return parsed


def parse_holiday_location(store_doc: dict) -> tuple[str, Optional[str]]:
    profile = store_doc.get("profile")
    if not isinstance(profile, dict):
        return "DE", None

    country_iso = normalize_country_iso(profile.get("country"))
    raw_subdivision = profile.get("holiday_subdivision") or profile.get("subdivision_code")
    subdivision_code = normalize_subdivision_code(raw_subdivision)
    return country_iso, subdivision_code


@app.post("/stores")
def post_store(payload: StoreIn, authorization: Optional[str] = Header(default=None)):
    try:
        uid = get_uid_or_dev(authorization)
        return create_store(uid, payload.store_name)
    except Exception as exc:
        raise_api_error(exc)


@app.get("/stores")
def get_stores(authorization: Optional[str] = Header(default=None)):
    try:
        uid = get_uid_or_dev(authorization)
        return list_stores(uid)
    except Exception as exc:
        raise_api_error(exc)


@app.post("/stores/{store_id}/upload")
def upload_store_csv(
    store_id: str,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    try:
        uid = get_uid_or_dev(authorization)
        verify_store_owner(store_id, uid)
        return upload_store_days_csv(store_id, file)
    except Exception as exc:
        raise_api_error(exc)


@app.get("/stores/{store_id}/status")
def store_status(store_id: str, authorization: Optional[str] = Header(default=None)):
    try:
        uid = get_uid_or_dev(authorization)
        verify_store_owner(store_id, uid)
        current_date = os.getenv("CURRENT_DATE", str(date.today()))
        return get_store_status(store_id, current_date)
    except Exception as exc:
        raise_api_error(exc)


@app.patch("/stores/{store_id}/profile")
def patch_store_profile(store_id: str, payload: ProfileIn, authorization: Optional[str] = Header(default=None)):
    try:
        uid = get_uid_or_dev(authorization)
        verify_store_owner(store_id, uid)
        updates = payload.dict(exclude_none=True)
        return update_store_profile(store_id, updates)
    except Exception as exc:
        raise_api_error(exc)


@app.get("/stores/{store_id}/profile")
def get_profile(store_id: str, authorization: Optional[str] = Header(default=None)):
    try:
        uid = get_uid_or_dev(authorization)
        verify_store_owner(store_id, uid)
        return {"store_id": store_id, "profile": get_store_profile(store_id)}
    except Exception as exc:
        raise_api_error(exc)

@app.get("/stores/{store_id}/meta")
def get_store_meta_endpoint(store_id: str, authorization: Optional[str] = Header(default=None)):
    try:
        uid = get_uid_or_dev(authorization)
        verify_store_owner(store_id, uid)
        meta = get_store_meta(store_id)
        return {"store_id": store_id, "store_meta": meta}
    except Exception as exc:
        raise_api_error(exc)


@app.patch("/stores/{store_id}/meta")
def patch_store_meta(store_id: str, payload: MetaIn, authorization: Optional[str] = Header(default=None)):
    try:
        uid = get_uid_or_dev(authorization)
        verify_store_owner(store_id, uid)
        updates = payload.dict(exclude_unset=True)  # keep explicit nulls
        if "StoreType" in updates and updates["StoreType"] is not None:
            if updates["StoreType"] not in ["a", "b", "c", "d"]:
                raise ValueError("StoreType must be one of: a, b, c, d")

        if "Assortment" in updates and updates["Assortment"] is not None:
            if updates["Assortment"] not in ["a", "b", "c"]:
                raise ValueError("Assortment must be one of: a (basic), b (extra), c (extended)")

        if "Promo2" in updates and updates["Promo2"] is not None:
            if updates["Promo2"] not in [0, 1]:
                raise ValueError("Promo2 must be 0 or 1")

        return update_store_meta(store_id, updates)

    except Exception as exc:
        raise_api_error(exc)
    
def build_forecast_range_response(
    store_id: str,
    k: int,
    day_overrides: dict[int, dict[str, int]],
    include_explanations: bool = False,
):
    if k < 1 or k > 14:
        raise ValueError("k must be between 1 and 14")

    store_doc = get_store_doc(store_id)
    meta = get_store_meta_from_doc(store_doc)
    if not meta:
        raise ValueError("Missing store_meta (seed it first).")
    customers_per_staff = parse_customers_per_staff(store_doc)
    holiday_country_iso, holiday_subdivision_code = parse_holiday_location(store_doc)

    issue_date = pd.to_datetime(os.getenv("CURRENT_DATE", str(date.today())))
    issue_date_str = str(issue_date.date())
    scenario_hash = hash_day_overrides(day_overrides)
    location_hash = hashlib.sha1(
        f"{holiday_country_iso}|{holiday_subdivision_code or ''}".encode("utf-8")
    ).hexdigest()[:8]
    explanation_flag = EXPLANATION_CACHE_VERSION if include_explanations else "exp0"
    cache_doc_id = (
        f"{issue_date_str}__{scenario_hash[:16]}__loc{location_hash}"
        f"__{FORECAST_CACHE_VERSION}__{explanation_flag}"
    )
    canonical_overrides = canonical_day_overrides(day_overrides)

    cached = get_store_forecast_cache(store_id, cache_doc_id)
    if isinstance(cached, dict):
        cached_forecasts = normalize_cached_forecasts(cached.get("forecasts", []))
        if all(day in cached_forecasts for day in range(1, 15)):
            return {
                "store_id": store_id,
                "issue_date": issue_date_str,
                "k": k,
                "cache_hit": True,
                "scenario_hash": scenario_hash,
                "day_overrides": canonical_overrides,
                "forecasts": [cached_forecasts[day] for day in range(1, k + 1)],
            }

    rows = get_store_days(store_id, recent_days=FORECAST_HISTORY_DAYS)
    if len(rows) < 35:
        raise ValueError("Not enough history uploaded (need ~35+ days).")

    hist = pd.DataFrame(rows)
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.sort_values("Date")
    holiday_context_by_h = build_holiday_context_by_horizon(
        issue_date=issue_date.date(),
        max_horizon=14,
        country_iso_code=holiday_country_iso,
        subdivision_code=holiday_subdivision_code,
    )

    full_forecasts = []
    for requested_h in range(1, 15):
        full_forecasts.append(
            predict_for_requested_horizon(
                store_id=store_id,
                requested_h=requested_h,
                current_issue_date=issue_date,
                hist_full=hist,
                store_doc=store_doc,
                meta=meta,
                customers_per_staff=customers_per_staff,
                day_overrides=day_overrides,
                holiday_context_by_h=holiday_context_by_h,
                include_explanation=include_explanations,
            )
        )

    set_store_forecast_cache(
        store_id=store_id,
        cache_doc_id=cache_doc_id,
        payload={
            "store_id": store_id,
            "issue_date": issue_date_str,
            "forecasts": full_forecasts,
        },
    )

    return {
        "store_id": store_id,
        "issue_date": issue_date_str,
        "k": k,
        "cache_hit": False,
        "scenario_hash": scenario_hash,
        "day_overrides": canonical_overrides,
        "forecasts": full_forecasts[:k],
    }


@app.get("/stores/{store_id}/forecast-range")
def forecast_store_range(
    store_id: str,
    k: int,
    include_explanations: bool = False,
    authorization: Optional[str] = Header(default=None),
):
    try:
        uid = get_uid_or_dev(authorization)
        verify_store_owner(store_id, uid)
        return build_forecast_range_response(
            store_id=store_id,
            k=k,
            day_overrides={},
            include_explanations=include_explanations,
        )
    except Exception as exc:
        raise_api_error(exc)


@app.post("/stores/{store_id}/forecast-range")
def forecast_store_range_with_scenario(
    store_id: str,
    payload: ForecastRangeIn,
    authorization: Optional[str] = Header(default=None),
):
    try:
        uid = get_uid_or_dev(authorization)
        verify_store_owner(store_id, uid)
        overrides = normalize_day_overrides(payload.day_overrides)
        return build_forecast_range_response(
            store_id=store_id,
            k=payload.k,
            day_overrides=overrides,
            include_explanations=payload.include_explanations,
        )
    except Exception as exc:
        raise_api_error(exc)


@app.get("/stores/{store_id}/catch-up-template")
def catch_up_template(store_id: str, authorization: Optional[str] = Header(default=None)):
    """Return a CSV pre-filled with missing dates so the manager can fill in
    Customers/Open/Promo and re-upload."""
    try:
        uid = get_uid_or_dev(authorization)
        verify_store_owner(store_id, uid)

        current_date_str = os.getenv("CURRENT_DATE", str(date.today()))
        current_dt = date.fromisoformat(current_date_str)
        status = get_store_status(store_id, current_date_str)

        last_str = status.get("last_uploaded_date")
        start_dt = date.fromisoformat(last_str) + timedelta(days=1) if last_str else current_dt

        missing = []
        d = start_dt
        while d <= current_dt:
            missing.append(d)
            d += timedelta(days=1)

        if not missing:
            raise HTTPException(status_code=400, detail="No missing dates - store is already up to date.")

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Date", "Customers", "Open", "Promo"])
        for d in missing:
            writer.writerow([d.isoformat(), "", 1, 0])

        filename = f"catch-up-{store_id[:8]}-{current_date_str}.csv"
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        raise_api_error(exc)


@app.post("/stores/{store_id}/catch-up")
def post_catch_up(
    store_id: str,
    days: list[CatchUpDayIn],
    authorization: Optional[str] = Header(default=None),
):
    """Accept a list of day rows and write them directly to Firestore."""
    try:
        uid = get_uid_or_dev(authorization)
        verify_store_owner(store_id, uid)

        if not days:
            raise ValueError("No days provided.")

        rows = []
        seen_dates: set[str] = set()
        for item in days:
            try:
                parsed_date = date.fromisoformat(item.date).isoformat()
            except ValueError as exc:
                raise ValueError("Each catch-up row date must be YYYY-MM-DD.") from exc

            if parsed_date in seen_dates:
                raise ValueError(f"Duplicate date in catch-up payload: {parsed_date}")
            seen_dates.add(parsed_date)

            open_value = int(item.open)
            promo_value = int(item.promo)
            customers_value = float(item.customers)
            if open_value == 0:
                customers_value = 0.0
                promo_value = 0

            rows.append(
                {
                    "date": parsed_date,
                    "customers": customers_value,
                    "open": open_value,
                    "promo": promo_value,
                }
            )
        n = write_day_rows(store_id, rows)
        return {"store_id": store_id, "rows_written": n}
    except Exception as exc:
        raise_api_error(exc)
