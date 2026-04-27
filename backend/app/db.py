import os
from datetime import date

import pandas as pd
from firebase_admin import firestore
from firebase_init import init_firebase_app
from holiday_service import (
    build_holiday_context_by_date_range,
    normalize_country_iso,
    normalize_subdivision_code,
)


STORE_META_FIELD = "store_meta"
FIRESTORE_TIMEOUT_SECONDS = float(os.getenv("FIRESTORE_TIMEOUT_SECONDS", "20"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)


def _firestore_call_kwargs() -> dict:
    return {"retry": None, "timeout": FIRESTORE_TIMEOUT_SECONDS}


def _init():
    init_firebase_app()


def _read_upload_with_size_limit(file_obj) -> bytes:
    content = file_obj.file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise ValueError("Empty file")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large. Max upload size is {MAX_UPLOAD_MB:g} MB.")
    return content


def list_stores(uid: str):
    _init()
    db = firestore.client()
    docs = db.collection("stores").where("owner_uid", "==", uid).stream(**_firestore_call_kwargs())

    out = []
    for d in docs:
        x = d.to_dict()
        out.append({"store_id": d.id, "store_name": x.get("store_name", "")})
    return out


def create_store(uid: str, store_name: str):
    _init()
    db = firestore.client()

    doc_ref = db.collection("stores").document()
    doc_ref.set({
        "owner_uid": uid,
        "store_name": store_name,
        "created_at": firestore.SERVER_TIMESTAMP,
    }, **_firestore_call_kwargs())

    return {"store_id": doc_ref.id, "store_name": store_name}


def upload_store_days_csv(store_id: str, file) -> dict:
    """
    CSV columns required: Date, Customers, Open, Promo
    Writes to: stores/{store_id}/days/{YYYY-MM-DD}
    """
    _init()
    db = firestore.client()

    content = _read_upload_with_size_limit(file)

    df = pd.read_csv(pd.io.common.BytesIO(content))

    required = ["Date", "Customers", "Open", "Promo"]
    for c in required:
        if c not in df.columns:
            raise ValueError("Missing column: " + c)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df["Customers"] = pd.to_numeric(df["Customers"], errors="coerce").fillna(0).astype(float)
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce").fillna(1).astype(int)
    df["Promo"] = pd.to_numeric(df["Promo"], errors="coerce").fillna(0).astype(int)
    if (df["Customers"] < 0).any():
        raise ValueError("Customers column cannot contain negative values.")
    if not df["Open"].isin([0, 1]).all():
        raise ValueError("Open column values must be 0 or 1.")
    if not df["Promo"].isin([0, 1]).all():
        raise ValueError("Promo column values must be 0 or 1.")
    df.loc[df["Open"] == 0, "Customers"] = 0.0
    df.loc[df["Open"] == 0, "Promo"] = 0

    store_ref = db.collection("stores").document(store_id)
    store_snap = store_ref.get(**_firestore_call_kwargs())
    if not store_snap.exists:
        raise ValueError("Store not found")

    store_doc = store_snap.to_dict() or {}
    profile = store_doc.get("profile") if isinstance(store_doc.get("profile"), dict) else {}
    country_iso = normalize_country_iso(profile.get("country"))
    subdivision_code = normalize_subdivision_code(
        profile.get("holiday_subdivision") or profile.get("subdivision_code")
    )

    holiday_by_date: dict[str, dict] = {}
    if not df.empty:
        upload_start = df["Date"].min().date()
        upload_end = df["Date"].max().date()
        holiday_by_date = build_holiday_context_by_date_range(
            start_date=upload_start,
            end_date=upload_end,
            country_iso_code=country_iso,
            subdivision_code=subdivision_code,
        )

    days_col = db.collection("stores").document(store_id).collection("days")

    batch = db.batch()
    n = 0

    for _, r in df.iterrows():
        date_str = r["Date"].date().isoformat()
        doc_ref = days_col.document(date_str)
        holiday = holiday_by_date.get(
            date_str,
            {
                "state_holiday": "0",
                "school_holiday": 0,
                "public_holiday_name": None,
                "school_holiday_name": None,
            },
        )

        batch.set(doc_ref, {
            "date": date_str,
            "customers": float(r["Customers"]),
            "open": int(r["Open"]),
            "promo": int(r["Promo"]),
            "state_holiday": str(holiday.get("state_holiday", "0")),
            "school_holiday": int(holiday.get("school_holiday", 0)),
            "public_holiday_name": holiday.get("public_holiday_name"),
            "school_holiday_name": holiday.get("school_holiday_name"),
            "holiday_country": country_iso,
            "holiday_subdivision": subdivision_code,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        n += 1

        if n % 450 == 0:
            batch.commit(**_firestore_call_kwargs())
            batch = db.batch()

    batch.commit(**_firestore_call_kwargs())
    clear_store_forecast_cache(store_id)
    return {"store_id": store_id, "rows_written": n}


def get_store_status(store_id: str, current_date_str: str) -> dict:
    _init()
    db = firestore.client()
    days_col = db.collection("stores").document(store_id).collection("days")

    latest_docs = list(
        days_col.order_by("__name__", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream(**_firestore_call_kwargs())
    )
    last_uploaded_date = latest_docs[0].id if latest_docs else None

    count_results = days_col.count().get(**_firestore_call_kwargs())
    days_uploaded = 0
    if count_results:
        # Firestore returns nested aggregation results: [[AggregationResult(...)]]
        first_row = count_results[0]
        if first_row:
            days_uploaded = int(first_row[0].value)

    gap_days = None
    ready_to_forecast = False

    if last_uploaded_date:
        a = date.fromisoformat(last_uploaded_date)
        b = date.fromisoformat(current_date_str)
        gap_days = (b - a).days
        if gap_days < 0:
            gap_days = 0
        ready_to_forecast = (gap_days == 0)

    return {
        "store_id": store_id,
        "current_date": current_date_str,
        "last_uploaded_date": last_uploaded_date,
        "gap_days": gap_days,
        "ready_to_forecast": ready_to_forecast,
        "days_uploaded": days_uploaded,
    }


def get_store_days(store_id: str, recent_days: int | None = None) -> list[dict]:
    _init()
    db = firestore.client()
    days_col = db.collection("stores").document(store_id).collection("days")

    query = days_col
    if recent_days is not None:
        if recent_days <= 0:
            return []
        query = query.order_by("__name__", direction=firestore.Query.DESCENDING).limit(recent_days)

    rows = []
    for d in query.stream(**_firestore_call_kwargs()):
        x = d.to_dict()
        rows.append({
            "Date": x.get("date", d.id),
            "Customers": x.get("customers", 0.0),
            "Open": x.get("open", 1),
            "Promo": x.get("promo", 0),
        })

    rows.sort(key=lambda r: r["Date"])
    return rows


def get_store_forecast_cache(store_id: str, cache_doc_id: str) -> dict | None:
    _init()
    db = firestore.client()
    snap = (
        db.collection("stores")
        .document(store_id)
        .collection("forecasts")
        .document(cache_doc_id)
        .get(**_firestore_call_kwargs())
    )
    if not snap.exists:
        return None
    return snap.to_dict() or {}


def set_store_forecast_cache(store_id: str, cache_doc_id: str, payload: dict) -> None:
    _init()
    db = firestore.client()
    (
        db.collection("stores")
        .document(store_id)
        .collection("forecasts")
        .document(cache_doc_id)
        .set(payload, merge=False, **_firestore_call_kwargs())
    )


def clear_store_forecast_cache(store_id: str) -> int:
    _init()
    db = firestore.client()
    forecasts_col = db.collection("stores").document(store_id).collection("forecasts")
    docs = list(forecasts_col.stream(**_firestore_call_kwargs()))
    if not docs:
        return 0

    batch = db.batch()
    deleted = 0
    for snap in docs:
        batch.delete(snap.reference)
        deleted += 1
        if deleted % 450 == 0:
            batch.commit(**_firestore_call_kwargs())
            batch = db.batch()

    if deleted % 450 != 0:
        batch.commit(**_firestore_call_kwargs())

    return deleted


def get_store_doc(store_id: str) -> dict:
    _init()
    db = firestore.client()
    snap = db.collection("stores").document(store_id).get(**_firestore_call_kwargs())
    if not snap.exists:
        raise ValueError("Store not found")
    return snap.to_dict() or {}


def write_day_rows(store_id: str, rows: list[dict]) -> int:
    """
    Write a list of day dicts directly (no CSV parsing).
    Each dict must have: date (YYYY-MM-DD), customers, open, promo.
    Reuses the same holiday lookup and Firestore batch logic as upload_store_days_csv.
    """
    _init()
    db = firestore.client()

    if not rows:
        return 0

    store_ref = db.collection("stores").document(store_id)
    snap = store_ref.get(**_firestore_call_kwargs())
    if not snap.exists:
        raise ValueError("Store not found")

    store_doc = snap.to_dict() or {}
    profile = store_doc.get("profile") if isinstance(store_doc.get("profile"), dict) else {}
    country_iso = normalize_country_iso(profile.get("country"))
    subdivision_code = normalize_subdivision_code(
        profile.get("holiday_subdivision") or profile.get("subdivision_code")
    )

    parsed_dates: list[date] = []
    for row in rows:
        date_raw = str(row.get("date", "")).strip()
        if not date_raw:
            raise ValueError("Each catch-up row must include a date.")
        try:
            parsed_dates.append(date.fromisoformat(date_raw))
        except ValueError as exc:
            raise ValueError(f"Invalid date format: {date_raw}. Expected YYYY-MM-DD.") from exc

    holiday_by_date = build_holiday_context_by_date_range(
        start_date=min(parsed_dates),
        end_date=max(parsed_dates),
        country_iso_code=country_iso,
        subdivision_code=subdivision_code,
    )

    days_col = db.collection("stores").document(store_id).collection("days")
    batch = db.batch()
    n = 0

    for row, parsed_date in zip(rows, parsed_dates):
        date_str = parsed_date.isoformat()

        open_value = int(row.get("open", 1))
        promo_value = int(row.get("promo", 0))
        if open_value not in [0, 1]:
            raise ValueError("open must be 0 or 1.")
        if promo_value not in [0, 1]:
            raise ValueError("promo must be 0 or 1.")

        customers_value = float(row.get("customers", 0))
        if customers_value < 0:
            raise ValueError("customers must be >= 0.")
        if open_value == 0:
            customers_value = 0.0
            promo_value = 0

        holiday = holiday_by_date.get(date_str, {
            "state_holiday": "0", "school_holiday": 0,
            "public_holiday_name": None, "school_holiday_name": None,
        })
        batch.set(days_col.document(date_str), {
            "date": date_str,
            "customers": customers_value,
            "open": open_value,
            "promo": promo_value,
            "state_holiday": str(holiday.get("state_holiday", "0")),
            "school_holiday": int(holiday.get("school_holiday", 0)),
            "public_holiday_name": holiday.get("public_holiday_name"),
            "school_holiday_name": holiday.get("school_holiday_name"),
            "holiday_country": country_iso,
            "holiday_subdivision": subdivision_code,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        n += 1
        if n % 450 == 0:
            batch.commit(**_firestore_call_kwargs())
            batch = db.batch()

    batch.commit(**_firestore_call_kwargs())
    clear_store_forecast_cache(store_id)
    return n


def verify_store_owner(store_id: str, uid: str) -> None:
    """Raises ValueError if the store does not exist or does not belong to uid."""
    doc = get_store_doc(store_id)
    if doc.get("owner_uid") != uid:
        raise ValueError("Store not found")


def get_store_meta_from_doc(doc: dict) -> dict:
    if not isinstance(doc, dict):
        return {}

    meta = doc.get(STORE_META_FIELD)
    if isinstance(meta, dict) and meta:
        return meta

    for legacy_name in ["rossmann_meta", "model_meta"]:
        legacy = doc.get(legacy_name)
        if isinstance(legacy, dict) and legacy:
            return legacy

    return {}


def get_store_meta(store_id: str) -> dict:
    doc = get_store_doc(store_id)
    return get_store_meta_from_doc(doc)

def update_store_meta(store_id: str, updates: dict) -> dict:
    _init()
    db = firestore.client()

    doc_ref = db.collection("stores").document(store_id)
    snap = doc_ref.get(**_firestore_call_kwargs())
    if not snap.exists:
        raise ValueError("Store not found")

    current = snap.to_dict() or {}
    meta = current.get(STORE_META_FIELD)
    if not isinstance(meta, dict):
        meta = {}

    merged = dict(meta)
    for k, v in updates.items():
        merged[k] = v

    doc_ref.set({STORE_META_FIELD: merged}, merge=True, **_firestore_call_kwargs())
    clear_store_forecast_cache(store_id)

    return {"store_id": store_id, "store_meta": merged}

def get_store_profile(store_id: str) -> dict:
    doc = get_store_doc(store_id)
    profile = doc.get("profile")
    if isinstance(profile, dict):
        return profile
    return {}

def update_store_profile(store_id: str, updates: dict) -> dict:
    _init()
    db = firestore.client()

    doc_ref = db.collection("stores").document(store_id)
    snap = doc_ref.get(**_firestore_call_kwargs())
    if not snap.exists:
        raise ValueError("Store not found")

    current = snap.to_dict() or {}
    profile = current.get("profile")
    if not isinstance(profile, dict):
        profile = {}

    merged = dict(profile)
    for k, v in updates.items():
        merged[k] = v

    doc_ref.set({"profile": merged}, merge=True, **_firestore_call_kwargs())
    clear_store_forecast_cache(store_id)
    return {"store_id": store_id, "profile": merged}
