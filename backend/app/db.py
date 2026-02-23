import os
from datetime import date

import firebase_admin
import pandas as pd
from firebase_admin import credentials, firestore


STORE_META_FIELD = "store_meta"


def _init():
    if firebase_admin._apps:
        return
    key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not key_path:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT env var not set")
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)


def list_stores(uid: str):
    _init()
    db = firestore.client()
    docs = db.collection("stores").where("owner_uid", "==", uid).stream()

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
    })

    return {"store_id": doc_ref.id, "store_name": store_name}


def upload_store_days_csv(store_id: str, file) -> dict:
    """
    CSV columns required: Date, Customers, Open, Promo
    Writes to: stores/{store_id}/days/{YYYY-MM-DD}
    """
    _init()
    db = firestore.client()

    content = file.file.read()
    if not content:
        raise ValueError("Empty file")

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

    days_col = db.collection("stores").document(store_id).collection("days")

    batch = db.batch()
    n = 0

    for _, r in df.iterrows():
        date_str = r["Date"].date().isoformat()
        doc_ref = days_col.document(date_str)

        batch.set(doc_ref, {
            "date": date_str,
            "customers": float(r["Customers"]),
            "open": int(r["Open"]),
            "promo": int(r["Promo"]),
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        n += 1

        if n % 450 == 0:
            batch.commit()
            batch = db.batch()

    batch.commit()
    return {"store_id": store_id, "rows_written": n}


def get_store_status(store_id: str, current_date_str: str) -> dict:
    _init()
    db = firestore.client()
    days_col = db.collection("stores").document(store_id).collection("days")

    docs = list(days_col.stream())
    last_uploaded_date = max((d.id for d in docs), default=None)
    days_uploaded = len(docs)

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


def get_store_days(store_id: str) -> list[dict]:
    _init()
    db = firestore.client()
    days_col = db.collection("stores").document(store_id).collection("days")

    rows = []
    for d in days_col.stream():
        x = d.to_dict()
        rows.append({
            "Date": x.get("date", d.id),
            "Customers": x.get("customers", 0.0),
            "Open": x.get("open", 1),
            "Promo": x.get("promo", 0),
        })

    def _date_key(r: dict):
        return r["Date"]

    rows.sort(key=_date_key)
    return rows


def get_store_doc(store_id: str) -> dict:
    _init()
    db = firestore.client()
    snap = db.collection("stores").document(store_id).get()
    if not snap.exists:
        raise ValueError("Store not found")
    return snap.to_dict() or {}


def get_store_meta(store_id: str) -> dict:
    doc = get_store_doc(store_id)

    meta = doc.get(STORE_META_FIELD)
    if isinstance(meta, dict) and meta:
        return meta

    for legacy_name in ["rossmann_meta", "model_meta"]:
        legacy = doc.get(legacy_name)
        if isinstance(legacy, dict) and legacy:
            return legacy

    return {}


def update_store_profile(store_id: str, updates: dict) -> dict:
    _init()
    db = firestore.client()

    doc_ref = db.collection("stores").document(store_id)
    snap = doc_ref.get()
    if not snap.exists:
        raise ValueError("Store not found")

    current = snap.to_dict() or {}
    profile = current.get("profile")
    if not isinstance(profile, dict):
        profile = {}

    merged = dict(profile)
    for k, v in updates.items():
        merged[k] = v

    doc_ref.set({"profile": merged}, merge=True)
    return {"store_id": store_id, "profile": merged}