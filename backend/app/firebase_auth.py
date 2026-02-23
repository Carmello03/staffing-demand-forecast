import os
import firebase_admin
from firebase_admin import credentials, auth


def _init():
    if firebase_admin._apps:
        return

    key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not key_path:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT env var not set")

    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)


def get_uid_from_bearer(authorization):
    if not authorization:
        raise ValueError("Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ValueError("Authorization must be: Bearer <token>")

    token = parts[1]
    if not token:
        raise ValueError("Empty token")

    _init()
    decoded = auth.verify_id_token(token)
    return decoded.get("uid")