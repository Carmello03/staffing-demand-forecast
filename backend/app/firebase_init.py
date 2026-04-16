import os

import firebase_admin
from firebase_admin import credentials


def init_firebase_app() -> None:
    if firebase_admin._apps:
        return

    key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not key_path:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT env var not set")

    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)
