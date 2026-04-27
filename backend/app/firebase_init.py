import os
import json

import firebase_admin
from firebase_admin import credentials


def init_firebase_app() -> None:
    if firebase_admin._apps:
        return

    key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if key_path:
        try:
            # Supports both:
            # 1) local file path, e.g. "backend/app/service-account.json"
            # 2) raw JSON string from Secret Manager env var
            if key_path.lstrip().startswith("{"):
                cred = credentials.Certificate(json.loads(key_path))
            else:
                cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            return
        except Exception as exc:
            raise ValueError(
                "Invalid FIREBASE_SERVICE_ACCOUNT. Provide a valid service-account JSON file path "
                "or a valid JSON string."
            ) from exc

    # Cloud Run/Cloud Build can use Application Default Credentials (ADC)
    # from the attached service account, so no key file is required.
    try:
        firebase_admin.initialize_app()
    except Exception as exc:
        raise ValueError(
            "Firebase init failed. Set FIREBASE_SERVICE_ACCOUNT for local use, "
            "or configure Cloud Run service account permissions for ADC."
        ) from exc
