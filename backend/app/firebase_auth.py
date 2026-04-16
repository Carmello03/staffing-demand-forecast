from firebase_admin import auth
from firebase_init import init_firebase_app


def _init():
    init_firebase_app()


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
