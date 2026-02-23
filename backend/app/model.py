import os
import joblib
import numpy as np


ARTIFACT_DIR = os.getenv(
    "MODEL_ARTIFACT_DIR",
    r"..\..\evaluation\experiments\model_selection\artifacts"
)

_models = None


def load_models():
    global _models
    if _models is not None:
        return

    _models = {}
    for h in [1, 7, 14]:
        path = os.path.join(ARTIFACT_DIR, "lightgbm_h" + str(h) + ".joblib")
        if not os.path.exists(path):
            raise FileNotFoundError("Missing model file: " + path)
        _models[h] = joblib.load(path)


def predict_one(X, horizon: int) -> float:
    load_models()

    if horizon not in _models:
        raise ValueError("h must be 1, 7, or 14")

    # model predicts log1p(customers)
    y_log = _models[horizon].predict(X)
    y = np.expm1(y_log)

    if y[0] < 0:
        return 0.0

    return float(y[0])