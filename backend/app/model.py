import os
from threading import Lock
import joblib
import numpy as np
try:
    import shap
except Exception:
    shap = None


ARTIFACT_DIR = os.getenv(
    "MODEL_ARTIFACT_DIR",
    os.path.join("..", "..", "evaluation", "experiments", "model_selection", "artifacts"),
)
MODEL_ARTIFACT_PREFIX = os.getenv("MODEL_ARTIFACT_PREFIX", "lightgbm")

_models = None
_models_lock = Lock()
_shap_explainers = {}
_shap_lock = Lock()


def load_models():
    global _models
    if _models is not None:
        return

    with _models_lock:
        if _models is not None:
            return

        loaded = {}
        for h in [1, 7, 14]:
            path = os.path.join(ARTIFACT_DIR, MODEL_ARTIFACT_PREFIX + "_h" + str(h) + ".joblib")
            if not os.path.exists(path):
                raise FileNotFoundError("Missing model file: " + path)
            loaded[h] = joblib.load(path)

        _models = loaded


def predict_one(X, horizon: int) -> float:
    load_models()

    if horizon not in _models:
        raise ValueError("h must be 1, 7, or 14")

    model_obj = _models[horizon]
    y_log = model_obj.predict(X)

    y = np.expm1(y_log)

    if y[0] < 0:
        return 0.0

    return float(y[0])


def _to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _pretty_feature_name(name: str) -> str:
    core = name
    if core.startswith("num__"):
        core = core[5:]
    elif core.startswith("cat__"):
        core = core[5:]

    for cat_field in ["StoreType", "Assortment", "StateHoliday", "PromoInterval", "Store"]:
        prefix = cat_field + "_"
        if core.startswith(prefix):
            return f"{cat_field} = {core[len(prefix):]}"

    return core.replace("_", " ")


def _get_sklearn_pipeline_parts(model_obj):
    """Extract (pre, estimator) from a sklearn Pipeline, or return (None, None).

    Handles pipelines whose last step is a tree-based model (LightGBM, XGBoost, etc.)
    or is itself a meta-estimator wrapping one.
    """
    try:
        from sklearn.pipeline import Pipeline
        if not isinstance(model_obj, Pipeline):
            return None, None
        pre = model_obj[:-1]   # all steps before the final estimator
        final = model_obj[-1]
        # Unwrap common wrappers (e.g. TransformedTargetRegressor)
        inner = getattr(final, "regressor_", getattr(final, "estimator_", final))
        return pre, inner
    except Exception:
        return None, None


def _get_tree_explainer(horizon: int, estimator):
    with _shap_lock:
        existing = _shap_explainers.get(horizon)
        if existing is not None:
            return existing
        explainer = shap.TreeExplainer(estimator)
        _shap_explainers[horizon] = explainer
        return explainer


def _safe_feature_names(pre, n_features: int):
    try:
        names = list(pre.get_feature_names_out())
        if len(names) == n_features:
            return names
    except Exception:
        pass
    return [f"f{i}" for i in range(n_features)]


def _build_explanation(feature_names, contrib_values, pred_log: float, top_n: int, base_log: float | None = None):
    vals = np.asarray(contrib_values).reshape(-1)
    if len(feature_names) != len(vals):
        feature_names = [f"f{i}" for i in range(len(vals))]

    factors = []
    for i, raw_name in enumerate(feature_names):
        sv = float(vals[i])
        if sv == 0.0:
            continue
        factors.append({
            "feature": raw_name,
            "label": _pretty_feature_name(raw_name),
            "shap_value": sv,
            "abs_shap_value": abs(sv),
        })

    positives = sorted(
        [f for f in factors if f["shap_value"] > 0],
        key=lambda x: x["shap_value"],
        reverse=True,
    )[:top_n]
    negatives = sorted(
        [f for f in factors if f["shap_value"] < 0],
        key=lambda x: x["shap_value"],
    )[:top_n]

    if base_log is None:
        base_log = float(pred_log - np.sum(vals))

    return {
        "base_log1p": float(base_log),
        "predicted_log1p": float(pred_log),
        "top_positive_factors": positives,
        "top_negative_factors": negatives,
    }


def explain_one(X, horizon: int, top_n: int = 10) -> dict | None:
    if shap is None:
        return None

    load_models()
    model_obj = _models.get(horizon)
    if model_obj is None:
        return None

    # LightGBM pipeline path
    pre, estimator = _get_sklearn_pipeline_parts(model_obj)
    if pre is None or estimator is None:
        return None

    try:
        X_t = pre.transform(X)
        X_dense = _to_dense(X_t)
        pred_log = float(model_obj.predict(X)[0])
        n_features = np.asarray(X_dense).reshape(1, -1).shape[1]
        feature_names = _safe_feature_names(pre, n_features)
    except Exception:
        return None

    try:
        explainer = _get_tree_explainer(horizon, estimator)
        out = explainer(np.asarray(X_dense).reshape(1, -1))
        values = np.asarray(out.values)
        shap_vals = values if values.ndim == 1 else values[0]
        base_vals = np.asarray(out.base_values).reshape(-1)
        base_log = float(base_vals[0]) if base_vals.size else None
        return _build_explanation(
            feature_names=feature_names,
            contrib_values=shap_vals,
            pred_log=pred_log,
            top_n=top_n,
            base_log=base_log,
        )
    except Exception:
        return None
