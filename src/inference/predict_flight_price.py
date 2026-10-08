"""Flight-price inference backed by the persisted sklearn pipeline."""

from __future__ import annotations

from functools import lru_cache

import joblib

from src.config import MODELS_DIR
from src.features.flight_features import feature_frame_from_payload, prepare_flight_features


@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    path = MODELS_DIR / "flight_price_model.pkl"
    if not path.exists():
        raise RuntimeError("Flight model is unavailable. Run train_flight_price_model first.")
    bundle = joblib.load(path)
    return bundle if isinstance(bundle, dict) else {"model": bundle}


def predict_flight_price(**payload: object) -> float:
    """Estimate a flight price from a JSON-ready quote payload."""
    origin = payload.get("from") or payload.get("from_city")
    destination = payload.get("to") or payload.get("to_city")
    if origin == destination:
        raise ValueError("Origin and destination must be different.")
    raw = feature_frame_from_payload(payload)
    if raw.isna().any().any():
        missing = raw.columns[raw.isna().any()].tolist()
        raise ValueError(f"Missing or invalid fields: {missing}")
    features = prepare_flight_features(raw)
    if features.empty:
        raise ValueError("date must be a valid ISO date (YYYY-MM-DD).")
    prediction = _load_bundle()["model"].predict(features)
    return round(float(prediction[0]), 2)
