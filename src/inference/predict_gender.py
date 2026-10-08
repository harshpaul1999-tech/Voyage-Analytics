"""Inference helper for the educational historic-label classification demo."""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from src.config import MODELS_DIR


@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    path = MODELS_DIR / "gender_model.pkl"
    if not path.exists():
        raise RuntimeError("Gender label model is unavailable. Run train_gender_model first.")
    saved = joblib.load(path)
    return saved if isinstance(saved, dict) else {"model": saved}


def predict_gender(name: str, age: int) -> str:
    """Return the dataset's historic binary-label prediction for a name and age."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if not 1 <= int(age) <= 120:
        raise ValueError("age must be between 1 and 120")
    first_name = name.strip().split()[0].lower()
    frame = pd.DataFrame({"name": [first_name], "age": [int(age)]})
    return str(_load_bundle()["model"].predict(frame)[0])
