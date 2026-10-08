"""Personalised and cold-start hotel recommendation helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

import joblib
import numpy as np

from src.config import MODELS_DIR


@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    path = MODELS_DIR / "travel_recommendation_model.pkl"
    if not path.exists():
        raise RuntimeError("Recommendation model is unavailable. Run train_recommendation_model first.")
    return joblib.load(path)


def get_recommendations(user_id: int, top_n: int = 5, candidate_hotel_keys: Optional[List[str]] = None) -> List[dict]:
    """Return unseen hotels in descending affinity score, with a popular fallback."""
    if not 1 <= int(top_n) <= 20:
        raise ValueError("top_n must be between 1 and 20")
    bundle = _load_bundle()
    user_ids = list(bundle["users"])
    hotel_keys = list(bundle["hotel_keys"])
    metadata = bundle["metadata"]
    selected: List[Tuple[str, Optional[float]]] = []
    if int(user_id) in user_ids:
        user_position = user_ids.index(int(user_id))
        scores = bundle["scores"][user_position].copy()
        scores[bundle["seen"][user_position]] = -np.inf
        candidates = set(candidate_hotel_keys) if candidate_hotel_keys else set(hotel_keys)
        ranked = np.argsort(scores)[::-1]
        for index in ranked:
            key = str(hotel_keys[index])
            if key in candidates and np.isfinite(scores[index]):
                selected.append((key, float(scores[index])))
            if len(selected) == top_n:
                break
    if len(selected) < top_n:
        already_selected = {key for key, _ in selected}
        for key in bundle["popularity"]:
            if key not in already_selected and (not candidate_hotel_keys or key in candidate_hotel_keys):
                selected.append((key, None))
            if len(selected) == top_n:
                break
    results = []
    for key, score in selected:
        item = dict(metadata.get(key, {}))
        item.update({"hotel_key": key, "recommendation_score": None if score is None else round(score, 4)})
        results.append(item)
    return results
