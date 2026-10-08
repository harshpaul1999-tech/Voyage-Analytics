"""Train an implicit-feedback SVD hotel recommender from completed hotel stays."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.config import MODELS_DIR, REPORTS_DIR, ensure_project_dirs
from src.data.loaders import load_hotels
from src.features.recommendation_features import hotel_key, make_interaction_matrix
from src.utils.tracking import mlflow_run


def _top_k_scores(scores: np.ndarray, seen: np.ndarray, k: int) -> np.ndarray:
    ranked = scores.copy()
    ranked[seen] = -np.inf
    return np.argsort(ranked)[::-1][:k]


def train(random_state: int = 42, top_k: int = 5) -> dict:
    """Fit SVD and evaluate a last-interaction holdout using HitRate@K."""
    ensure_project_dirs()
    hotels = load_hotels().dropna(subset=["userCode", "name", "place", "date", "days"])
    hotels = hotels.sort_values(["userCode", "date"]).copy()
    hotels["hotel_key"] = hotel_key(hotels)
    eligible = hotels.groupby("userCode").filter(lambda rows: len(rows) >= 2)
    holdout_index = eligible.groupby("userCode").tail(1).index
    test_events, train_events = hotels.loc[holdout_index], hotels.drop(index=holdout_index)
    matrix = make_interaction_matrix(train_events)
    components = max(1, min(12, matrix.shape[0] - 1, matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=components, random_state=random_state)

    with mlflow_run("hotel-recommendation") as mlflow:
        latent = svd.fit_transform(matrix)
        scores = latent @ svd.components_
        seen = matrix.to_numpy() > 0
        users = matrix.index.to_numpy()
        hotels_keys = matrix.columns.to_numpy()
        user_lookup = {int(code): position for position, code in enumerate(users)}
        item_lookup = {str(key): position for position, key in enumerate(hotels_keys)}
        evaluable = test_events[test_events["userCode"].isin(user_lookup)]
        hits = 0
        for _, event in evaluable.iterrows():
            ranked = _top_k_scores(scores[user_lookup[int(event.userCode)]], seen[user_lookup[int(event.userCode)]], top_k)
            if item_lookup.get(str(event.hotel_key)) in ranked:
                hits += 1
        metrics = {
            f"hit_rate_at_{top_k}": float(hits / len(evaluable)) if len(evaluable) else 0.0,
            "users": int(matrix.shape[0]), "hotels": int(matrix.shape[1]), "components": int(components),
        }
        if mlflow:
            mlflow.log_params({"algorithm": "TruncatedSVD", "components": components, "top_k": top_k})
            mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, float)})
            mlflow.sklearn.log_model(svd, artifact_path="hotel_recommender_svd")

    metadata = (hotels.sort_values("date").groupby("hotel_key", as_index=False).tail(1)
                [["hotel_key", "name", "place", "price"]].set_index("hotel_key").to_dict("index"))
    popularity = train_events["hotel_key"].value_counts().index.tolist()
    bundle = {
        "svd": svd, "scores": scores, "seen": seen, "users": users, "hotel_keys": hotels_keys,
        "metadata": metadata, "popularity": popularity, "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, MODELS_DIR / "travel_recommendation_model.pkl")
    # Compatibility artifact retained for the reference repository layout.
    joblib.dump(bundle, MODELS_DIR / "svd_recommendation_model.pkl")
    (REPORTS_DIR / "recommendation_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
