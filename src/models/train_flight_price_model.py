"""Train, evaluate, track, and persist the flight-price regression pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import MODELS_DIR, REPORTS_DIR, ensure_project_dirs
from src.data.loaders import load_flights
from src.features.flight_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_flight_features
from src.utils.tracking import mlflow_run


def train(sample_size: Optional[int] = None, random_state: int = 42) -> dict:
    """Fit a time-aware regression model and return reproducible evaluation metrics."""
    ensure_project_dirs()
    flights = load_flights().dropna(subset=["date", "price"])
    if sample_size and sample_size < len(flights):
        flights = flights.sample(sample_size, random_state=random_state)
    flights = flights.sort_values("date").reset_index(drop=True)
    features = prepare_flight_features(flights)
    target = flights.loc[features.index, "price"].astype(float)

    split_at = int(len(features) * 0.80)
    X_train, X_test = features.iloc[:split_at], features.iloc[split_at:]
    y_train, y_test = target.iloc[:split_at], target.iloc[split_at:]
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
         ("numeric", "passthrough", NUMERIC_FEATURES)],
        sparse_threshold=0,
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(
            learning_rate=0.08, max_leaf_nodes=63, max_iter=180,
            l2_regularization=1.0, random_state=random_state,
        )),
    ])

    with mlflow_run("flight-price-regression") as mlflow:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
        if mlflow:
            mlflow.log_params({"model": "HistGradientBoostingRegressor", "max_iter": 180})
            mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, float)})
            mlflow.sklearn.log_model(model, artifact_path="flight_price_model")

    bundle = {
        "model": model, "metrics": metrics, "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(features.columns),
    }
    joblib.dump(bundle, MODELS_DIR / "flight_price_model.pkl")
    (REPORTS_DIR / "flight_regression_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()
    train(sample_size=args.sample_size)
