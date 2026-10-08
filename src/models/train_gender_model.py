"""Train a transparent baseline that predicts the historic binary dataset label.

This model is educational only. Names and age should not be used to infer a person's
gender identity in a production decision process.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import MODELS_DIR, REPORTS_DIR, ensure_project_dirs
from src.data.loaders import load_users
from src.utils.tracking import mlflow_run


def train(random_state: int = 42) -> dict:
    """Fit a character-ngram classifier and save it with its test-set metrics."""
    ensure_project_dirs()
    users = load_users()
    users = users[users["gender"].str.lower().isin(["male", "female"])].copy()
    users["name"] = users["name"].str.split().str[0].str.lower()
    users["gender"] = users["gender"].str.lower()
    X, y = users[["name", "age"]], users["gender"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    pipeline = Pipeline([
        ("features", ColumnTransformer([
            ("name_ngrams", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True), "name"),
            ("age", StandardScaler(), ["age"]),
        ])),
        ("classifier", LinearSVC(C=0.8, class_weight="balanced")),
    ])
    with mlflow_run("gender-label-classification") as mlflow:
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "f1_weighted": float(f1_score(y_test, predicted, average="weighted")),
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        }
        if mlflow:
            mlflow.log_params({"model": "LinearSVC", "name_ngrams": "char_wb_2_to_5"})
            mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, float)})
            mlflow.sklearn.log_model(pipeline, artifact_path="gender_label_model")
    bundle = {
        "model": pipeline, "metrics": metrics, "trained_at": datetime.now(timezone.utc).isoformat(),
        "classification_report": classification_report(y_test, predicted, output_dict=True),
        "notice": "Educational historic-label demo; not a gender-identity determination system.",
    }
    joblib.dump(bundle, MODELS_DIR / "gender_model.pkl")
    (REPORTS_DIR / "gender_classification_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
