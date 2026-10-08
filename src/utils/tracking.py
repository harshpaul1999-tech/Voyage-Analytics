"""Small optional MLflow integration used by all training scripts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def mlflow_run(run_name: str) -> Iterator[Optional[object]]:
    """Yield an active MLflow module, or ``None`` if tracking is unavailable.

    Training should remain runnable in a minimal notebook environment; therefore a
    tracking outage is recorded to stdout instead of preventing model creation.
    """
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "voyage-analytics"))
    except Exception as error:  # pragma: no cover - depends on a tracking server
        print(f"[MLflow disabled] {error}")
        yield None
        return
    with mlflow.start_run(run_name=run_name):
        yield mlflow
