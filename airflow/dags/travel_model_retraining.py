"""Daily Airflow workflow for data validation, model retraining, and test gating."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = os.getenv("VOYAGE_PROJECT_ROOT", "/opt/airflow/voyage-analytics")
COMMON_ENV = {"PYTHONPATH": PROJECT_ROOT, "MLFLOW_EXPERIMENT_NAME": "voyage-analytics-airflow"}
DEFAULT_ARGS = {"owner": "mlops", "retries": 2, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="travel_model_retraining",
    description="Validate travel data and retrain all capstone models",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["travel", "mlops", "mlflow"],
) as dag:
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && python -c \"from src.data.loaders import load_flights, load_hotels, load_users; "
            "print(len(load_flights()), len(load_hotels()), len(load_users()))\""
        ),
        env=COMMON_ENV,
    )
    train_flight = BashOperator(
        task_id="train_flight_price_model",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.models.train_flight_price_model",
        env=COMMON_ENV,
    )
    train_gender = BashOperator(
        task_id="train_historic_label_model",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.models.train_gender_model",
        env=COMMON_ENV,
    )
    train_recommender = BashOperator(
        task_id="train_hotel_recommender",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.models.train_recommendation_model",
        env=COMMON_ENV,
    )
    quality_gate = BashOperator(
        task_id="quality_gate",
        bash_command=f"cd {PROJECT_ROOT} && pytest -q",
        env=COMMON_ENV,
    )

    validate_data >> [train_flight, train_gender, train_recommender] >> quality_gate
