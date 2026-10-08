# Voyage Analytics — Travel MLOps Capstone

An end-to-end travel analytics project built on the supplied `users`, `flights`, and `hotels` datasets. It provides a flight-price regression API, an educational historic-label classifier, personalised hotel recommendations, a Streamlit dashboard, MLflow tracking, Airflow orchestration, Docker/Kubernetes deployment files, and Jenkins CI/CD configuration.

> **Responsible-use note:** the classification component reproduces a binary historical label found in the dataset from a name and age. It is a capstone demonstration only, not a valid or appropriate way to determine a person's gender identity or make decisions about them.

## Models and evaluation

| Component | Method | Evaluation |
|---|---|---|
| Flight pricing | One-hot features + Histogram Gradient Boosting | RMSE, MAE, R² on a chronological 80/20 holdout |
| Historic-label classification | Character n-gram TF–IDF + Linear SVC | Accuracy and weighted F1 on a stratified holdout |
| Hotel recommendation | Implicit booking affinities + Truncated SVD | Last-booking HitRate@5 |

Each training command writes a versioned model artifact into `models/`, a JSON metric report into `reports/`, and an MLflow run. Run `make train` before serving the API after replacing any dataset.

## Repository layout

```text
data/raw/                    Source CSVs (flights, hotels, users)
notebooks/                   Submission-ready EDA and modelling notebook
src/data/                    Schema validation and loading
src/features/                Reusable feature engineering
src/models/                  Training and evaluation entry points
src/inference/               Production inference helpers
src/api/                     Flask REST service
streamlit_app/               Interactive dashboard
airflow/dags/                Scheduled retraining DAG
docker/                      API, dashboard, and MLflow images
kubernetes/                  Deployments, services, PVC, and HPA
jenkins/                     Declarative CI/CD pipeline
.github/workflows/           GitHub Actions quality gate
tests/                       API, feature, and recommendation tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make train
make test
make api                           # http://localhost:5000
# in another terminal
make dashboard                     # http://localhost:8501
```

For a standalone Airflow instance that executes the DAG, install its dependency in
that Airflow environment as well:

```bash
pip install -r requirements-airflow.txt
```

Useful commands:

```bash
make mlflow                         # http://localhost:5001
docker compose up --build
kubectl apply -f kubernetes/
```

## REST API examples

```bash
curl http://localhost:5000/health

curl -X POST http://localhost:5000/predict_flight_price \
  -H 'Content-Type: application/json' \
  -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"economic","agency":"CloudFy","time":1.76,"distance":676.53,"date":"2019-09-26"}'

curl -X POST http://localhost:5000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":0,"top_n":5}'
```

## MLOps workflow

1. The Airflow DAG validates data, retrains all three models, runs tests, and records MLflow metadata on a daily schedule.
2. The Jenkins pipeline installs dependencies, runs tests, builds Docker images, and applies Kubernetes manifests when `DEPLOY_TO_K8S=true` is supplied.
3. Kubernetes uses readiness/liveness probes and an HPA for the API. Update image names in the manifests before using a remote registry.
4. MLflow defaults to local `mlruns/`; point `MLFLOW_TRACKING_URI` at an MLflow server for team tracking.

## Google Colab

Open `notebooks/Travel_Analytics_EDA_and_MLOps.ipynb` in Colab, upload the three CSV files when prompted (or clone this repository), then use **Runtime → Run all**. The notebook follows the supplied EDA template, including structured UBM analysis and 20 documented charts.

## Reference layout and data

The requested project structure was adapted from the supplied [Voyage Analytics reference repository](https://github.com/pjindal01/Voyage-Analytics). The implementation, training code, API validation, model artefacts, reporting, workflow automation, and documentation in this repository were completed for the provided `travel_capstone` data.

## Deployment checklist

- [ ] Train models and inspect the JSON reports.
- [ ] Test the three API endpoints locally.
- [ ] Set the container image registry/tag in `kubernetes/*.yaml`.
- [ ] Configure Jenkins credentials and set `DEPLOY_TO_K8S=true` for deployment.
- [ ] Configure an MLflow backend/database for shared production tracking.
