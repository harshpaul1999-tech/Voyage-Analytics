.PHONY: install train test api dashboard mlflow docker-up k8s-apply

install:
	python -m pip install --upgrade pip && python -m pip install -r requirements.txt

train:
	python -m src.models.train_flight_price_model
	python -m src.models.train_gender_model
	python -m src.models.train_recommendation_model

test:
	pytest -q

api:
	gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 src.api.app:app

dashboard:
	streamlit run streamlit_app/gender_classification_app.py

mlflow:
	mlflow ui --backend-store-uri ./mlruns --port 5001

docker-up:
	docker compose up --build

k8s-apply:
	kubectl apply -f kubernetes/
