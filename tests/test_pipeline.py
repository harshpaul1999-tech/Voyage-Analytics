from src.api.classification_api import app
from src.features.flight_features import prepare_flight_features
from src.features.recommendation_features import hotel_key


def test_flight_feature_engineering_creates_expected_columns():
    frame = prepare_flight_features(__import__("pandas").DataFrame([{
        "from": "A", "to": "B", "flightType": "economic", "agency": "X",
        "time": 2.0, "distance": 100.0, "date": "2020-01-04",
    }]))
    assert frame.loc[0, "route"] == "A → B"
    assert frame.loc[0, "is_weekend"] == 1


def test_hotel_key_includes_name_and_place():
    import pandas as pd
    keys = hotel_key(pd.DataFrame({"name": ["Hotel A"], "place": ["City X"]}))
    assert keys.iloc[0] == "Hotel A | City X"


def test_health_endpoint_is_available():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_prediction_endpoints_return_valid_responses():
    client = app.test_client()
    flight = client.post("/predict_flight_price", json={
        "from": "Recife (PE)", "to": "Florianopolis (SC)",
        "flightType": "firstClass", "agency": "FlyingDrops",
        "time": 1.76, "distance": 676.53, "date": "2019-09-26",
    })
    recommendations = client.post("/recommend", json={"user_id": 1, "top_n": 3})
    assert flight.status_code == 200
    assert flight.json["predicted_price"] > 0
    assert recommendations.status_code == 200
    assert len(recommendations.json["recommendations"]) == 3
