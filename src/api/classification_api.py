"""Flask service exposing the three Voyage Analytics model endpoints."""

from __future__ import annotations

from flask import Flask, jsonify, request

from src.inference.predict_flight_price import predict_flight_price
from src.inference.predict_gender import predict_gender
from src.inference.predict_recommendations import get_recommendations


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def _json_body() -> dict:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    return body


@app.get("/health")
def health():
    """Kubernetes and Docker health-check endpoint."""
    return jsonify({"status": "ok", "service": "voyage-analytics-api", "version": "1.0.0"})


@app.post("/predict_flight_price")
def predict_flight_price_route():
    """Estimate price from origin, destination, flight type, time, distance, and date."""
    try:
        body = _json_body()
        required = {"time", "distance", "agency"}
        absent = required - set(body)
        if not (body.get("from") or body.get("from_city")):
            absent.add("from")
        if not (body.get("to") or body.get("to_city")):
            absent.add("to")
        if not (body.get("flightType") or body.get("flight_type")):
            absent.add("flightType")
        if not (body.get("date") or {"year", "month", "day"}.issubset(body)):
            absent.add("date")
        if absent:
            return jsonify({"error": f"Missing fields: {sorted(absent)}"}), 400
        price = predict_flight_price(**body)
        return jsonify({"predicted_price": price, "currency": "USD", "model": "flight_price_model"})
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503


@app.post("/predict_gender")
def predict_gender_route():
    """Predict the historical binary label solely for the capstone demonstration."""
    try:
        body = _json_body()
        if {"name", "age"} - set(body):
            return jsonify({"error": "name and age are required"}), 400
        label = predict_gender(str(body["name"]), int(body["age"]))
        return jsonify({
            "predicted_historic_label": label,
            "notice": "Educational dataset-label demo; it does not determine gender identity.",
        })
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503


@app.post("/recommend")
def recommend_route():
    """Recommend hotels from a user's observed booking history."""
    try:
        body = _json_body()
        if "user_id" not in body:
            return jsonify({"error": "user_id is required"}), 400
        candidate_keys = body.get("candidate_hotel_keys")
        if candidate_keys is not None and not isinstance(candidate_keys, list):
            return jsonify({"error": "candidate_hotel_keys must be a list"}), 400
        recommendations = get_recommendations(
            user_id=int(body["user_id"]), top_n=int(body.get("top_n", 5)),
            candidate_hotel_keys=candidate_keys,
        )
        return jsonify({"user_id": int(body["user_id"]), "recommendations": recommendations})
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
