"""Feature engineering shared by flight training and API inference."""

from __future__ import annotations

import pandas as pd


NUMERIC_FEATURES = [
    "time", "distance", "year", "month", "day", "dayofweek", "quarter", "is_weekend",
]
CATEGORICAL_FEATURES = ["from", "to", "flightType", "agency", "route"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_flight_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model features from a flights frame containing a ``date`` column.

    Travel and user IDs are intentionally excluded: they identify records rather than
    travel attributes available at the time a quote is requested.
    """
    required = {"from", "to", "flightType", "agency", "time", "distance", "date"}
    absent = required - set(frame.columns)
    if absent:
        raise ValueError(f"Cannot engineer flight features; missing {sorted(absent)}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date", "time", "distance"])
    result["year"] = result["date"].dt.year.astype(int)
    result["month"] = result["date"].dt.month.astype(int)
    result["day"] = result["date"].dt.day.astype(int)
    result["dayofweek"] = result["date"].dt.dayofweek.astype(int)
    result["quarter"] = result["date"].dt.quarter.astype(int)
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)
    result["route"] = result["from"].astype(str) + " → " + result["to"].astype(str)
    return result[ALL_FEATURES]


def feature_frame_from_payload(payload: dict) -> pd.DataFrame:
    """Turn one API payload into the same raw shape expected by feature engineering."""
    date = payload.get("date")
    if date is None and all(key in payload for key in ("year", "month", "day")):
        date = f"{int(payload['year']):04d}-{int(payload['month']):02d}-{int(payload['day']):02d}"
    return pd.DataFrame([{
        "from": payload.get("from") or payload.get("from_city"),
        "to": payload.get("to") or payload.get("to_city"),
        "flightType": payload.get("flightType") or payload.get("flight_type"),
        "agency": payload.get("agency"),
        "time": payload.get("time"),
        "distance": payload.get("distance"),
        "date": date,
    }])
