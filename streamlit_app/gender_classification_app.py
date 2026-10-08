"""Interactive Streamlit front end for the Voyage Analytics REST API."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
API_DEFAULT = os.getenv("FLASK_API_URL", "http://127.0.0.1:5000")

st.set_page_config(page_title="Voyage Analytics", page_icon="✈️", layout="wide")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flights = pd.read_csv(DATA_DIR / "flights.csv", parse_dates=["date"])
    hotels = pd.read_csv(DATA_DIR / "hotels.csv", parse_dates=["date"])
    users = pd.read_csv(DATA_DIR / "users.csv")
    return flights, hotels, users


def post_json(api_url: str, endpoint: str, payload: dict) -> dict:
    response = requests.post(f"{api_url}{endpoint}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


flights, hotels, users = load_data()
with st.sidebar:
    st.title("Voyage Analytics")
    api_url = st.text_input("REST API URL", API_DEFAULT).rstrip("/")
    if st.button("Check API health"):
        try:
            st.success(requests.get(f"{api_url}/health", timeout=5).json())
        except requests.RequestException as error:
            st.error(f"API unavailable: {error}")
    st.caption("Train models before calling prediction endpoints.")

st.title("✈️ Travel Analytics & MLOps")
st.caption("Explore the supplied travel data and consume the deployed model API.")

overview_tab, price_tab, recommendation_tab, label_tab = st.tabs([
    "Overview", "Flight price", "Hotel recommendations", "Historic label demo",
])

with overview_tab:
    st.subheader("Dataset overview")
    cols = st.columns(3)
    cols[0].metric("Flights", f"{len(flights):,}")
    cols[1].metric("Hotel bookings", f"{len(hotels):,}")
    cols[2].metric("Users", f"{len(users):,}")
    left, right = st.columns(2)
    with left:
        st.write("**Median flight price by ticket type**")
        st.bar_chart(flights.groupby("flightType")["price"].median().sort_values(ascending=False))
    with right:
        st.write("**Most booked hotel destinations**")
        st.bar_chart(hotels["place"].value_counts().head(10))
    metrics = {}
    for path in (ROOT / "reports").glob("*_metrics.json"):
        metrics[path.stem] = json.loads(path.read_text())
    if metrics:
        st.subheader("Latest model reports")
        st.json(metrics)

with price_tab:
    st.subheader("Flight price prediction")
    cities = sorted(set(flights["from"]) | set(flights["to"]))
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("Origin", cities, index=0)
        destination = st.selectbox("Destination", cities, index=min(1, len(cities) - 1))
        flight_type = st.selectbox("Ticket type", sorted(flights["flightType"].unique()))
        agency = st.selectbox("Agency", sorted(flights["agency"].unique()))
    with col2:
        time = st.number_input("Duration (hours)", 0.1, 30.0, float(flights["time"].median()), 0.1)
        distance = st.number_input("Distance (km)", 1.0, 20000.0, float(flights["distance"].median()), 1.0)
        date = st.date_input("Travel date", value=flights["date"].max().date())
    if st.button("Estimate flight price", type="primary"):
        if origin == destination:
            st.warning("Choose different origin and destination cities.")
        else:
            try:
                result = post_json(api_url, "/predict_flight_price", {
                    "from": origin, "to": destination, "flightType": flight_type, "agency": agency,
                    "time": time, "distance": distance, "date": date.isoformat(),
                })
                st.success(f"Estimated price: ${result['predicted_price']:,.2f}")
                st.json(result)
            except requests.RequestException as error:
                st.error(f"Prediction failed: {error.response.text if error.response else error}")

with recommendation_tab:
    st.subheader("Personalised hotel recommendations")
    user_code = st.selectbox("User code", sorted(users["code"].astype(int).unique()))
    top_n = st.slider("Recommendations", 1, 10, 5)
    if st.button("Recommend hotels", type="primary"):
        try:
            result = post_json(api_url, "/recommend", {"user_id": int(user_code), "top_n": top_n})
            st.dataframe(pd.DataFrame(result["recommendations"]), use_container_width=True, hide_index=True)
        except requests.RequestException as error:
            st.error(f"Recommendation failed: {error.response.text if error.response else error}")

with label_tab:
    st.subheader("Historic dataset-label demonstration")
    st.warning("This is not a gender-identity classifier and must not be used for decisions about people.")
    name = st.text_input("First name", value=str(users.iloc[0]["name"]).split()[0])
    age = st.number_input("Age", 1, 120, int(users.iloc[0]["age"]))
    if st.button("Run educational demo"):
        try:
            result = post_json(api_url, "/predict_gender", {"name": name, "age": age})
            st.info(f"Historic data label: {result['predicted_historic_label']}")
            st.caption(result["notice"])
        except requests.RequestException as error:
            st.error(f"Demo failed: {error.response.text if error.response else error}")
