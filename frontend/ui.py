import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Quick-Commerce SLA Risk Operations",
    page_icon="⚡",
    layout="wide"
)

def resolve_api_url() -> str:
    try:
        if "API_URL" in st.secrets:
            return str(st.secrets["API_URL"]).rstrip("/")
    except Exception:
        pass
    return os.getenv("API_URL", "http://localhost:8000").rstrip("/")

DEFAULT_API_URL = resolve_api_url()

st.title("⚡ Quick-Commerce SLA Risk Management System")
st.markdown("Real-time Delivery Breach Risk Prediction & Dispatch Optimization Console")

st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API Base URL", value=DEFAULT_API_URL).rstrip("/")
st.sidebar.caption("EC2 Docker default: http://api:8000 | Public/Cloud: http://35.154.79.54:8000")

tab1, tab2, tab3 = st.tabs(["🎯 Single Order Risk Predictor", "📦 Batch Dispatch Evaluator", "📊 System Health & Metrics"])

with tab1:
    st.subheader("Single Delivery Order Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Rider Age", min_value=18, max_value=60, value=28)
        ratings = st.number_input("Rider Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        distance = st.number_input("Delivery Distance (km)", min_value=0.1, max_value=50.0, value=7.5, step=0.1)
        weather = st.selectbox("Weather Condition", ["Sunny", "Stormy", "Sandstorms", "Windy", "Fog", "Cloudy"])
    with col2:
        traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
        vehicle_cond = st.selectbox("Vehicle Condition Code", [0, 1, 2])
        order_type = st.selectbox("Order Type", ["Snack", "Meal", "Drinks", "Buffet"])
        vehicle_type = st.selectbox("Vehicle Type", ["scooter", "motorcycle", "electric_scooter", "bicycle"])
    with col3:
        deliveries = st.number_input("Multiple Deliveries in Trip", min_value=0, max_value=5, value=1)
        festival = st.selectbox("Festival Period?", ["No", "Yes"])
        city = st.selectbox("City Zone Type", ["Metropolitian", "Urban", "Semi-Urban"])

    if st.button("Predict SLA Breach Risk", type="primary"):
        payload = {
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": ratings,
            "distance_km": distance,
            "Weatherconditions": weather,
            "Road_traffic_density": traffic,
            "Vehicle_condition": vehicle_cond,
            "Type_of_order": order_type,
            "Type_of_vehicle": vehicle_type,
            "multiple_deliveries": float(deliveries),
            "Festival": festival,
            "City": city
        }
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=15)
            if resp.status_code == 200:
                result = resp.json()
                st.markdown("---")
                if result.get("will_breach_sla") == 1:
                    st.error(f"⚠️ **HIGH SLA BREACH RISK DETECTED** (Probability: {result.get('sla_breach_probability', 0):.2%})")
                else:
                    st.success(f"✅ **ON-TIME EXPECTED** (Breach risk: {result.get('sla_breach_probability', 0):.2%})")
                st.info(f"**Recommendation:** {result.get('recommendation', 'N/A')}")
            else:
                st.error(f"API Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Failed to connect to API backend: {e}")

with tab2:
    st.subheader("Batch Order Dispatch Evaluation")
    st.markdown("Upload a CSV file containing multiple delivery orders for batch risk scoring.")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:", df.head())
            if st.button("Run Batch Prediction"):
                orders = df.to_dict(orient="records")
                resp = requests.post(f"{api_url}/predict/batch", json={"orders": orders}, timeout=60)
                if resp.status_code == 200:
                    results = resp.json()
                    preds = results.get("predictions", results)
                    st.success("Batch prediction completed successfully!")
                    st.dataframe(pd.DataFrame(preds) if isinstance(preds, list) else results)
                else:
                    st.error(f"Batch Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"CSV / Batch Processing Error: {e}")

with tab3:
    st.subheader("System Health & ML Model Status")
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=10)
        if health_resp.status_code == 200:
            st.success("API reachable")
            st.json(health_resp.json())
        else:
            st.warning(f"Health HTTP {health_resp.status_code}: {health_resp.text}")
    except Exception as e:
        st.error(f"API Connection Error: {e}")
