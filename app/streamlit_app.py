import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =========================
# Config
# =========================
BASE_API = "http://localhost:8000"
PREDICT_URL = f"{BASE_API}/predict"
HEALTH_URL = f"{BASE_API}/health"
META_URL = f"{BASE_API}/metadata"

# =========================
# Page Setup
# =========================
st.set_page_config(
    page_title="Concrete Strength ML System",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Concrete Strength Prediction System")
st.caption("ML-powered prediction with engineered features and model metadata")

# =========================
# Backend Health Check
# =========================
@st.cache_data
def check_backend():
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.json()
    except:
        return None

health = check_backend()

if health is None:
    st.error("Backend server is not running.")
    st.stop()

if health["status"] != "healthy":
    st.error("Model is not loaded in backend.")
    st.stop()

# =========================
# Load Metadata
# =========================
@st.cache_data
def get_metadata():
    try:
        r = requests.get(META_URL)
        return r.json()
    except:
        return None

meta = get_metadata()

# =========================
# Sidebar Inputs
# =========================
with st.sidebar:
    st.header("Concrete Mix")

    cement = st.number_input("Cement", min_value=1.0, value=540.0)
    slag = st.number_input("Blast Furnace Slag", min_value=0.0, value=0.0)
    flyash = st.number_input("Fly Ash", min_value=0.0, value=0.0)
    water = st.number_input("Water", min_value=1.0, value=162.0)
    sp = st.number_input("Superplasticizer", min_value=0.0, value=2.5)
    coarse = st.number_input("Coarse Aggregate", min_value=1.0, value=1040.0)
    fine = st.number_input("Fine Aggregate", min_value=1.0, value=676.0)
    age = st.number_input("Age (days)", min_value=1, value=28)

    predict = st.button("Predict")

# =========================
# Layout
# =========================
col1, col2 = st.columns([1,2])

# =========================
# Engineered Feature Preview
# =========================
def preview_features():
    binder = cement + slag + flyash
    wbr = water / binder
    scm = (slag + flyash) / cement
    return {
        "Water/Binder Ratio": round(wbr, 3),
        "SCM Ratio": round(scm, 3),
        "Log Age": round(pd.Series([age]).apply(lambda x: pd.np.log(x))[0], 3),
        "Cement x Age": cement * age
    }

# =========================
# Prediction
# =========================
if predict:

    payload = {
        "Cement": cement,
        "Blast_Furnace_Slag": slag,
        "Fly_Ash": flyash,
        "Water": water,
        "Superplasticizer": sp,
        "Coarse_Aggregate": coarse,
        "Fine_Aggregate": fine,
        "Age": age
    }

    try:
        response = requests.post(PREDICT_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        strength = result["predicted_strength_mpa"]
        category = result["strength_category"]

        # =========================
        # Result Panel
        # =========================
        with col1:
            st.metric("Predicted Strength", f"{strength:.2f} MPa")
            st.success(category)

            st.subheader("Engineered Features")
            st.json(preview_features())

        # =========================
        # Gauge Chart
        # =========================
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=strength,
                number={"suffix": " MPa"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0,20], "color": "#ffd6d6"},
                        {"range": [20,40], "color": "#fff3cd"},
                        {"range": [40,60], "color": "#d4edda"},
                        {"range": [60,100], "color": "#cce5ff"},
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend.")
    except Exception as e:
        st.error(str(e))

else:

    with col1:
        st.info("Enter mix values and click predict.")

    with col2:
        if meta:
            st.subheader("Model Info")
            st.write("Model:", meta["model_type"])
            st.write("Preprocessor:", meta["preprocessor_type"])
            st.write("Version:", meta["model_version"])