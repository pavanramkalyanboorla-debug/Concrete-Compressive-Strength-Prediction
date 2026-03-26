import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
    }
    .metric-card h1 { font-size: 2.5rem; margin: 0; }
    .metric-card p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.95rem; }
    .category-badge {
        display: inline-block;
        background: #e8f4fd; color: #1e3a5f;
        border-radius: 20px; padding: 6px 18px;
        font-weight: 600; font-size: 0.9rem; margin-top: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white; border: none; border-radius: 8px;
        padding: 12px 32px; font-size: 1rem; font-weight: 600;
        width: 100%; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/predict"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏗️ Concrete Compressive Strength Predictor")
st.markdown("Enter your concrete mix composition below to predict its **compressive strength (MPa)**.")
st.divider()

# ── Sidebar – inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Mix Composition")
    st.markdown("All quantities in **kg/m³** unless noted.")

    cement              = st.number_input("Cement",              min_value=0.0, value=540.0,  step=10.0)
    blast_furnace_slag  = st.number_input("Blast Furnace Slag",  min_value=0.0, value=0.0,    step=10.0)
    fly_ash             = st.number_input("Fly Ash",             min_value=0.0, value=0.0,    step=10.0)
    water               = st.number_input("Water",               min_value=0.0, value=162.0,  step=5.0)
    superplasticizer    = st.number_input("Superplasticizer",    min_value=0.0, value=2.5,    step=0.5)
    coarse_aggregate    = st.number_input("Coarse Aggregate",    min_value=0.0, value=1040.0, step=10.0)
    fine_aggregate      = st.number_input("Fine Aggregate",      min_value=0.0, value=676.0,  step=10.0)
    age                 = st.number_input("Age (days)",          min_value=1.0, value=28.0,   step=1.0)

    st.divider()
    predict_btn = st.button("🔍 Predict Strength")

# ── Main area ─────────────────────────────────────────────────────────────────
col_result, col_chart = st.columns([1, 2])

if predict_btn:
    payload = {
        "Cement":             cement,
        "Blast_Furnace_Slag": blast_furnace_slag,
        "Fly_Ash":            fly_ash,
        "Water":              water,
        "Superplasticizer":   superplasticizer,
        "Coarse_Aggregate":   coarse_aggregate,
        "Fine_Aggregate":     fine_aggregate,
        "Age":                int(age),
    }

    with st.spinner("Running prediction…"):
        try:
            resp = requests.post(API_URL, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            strength   = result["predicted_strength_mpa"]
            category   = result["strength_category"]
            importance = result["feature_importance"]

            # ── Result card ───────────────────────────────────────────────
            with col_result:
                st.markdown(f"""
                <div class="metric-card">
                    <p>Predicted Compressive Strength</p>
                    <h1>{strength} MPa</h1>
                    <span class="category-badge">{category}</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📊 Strength Gauge")
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=strength,
                    number={"suffix": " MPa", "font": {"size": 24}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#2d6a9f"},
                        "steps": [
                            {"range": [0,  20], "color": "#ffd6d6"},
                            {"range": [20, 40], "color": "#fff3cd"},
                            {"range": [40, 60], "color": "#d4edda"},
                            {"range": [60,100], "color": "#cce5ff"},
                        ],
                        "threshold": {
                            "line": {"color": "#1e3a5f", "width": 3},
                            "thickness": 0.75,
                            "value": strength
                        }
                    }
                ))
                gauge.update_layout(height=260, margin=dict(t=20, b=10, l=20, r=20))
                st.plotly_chart(gauge, use_container_width=True)

            # ── Feature importance chart ───────────────────────────────────
            with col_chart:
                st.markdown("#### 🔬 Feature Importance")

                if importance:
                    labels = [k.replace("_", " ").title() for k in importance.keys()]
                    values = list(importance.values())
                    sorted_pairs = sorted(zip(values, labels), reverse=True)
                    values_s, labels_s = zip(*sorted_pairs)

                    fig = px.bar(
                        x=list(values_s),
                        y=list(labels_s),
                        orientation="h",
                        color=list(values_s),
                        color_continuous_scale=["#a8d5f5", "#1e3a5f"],
                        labels={"x": "Importance Score", "y": "Feature"},
                        text=[f"{v:.4f}" for v in values_s]
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(
                        height=380,
                        coloraxis_showscale=False,
                        margin=dict(t=10, b=10, l=10, r=60),
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Feature importance is currently unavailable for this model. Showing input summary instead.")

                # Input summary table
                st.markdown("#### 📋 Input Summary")
                input_df = {
                    "Feature": [
                        "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
                        "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age"
                    ],
                    "Value (kg/m³ or days)": [
                        cement, blast_furnace_slag, fly_ash, water,
                        superplasticizer, coarse_aggregate, fine_aggregate, age
                    ]
                }
                st.dataframe(input_df, use_container_width=True, hide_index=True)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API error: {e.response.text}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

else:
    with col_result:
        st.info("👈 Enter mix composition in the sidebar and click **Predict Strength**.")

    with col_chart:
        st.markdown("#### How it works")
        st.markdown("""
        This app uses a trained **machine learning model** to predict the compressive
        strength of concrete based on 8 input features:

        | Feature | Unit |
        |---|---|
        | Cement | kg/m³ |
        | Blast Furnace Slag | kg/m³ |
        | Fly Ash | kg/m³ |
        | Water | kg/m³ |
        | Superplasticizer | kg/m³ |
        | Coarse Aggregate | kg/m³ |
        | Fine Aggregate | kg/m³ |
        | Age | days |

        **Strength categories:**
        - 🔴 Low: < 20 MPa
        - 🟡 Normal: 20–40 MPa
        - 🟢 High: 40–60 MPa
        - 🔵 Ultra-High: > 60 MPa
        """)