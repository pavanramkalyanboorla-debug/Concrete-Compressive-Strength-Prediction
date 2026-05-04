# app/streamlit_app.py
# AI Concrete Mix Designer – Decision Support System
# Entrypoint for HuggingFace Spaces & local use

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.constants import COST, CONFIG, FINAL_FEATURES
from src.physics import apply_physics
from src.constraints import enforce_constraints, estimate_slump
from src.pipeline.predict_pipeline import PredictPipeline
from src.optimization import single_objective_study, multi_objective_study, best_single_mix
from src.shap_utils import ShapExplainer

# ---------------- Page config & custom CSS ----------------
st.set_page_config(page_title="Concrete Mix Optimizer", page_icon="🧱", layout="wide")

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1.5rem 0 0.5rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center; margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.9rem; color: #6c757d; margin-bottom: 0.3rem; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #212529; }
    .stButton > button {
        width: 100%; border-radius: 8px; background: #0d6efd;
        color: white; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- Header ----------------------
st.markdown("<h1 class='main-header'>Concrete Mix Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6c757d;'>Physics-constrained optimisation with Pareto trade-offs and SHAP explainability</p>", unsafe_allow_html=True)
st.divider()

# ---------------------- Initialise helpers ----------------------
pipeline = PredictPipeline()
explainer = ShapExplainer()

# ---------------------- Tabs ----------------------
tab1, tab2, tab3 = st.tabs([
    "Predict Strength",
    "Optimize Mix",
    "SHAP Explain"
])

# ========================================
# TAB 1 – Prediction
# ========================================
with tab1:
    st.header("Concrete Strength Predictor")
    st.markdown("Adjust the sliders to instantly see the predicted 28‑day compressive strength.")

    col_left, col_right = st.columns(2)
    with col_left:
        cement = st.number_input("Cement (kg/m³)", 1.0, 600.0, 350.0)
        slag = st.number_input("Blast Furnace Slag", 0.0, 300.0, 50.0)
        flyash = st.number_input("Fly Ash", 0.0, 200.0, 50.0)
        water = st.number_input("Water", 1.0, 250.0, 180.0)
    with col_right:
        sp = st.number_input("Superplasticizer", 0.0, 30.0, 5.0)
        coarse = st.number_input("Coarse Aggregate", 600.0, 1500.0, 1000.0)
        fine = st.number_input("Fine Aggregate", 400.0, 1200.0, 700.0)
        age = st.number_input("Age (days)", 1, 365, 28)

    predict_btn = st.button("Predict Strength", type="primary")

    if predict_btn:
        input_dict = {
            'Cement': cement, 'Blast_Furnace_Slag': slag,
            'Fly_Ash': flyash, 'Water': water,
            'Superplasticizer': sp,
            'Coarse_Aggregate': coarse, 'Fine_Aggregate': fine,
            'Age': age
        }
        try:
            strength = float(pipeline.predict(input_dict)[0])

            col_gauge, col_metrics = st.columns([1, 1])
            with col_gauge:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=strength,
                    number={"suffix": " MPa", "font": {"size": 32}},
                    gauge={
                        "axis": {"range": [0, 100]}, "bar": {"color": "#0d6efd"},
                        "steps": [
                            {"range": [0, 20], "color": "#ffcccc"},
                            {"range": [20, 40], "color": "#fff3cd"},
                            {"range": [40, 60], "color": "#d4edda"},
                            {"range": [60, 100], "color": "#cce5ff"}
                        ]
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col_metrics:
                if strength < 20:
                    cat = "Low Strength"
                elif strength < 40:
                    cat = "Normal Strength"
                elif strength < 60:
                    cat = "High Strength"
                else:
                    cat = "Ultra High Strength"

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Strength Category</div>
                    <div class="metric-value">{cat}</div>
                </div>""", unsafe_allow_html=True)

                binder = cement + slag + flyash
                wbr = water / binder if binder > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Water‑Binder Ratio</div>
                    <div class="metric-value">{wbr:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">SCM Ratio</div>
                    <div class="metric-value">{((slag + flyash) / cement if cement else 0):.3f}</div>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ========================================
# TAB 2 – Optimisation (includes Pareto Front)
# ========================================
with tab2:
    st.subheader("Find Optimal Mix for Target Strength")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        target = st.number_input("Target Strength (MPa)", 20.0, 80.0, 40.0)
    with col_b:
        n_trials = st.slider("Optuna Trials", 50, 300, 120)
    with col_c:
        multi_obj = st.checkbox("Multi‑objective (Pareto)", value=True)

    if st.button("Run Optimisation", type="primary"):
        with st.spinner(f"Searching {n_trials} mixes with Optuna..."):
            try:
                if multi_obj:
                    study = multi_objective_study(target_strength=target, n_trials=n_trials)
                    pareto = []
                    for t in study.best_trials:
                        p = t.params
                        mix = {
                            "Cement": p["Cement"], "Blast_Furnace_Slag": p["Slag"],
                            "Fly_Ash": p["FlyAsh"], "Water": p["Water"],
                            "Superplasticizer": p["SP"],
                            "Fine_Aggregate": 700, "Coarse_Aggregate": 1000, "Age": 28
                        }
                        mix, _ = apply_physics(mix)
                        if mix is None:
                            continue
                        strength = float(pipeline.predict(mix)[0])
                        cost = sum(mix[k] * COST[k] for k in COST)
                        _, violations = enforce_constraints(mix)
                        pareto.append({
                            "mix": {k: round(v, 2) for k, v in mix.items()
                                    if k in ['Cement','Blast_Furnace_Slag','Fly_Ash','Water',
                                             'Superplasticizer','Fine_Aggregate','Coarse_Aggregate']},
                            "predicted_strength": round(strength, 2),
                            "cost": round(cost, 2),
                            "strength_error": round(abs(strength - target), 2),
                            "violations": violations,
                            "slump_mm": round(estimate_slump(mix), 0)
                        })
                    st.session_state["last_pareto"] = pareto
                    st.success(f"Found {len(pareto)} Pareto‑optimal mixes.")

                    df = pd.DataFrame(pareto)
                    fig_pareto = px.scatter(
                        df, x="cost", y="strength_error",
                        hover_data=df.columns,
                        title="Pareto Front — Cost vs. Strength Accuracy"
                    )
                    fig_pareto.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
                    fig_pareto.update_layout(xaxis_title="Cost (Rs/m³)", yaxis_title="Strength Error (MPa)")
                    st.plotly_chart(fig_pareto, use_container_width=True)

                    st.subheader("Recommended Trade-offs")
                    df_clean = df[df["violations"].apply(lambda x: len(x) == 0)]
                    if len(df_clean) == 0:
                        st.warning("No mix fully satisfies all constraints.")
                    else:
                        best_cost = df_clean.loc[df_clean["cost"].idxmin()]
                        best_accuracy = df_clean.loc[df_clean["strength_error"].idxmin()]
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            st.markdown("#### Cheapest Compliant Mix")
                            st.json(best_cost.to_dict())
                        with col_p2:
                            st.markdown("#### Most Accurate Mix")
                            st.json(best_accuracy.to_dict())
                        st.caption("Choose the mix that matches your budget and performance needs.")

                    st.subheader("All Pareto‑Optimal Mixes")
                    for i, row in df.iterrows():
                        with st.expander(f"Mix #{i+1} — {row['predicted_strength']:.1f} MPa, Rs{row['cost']:.0f}/m³"):
                            st.json(row["mix"])
                            if row["violations"]:
                                st.warning("Violations: " + ", ".join(row["violations"]))
                            st.write(f"Slump: {row['slump_mm']} mm")

                else:
                    study = single_objective_study(target_strength=target, n_trials=n_trials)
                    mix = best_single_mix(study)
                    strength = float(pipeline.predict(mix)[0])
                    cost = sum(mix[k] * COST[k] for k in COST)
                    _, violations = enforce_constraints(mix)
                    slump = estimate_slump(mix)

                    st.subheader("Recommended Mix")
                    mix_df = pd.DataFrame([{k: round(v, 2) for k, v in mix.items()
                                            if k in ['Cement','Blast_Furnace_Slag','Fly_Ash','Water',
                                                     'Superplasticizer','Fine_Aggregate','Coarse_Aggregate']}
                                          ]).T.rename(columns={0: "kg/m³"})
                    st.dataframe(mix_df)
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Predicted Strength", f"{strength:.1f} MPa")
                    col_m2.metric("Cost", f"Rs{cost:.0f}/m³")
                    col_m3.metric("Est. Slump", f"{slump:.0f} mm")
                    if violations:
                        st.warning("Violations: " + ", ".join(violations))
                    else:
                        st.success("All engineering constraints passed")
            except Exception as e:
                st.error(f"Optimisation failed: {e}")

# ========================================
# TAB 3 – SHAP Explainability
# ========================================
with tab3:
    st.subheader("SHAP Waterfall – Understand Feature Contributions")
    with st.form("shap_form"):
        cols = st.columns(4)
        s_cement = cols[0].number_input("Cement", 100, 600, 350, key="sh_cement")
        s_slag = cols[1].number_input("Slag", 0, 300, 50, key="sh_slag")
        s_flyash = cols[2].number_input("Fly Ash", 0, 200, 50, key="sh_fa")
        s_water = cols[3].number_input("Water", 100, 250, 180, key="sh_water")
        s_sp = cols[0].number_input("SP", 0, 30, 5, key="sh_sp")
        s_coarse = cols[1].number_input("Coarse Agg.", 600, 1500, 1000, key="sh_ca")
        s_fine = cols[2].number_input("Fine Agg.", 400, 1200, 700, key="sh_fa2")
        s_age = cols[3].number_input("Age (days)", 1, 365, 28, key="sh_age")
        shap_submit = st.form_submit_button("Explain with SHAP")

    if shap_submit:
        mix_dict = {
            "Cement": s_cement, "Blast_Furnace_Slag": s_slag,
            "Fly_Ash": s_flyash, "Water": s_water,
            "Superplasticizer": s_sp,
            "Coarse_Aggregate": s_coarse, "Fine_Aggregate": s_fine,
            "Age": s_age
        }
        with st.spinner("Computing SHAP values..."):
            try:
                mix_phys, _ = apply_physics(mix_dict)
                if mix_phys is None:
                    st.error("Volume overflow – adjust inputs")
                else:
                    fig, shap_vals = explainer.waterfall_plot(mix_phys)
                    st.pyplot(fig)
                    reasoning = explainer.reasoning_text(shap_vals, None)
                    st.markdown(reasoning)
            except Exception as e:
                st.error(f"SHAP failed: {e}")