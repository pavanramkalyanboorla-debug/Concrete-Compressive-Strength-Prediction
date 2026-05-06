# app/streamlit_app.py
# Concrete Mix Optimizer — Stable UI (no layout-shift / shaking)

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle

from src.constants import COST, CONFIG
from src.physics import apply_physics
from src.constraints import enforce_constraints, estimate_slump
from src.pipeline.predict_pipeline import PredictPipeline
from src.optimization import single_objective_study, multi_objective_study, best_single_mix, _physics as opt_physics
from src.shap_utils import ShapExplainer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Concrete Mix Optimizer",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS – unchanged
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, .main { overflow-x: hidden !important; }
.block-container { max-width: 1150px; margin: auto; padding-top: 0.5rem; }
.main-header { text-align:center; font-size:2.2rem; font-weight:700; margin-bottom:0.2rem; }
.subtext     { text-align:center; color:#8b949e; margin-bottom:1rem; }
.metric-card {
    background:#161b22; border-radius:12px; padding:1rem;
    text-align:center; border:1px solid #30363d; margin-bottom:0.75rem;
    min-height: 90px;
}
.metric-label { font-size:0.85rem; color:#8b949e; }
.metric-value { font-size:1.6rem; font-weight:700; }
.violation-box {
    background:#2d1b1b; border:1px solid #f85149; border-radius:8px;
    padding:0.75rem 1rem; color:#f85149; font-size:0.9rem;
    min-height:48px;
}
.ok-box {
    background:#1b2d1b; border:1px solid #3fb950; border-radius:8px;
    padding:0.75rem 1rem; color:#3fb950; font-size:0.9rem;
    min-height:48px;
}
.stButton > button {
    width:100%; border-radius:8px;
    background:#238636; color:white; font-weight:600;
}
.js-plotly-plot .plotly { min-height:260px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🧱 Concrete Mix Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Physics-constrained optimisation • Pareto trade-offs • SHAP explainability</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return PredictPipeline()

@st.cache_resource
def load_explainer(_pipeline):
    return ShapExplainer(model=_pipeline.model, preprocessor=_pipeline.preprocessor)

@st.cache_resource
def load_uncertainty_params():
    try:
        with open("artifacts/uncertainty_params.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

pipeline = load_pipeline()
explainer = load_explainer(pipeline)
uncertainty_params = load_uncertainty_params()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key in ("pred_result", "opt_result", "shap_result"):
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_mix(cement, slag, flyash, water, sp, coarse, fine, age):
    return {
        "Cement": cement, "Blast_Furnace_Slag": slag, "Fly_Ash": flyash,
        "Water": water, "Superplasticizer": sp,
        "Coarse_Aggregate": coarse, "Fine_Aggregate": fine, "Age": age,
    }

def grade_label(s):
    if s < 20:  return "M15 or below"
    if s < 25:  return "M20"
    if s < 30:  return "M25"
    if s < 35:  return "M30"
    if s < 40:  return "M35"
    if s < 50:  return "M40"
    return "M50+"

def strength_color(s):
    if s < 25:  return "#f85149"
    if s < 40:  return "#d29922"
    return "#3fb950"

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Predict", "⚙️ Optimize", "🧠 Explain"])

with tab1:
    st.subheader("Predict Concrete Strength")

    col1, col2 = st.columns(2)
    with col1:
        cement = st.number_input("Cement (kg/m³)",           100.0, 600.0,  350.0, key="p_cement")
        slag   = st.number_input("Slag (kg/m³)",             0.0,   300.0,   50.0, key="p_slag")
        flyash = st.number_input("Fly Ash (kg/m³)",          0.0,   200.0,   50.0, key="p_flyash")
        water  = st.number_input("Water (kg/m³)",            100.0, 250.0,  180.0, key="p_water")
    with col2:
        sp     = st.number_input("Superplasticizer (kg/m³)", 0.0,   30.0,    5.0,  key="p_sp")
        coarse = st.number_input("Coarse Aggregate (kg/m³)", 600.0, 1500.0, 1000.0, key="p_coarse")
        fine   = st.number_input("Fine Aggregate (kg/m³)",   400.0, 1200.0,  700.0, key="p_fine")
        age    = st.number_input("Age (days)",                1,     365,     28,    key="p_age")

    if st.button("Predict Strength", key="btn_predict"):
        raw_mix  = build_mix(cement, slag, flyash, water, sp, coarse, fine, age)
        phys_mix, _ = apply_physics(raw_mix)
        if phys_mix is None:
            st.session_state.pred_result = {"error": "Paste volume failed — reduce cement or water."}
        else:
            valid_mix, violations = enforce_constraints(phys_mix)
            slump    = estimate_slump(phys_mix)
            strength = float(pipeline.predict(raw_mix)[0])
            lower = upper = None
            if uncertainty_params is not None:
                sigma = uncertainty_params["cv_std"]
                lower = strength - 1.645 * sigma
                upper = strength + 1.645 * sigma
            binder   = cement + slag + flyash
            wbr      = water / binder if binder else 0
            cost     = sum(raw_mix[k] * COST[k] for k in COST if k in raw_mix)
            st.session_state.pred_result = {
                "strength": strength, "lower": lower, "upper": upper,
                "wbr": wbr, "slump": slump, "cost": cost, "violations": violations,
            }

    # ── Result display (unchanged) ───────────────────────────────────────
    result = st.session_state.pred_result
    colA, colB = st.columns(2)

    with colA:
        gauge_slot = st.empty()
        if result and "error" not in result:
            s     = result["strength"]
            color = strength_color(s)
            fig   = go.Figure(go.Indicator(
                mode="gauge+number",
                value=s,
                number={"suffix": " MPa", "font": {"color": color, "size": 28}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0, 25],   "color": "#2d1b1b"},
                        {"range": [25, 40],  "color": "#2d2510"},
                        {"range": [40, 100], "color": "#1b2d1b"},
                    ],
                },
            ))
            fig.update_layout(margin=dict(t=30, b=10, l=20, r=20), height=260, autosize=False)
            gauge_slot.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})
        elif result and "error" in result:
            gauge_slot.error(f"❌ {result['error']}")
        else:
            gauge_slot.markdown(
                '<div style="height:260px;display:flex;align-items:center;justify-content:center;color:#8b949e;border:1px dashed #30363d;border-radius:12px;">Enter values and click Predict</div>',
                unsafe_allow_html=True,
            )

    with colB:
        if result and "error" not in result:
            s     = result["strength"]
            color = strength_color(s)
            viol  = result["violations"]
            interval_text = ""
            if result["lower"] is not None and result["upper"] is not None:
                interval_text = (f'<br><span style="font-size:0.8rem;color:#8b949e;">'
                                 f'90% CI: {result["lower"]:.1f} – {result["upper"]:.1f} MPa</span>')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Strength</div>
                <div class="metric-value" style="color:{color}">{s:.1f} MPa</div>
                <div class="metric-label">≈ {grade_label(s)}</div>
                {interval_text}
            </div>
            <div class="metric-card">
                <div class="metric-label">Water / Binder Ratio</div>
                <div class="metric-value">{result['wbr']:.3f}</div>
                <div class="metric-label">Limit ≤ {CONFIG['max_wb']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Estimated Slump</div>
                <div class="metric-value">{result['slump']:.0f} mm</div>
                <div class="metric-label">Target {CONFIG['target_slump_mm'][0]}–{CONFIG['target_slump_mm'][1]} mm</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Approx. Cost</div>
                <div class="metric-value">₹{result['cost']:.0f}/m³</div>
            </div>
            """, unsafe_allow_html=True)
            if viol:
                items = "".join(f"<li>{v}</li>" for v in viol)
                st.markdown(f'<div class="violation-box">⚠️ <strong>Violations:</strong><ul>{items}</ul></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ok-box">✅ All engineering constraints satisfied.</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card"><div class="metric-label">Strength</div><div class="metric-value" style="color:#30363d">— MPa</div></div>
            <div class="metric-card"><div class="metric-label">Water / Binder</div><div class="metric-value" style="color:#30363d">—</div></div>
            <div class="metric-card"><div class="metric-label">Slump</div><div class="metric-value" style="color:#30363d">— mm</div></div>
            <div class="metric-card"><div class="metric-label">Cost</div><div class="metric-value" style="color:#30363d">₹—/m³</div></div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — OPTIMIZE (unchanged)
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Optimize Mix")
    st.caption("Finds the best mix hitting your target strength at minimum cost.")
    c1, c2, c3 = st.columns(3)
    with c1:
        target = st.number_input("Target Strength (MPa)", 20.0, 80.0, 40.0)
    with c2:
        trials = st.slider("Optuna Trials", 50, 300, 100)
    with c3:
        multi = st.checkbox("Pareto Mode", value=True)

    if st.button("Run Optimization", key="btn_opt"):
        with st.spinner("Running optimization… (~20–60 s)"):
            if multi:
                study = multi_objective_study(target, trials)
                rows  = []
                for t in study.best_trials:
                    p   = t.params
                    mix = {
                        "Cement": p["Cement"], "Blast_Furnace_Slag": p["Slag"],
                        "Fly_Ash": p["FlyAsh"], "Water": p["Water"],
                        "Superplasticizer": p["SP"],
                        "Fine_Aggregate": 700, "Coarse_Aggregate": 1000, "Age": 28,
                    }
                    mix = opt_physics(mix)
                    if mix is None: continue
                    sv  = float(pipeline.predict(mix)[0])
                    cv  = sum(mix[k] * COST[k] for k in COST if k in mix)
                    rows.append({
                        "Cement": round(p["Cement"],1), "Slag": round(p["Slag"],1),
                        "Fly Ash": round(p["FlyAsh"],1), "Water": round(p["Water"],1),
                        "SP": round(p["SP"],2), "Strength (MPa)": round(sv,1),
                        "Error (MPa)": round(abs(sv-target),2), "Cost (₹/m³)": round(cv,0),
                    })
                st.session_state.opt_result = {"mode": "pareto", "rows": rows, "target": target}
            else:
                study = single_objective_study(target, trials)
                mix   = best_single_mix(study)
                if mix is None:
                    st.session_state.opt_result = {"mode": "single", "error": "No valid mix found."}
                else:
                    sv = float(pipeline.predict(mix)[0])
                    cv = sum(mix[k] * COST[k] for k in COST if k in mix)
                    _, viol = enforce_constraints(mix)
                    st.session_state.opt_result = {
                        "mode": "single", "mix": mix,
                        "strength": sv, "cost": cv, "violations": viol,
                    }

    opt = st.session_state.opt_result
    opt_slot = st.empty()
    if opt is None:
        opt_slot.markdown(
            '<div style="min-height:200px;display:flex;align-items:center;justify-content:center;color:#8b949e;border:1px dashed #30363d;border-radius:12px;margin-top:1rem;">Set parameters and click Run Optimization</div>',
            unsafe_allow_html=True,
        )
    elif "error" in opt:
        opt_slot.error(opt["error"])
    elif opt["mode"] == "pareto":
        with opt_slot.container():
            rows = opt["rows"]
            if not rows:
                st.warning("No valid Pareto mixes found — try more trials.")
            else:
                df  = pd.DataFrame(rows)
                fig = px.scatter(
                    df, x="Cost (₹/m³)", y="Error (MPa)",
                    color="Strength (MPa)", color_continuous_scale="RdYlGn",
                    hover_data=["Cement","Slag","Fly Ash","Water","SP","Strength (MPa)"],
                    title=f"Pareto Front — Target: {opt['target']} MPa",
                )
                fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
                fig.update_layout(height=400, autosize=False, coloraxis_colorbar=dict(title="Strength<br>(MPa)"))
                st.plotly_chart(fig, use_container_width=True)
                best = df.sort_values("Error (MPa)").iloc[0]
                st.success(f"🏆 Best: **{best['Strength (MPa)']} MPa** | **₹{best['Cost (₹/m³)']:.0f}/m³** | error {best['Error (MPa)']} MPa")
                st.dataframe(df.sort_values("Error (MPa)").reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        with opt_slot.container():
            mix   = opt["mix"]
            sv    = opt["strength"]
            cv    = opt["cost"]
            viol  = opt["violations"]
            binder = mix["Cement"] + mix["Blast_Furnace_Slag"] + mix["Fly_Ash"]
            st.success(f"🏆 {sv:.1f} MPa | ₹{cv:.0f}/m³")
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("Cement",  f"{mix['Cement']:.0f} kg")
            r1c2.metric("Slag",    f"{mix['Blast_Furnace_Slag']:.0f} kg")
            r1c3.metric("Fly Ash", f"{mix['Fly_Ash']:.0f} kg")
            r1c4.metric("Water",   f"{mix['Water']:.0f} kg")
            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            r2c1.metric("SP",     f"{mix['Superplasticizer']:.1f} kg")
            r2c2.metric("Fine",   f"{mix['Fine_Aggregate']:.0f} kg")
            r2c3.metric("Coarse", f"{mix['Coarse_Aggregate']:.0f} kg")
            r2c4.metric("W/B",    f"{mix['Water']/binder:.3f}")
            if viol:
                st.warning("Constraint warnings: " + ", ".join(viol))
            else:
                st.markdown('<div class="ok-box">✅ All engineering constraints satisfied.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — SHAP EXPLAIN (unchanged)
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Explain Prediction")
    st.caption("SHAP waterfall shows how each ingredient pushes strength above/below the model baseline.")
    with st.form("shap_form"):
        sc1, sc2 = st.columns(2)
        with sc1:
            s_cement = st.number_input("Cement",             100, 600,  350, key="s_cement")
            s_slag   = st.number_input("Slag",               0,   300,  50,  key="s_slag")
            s_flyash = st.number_input("Fly Ash",            0,   200,  50,  key="s_flyash")
            s_water  = st.number_input("Water",              100, 250,  180, key="s_water")
        with sc2:
            s_sp     = st.number_input("Superplasticizer",   0,   30,   5,   key="s_sp")
            s_coarse = st.number_input("Coarse Aggregate",   600, 1500, 1000, key="s_coarse")
            s_fine   = st.number_input("Fine Aggregate",     400, 1200, 700,  key="s_fine")
            s_age    = st.number_input("Age (days)",         1,   365,  28,   key="s_age")
        submit = st.form_submit_button("Explain")

    if submit:
        raw_mix = build_mix(s_cement, s_slag, s_flyash, s_water, s_sp, s_coarse, s_fine, s_age)
        try:
            fig, shap_vals = explainer.waterfall_plot(raw_mix)
            reasoning      = explainer.reasoning_text(shap_vals, None)
            st.session_state.shap_result = {"fig": fig, "reasoning": reasoning}
        except Exception as e:
            st.session_state.shap_result = {"error": str(e)}

    shap_slot = st.empty()
    sr = st.session_state.shap_result
    if sr is None:
        shap_slot.markdown(
            '<div style="min-height:340px;display:flex;align-items:center;justify-content:center;color:#8b949e;border:1px dashed #30363d;border-radius:12px;margin-top:1rem;">Fill in the mix above and click Explain</div>',
            unsafe_allow_html=True,
        )
    elif "error" in sr:
        shap_slot.error(f"SHAP failed: {sr['error']}")
    else:
        with shap_slot.container():
            st.pyplot(sr["fig"], use_container_width=True)
            st.markdown(sr["reasoning"])