# app/streamlit_app.py
# Concrete Mix Optimizer — Premium Refined Dark UI

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib

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
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PREMIUM REFINED DARK ENGINEERING CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global overrides ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: #0a0a0f;
    }
    .stApp {
        background: radial-gradient(ellipse at 20% 50%, #13162a 0%, #0a0a0f 70%);
    }

    /* ── Full-width header with soft glow ── */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: left;
        background: linear-gradient(135deg, #a78bfa, #7c3aed, #6d28d9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
        filter: drop-shadow(0 0 8px rgba(124, 58, 237, 0.4));
        white-space: nowrap;
        overflow: visible;
    }
    .subtext {
        text-align: left;
        color: #8b8fa3;
        font-size: 1.05rem;
        font-weight: 300;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }

    /* ── LEFT SIDEBAR NAVIGATION ── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 25, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(124, 58, 237, 0.15);
    }
    /* Navigation buttons */
    div[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 12px;
        padding: 16px 24px;
        text-align: left;
        font-size: 1rem;
        font-weight: 500;
        color: #c7c9d4;
        cursor: pointer;
        margin-bottom: 0.6rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.3px;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(124, 58, 237, 0.12);
        border-color: #a78bfa;
        transform: translateX(4px);
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.15);
    }
    /* Active navigation state (handled via Python logic) */
    div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
        color: #ffffff;
        font-weight: 600;
        border-color: transparent;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.5);
    }

    /* ── METRIC CARDS (glassmorphism) ── */
    .metric-card {
        background: rgba(25, 25, 45, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        border: 1px solid rgba(124, 58, 237, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 0.9rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 88px;
    }
    .metric-card:hover {
        border-color: rgba(124, 58, 237, 0.4);
        background: rgba(35, 35, 60, 0.8);
        box-shadow: 0 12px 40px rgba(124, 58, 237, 0.2);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.78rem;
        color: #a0a0b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f1f1f3;
        letter-spacing: -0.3px;
    }

    /* ── Constraint status boxes ── */
    .violation-box {
        background: rgba(220, 38, 38, 0.08);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(220, 38, 38, 0.3);
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        color: #fca5a5;
        font-size: 0.88rem;
        min-height: 48px;
        margin-top: 0.8rem;
    }
    .ok-box {
        background: rgba(16, 185, 129, 0.06);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        color: #86efac;
        font-size: 0.9rem;
        min-height: 48px;
        margin-top: 0.8rem;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
        color: #ffffff;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        padding: 0.7rem 1.5rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.3px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #a78bfa, #7c3aed);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── INPUT FIELDS ── */
    [data-testid="stNumberInput"] input {
        border-radius: 10px;
        border: 1px solid rgba(124, 58, 237, 0.2);
        background: rgba(255,255,255,0.04);
        color: #e0e2ec;
        padding: 0.6rem 0.8rem;
        transition: all 0.2s;
    }
    [data-testid="stNumberInput"] input:focus {
        border-color: #a78bfa;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2);
    }

    /* ── SLIDER ── */
    div[data-testid="stSlider"] .st-bv {
        background: linear-gradient(90deg, #7c3aed, #a78bfa);
    }

    /* ── OPTIMIZATION SUCCESS CARD ── */
    .success-card {
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(124,58,237,0.1) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);
    }

    /* ── SCROLLBAR (thin, elegant) ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb {
        background: #2d2d4a;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #3d3d5c; }

    /* ── PLACEHOLDERS ── */
    .placeholder-box {
        height:280px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:#8b8fa3;
        border:2px dashed rgba(124,58,237,0.2);
        border-radius:18px;
        background:rgba(124,58,237,0.03);
        backdrop-filter: blur(4px);
    }
</style>
""", unsafe_allow_html=True)

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
        return joblib.load("artifacts/uncertainty_params.pkl")
    except FileNotFoundError:
        return None

pipeline = load_pipeline()
explainer = load_explainer(pipeline)
uncertainty_params = load_uncertainty_params()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key in ("pred_result", "opt_result", "shap_result", "active_tab"):
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.active_tab is None:
    st.session_state.active_tab = "Predict"

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
    if s < 25:  return "#ef4444"
    if s < 40:  return "#f59e0b"
    return "#10b981"

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🧱 Concrete Mix Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">AI‑powered mix design · Physics‑constrained · 90% CI · SHAP explainability</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# LEFT SIDEBAR NAVIGATION (vertical tabs)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Navigation")
    # Use buttons to simulate vertical tabs; store active state in session
    st.button("📊 Predict Strength", key="nav_predict", on_click=lambda: st.session_state.update(active_tab="Predict"))
    st.button("⚙️ Optimize Mix", key="nav_optimize", on_click=lambda: st.session_state.update(active_tab="Optimize"))
    st.button("🧠 Explain Prediction", key="nav_explain", on_click=lambda: st.session_state.update(active_tab="Explain"))

# ─────────────────────────────────────────────
# MAIN CONTENT AREA (switches based on active tab)
# ─────────────────────────────────────────────
if st.session_state.active_tab == "Predict":
    # ════════════════════════ TAB 1 — PREDICT ════════════════════════
    st.subheader("Concrete Compressive Strength Predictor")
    st.caption("Enter your mix proportions below. Results include 90% confidence intervals, slump estimate, and constraint checks.")

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

    if st.button("⚡ Predict Strength", key="btn_predict"):
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
                number={"suffix": " MPa", "font": {"color": color, "size": 30, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b8fa3"},
                    "bar":  {"color": color, "thickness": 0.25},
                    "bgcolor": "rgba(255,255,255,0.03)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 25],   "color": "rgba(239,68,68,0.15)"},
                        {"range": [25, 40],  "color": "rgba(245,158,11,0.15)"},
                        {"range": [40, 100], "color": "rgba(16,185,129,0.15)"},
                    ],
                },
            ))
            fig.update_layout(margin=dict(t=40, b=10, l=30, r=30), height=280, autosize=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#8b8fa3", family="Inter"))
            gauge_slot.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})
        elif result and "error" in result:
            gauge_slot.error(f"❌ {result['error']}")
        else:
            gauge_slot.markdown(
                '<div class="placeholder-box">Enter values and click <strong>Predict Strength</strong></div>',
                unsafe_allow_html=True,
            )
    with colB:
        if result and "error" not in result:
            s     = result["strength"]
            color = strength_color(s)
            viol  = result["violations"]
            interval_text = ""
            if result["lower"] is not None and result["upper"] is not None:
                interval_text = (f'<span style="font-size:0.85rem;color:#8b8fa3;">90% CI: '
                                 f'<strong style="color:#c4b5fd;">{result["lower"]:.1f}</strong> – '
                                 f'<strong style="color:#c4b5fd;">{result["upper"]:.1f}</strong> MPa</span>')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Strength</div>
                <div class="metric-value" style="color:{color};font-size:1.7rem;">{s:.1f} MPa</div>
                <div style="color:#8b8fa3;font-size:0.85rem;margin-top:0.1rem;">≈ {grade_label(s)}</div>
                {interval_text}
            </div>
            <div class="metric-card">
                <div class="metric-label">Water / Binder Ratio</div>
                <div class="metric-value">{result['wbr']:.3f}</div>
                <div style="color:#8b8fa3;font-size:0.8rem;">Limit ≤ {CONFIG['max_wb']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Estimated Slump</div>
                <div class="metric-value">{result['slump']:.0f} mm</div>
                <div style="color:#8b8fa3;font-size:0.8rem;">Target {CONFIG['target_slump_mm'][0]}–{CONFIG['target_slump_mm'][1]} mm</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Approx. Cost</div>
                <div class="metric-value" style="color:#f59e0b;">₹ {result['cost']:.0f} /m³</div>
            </div>
            """, unsafe_allow_html=True)
            if viol:
                items = "".join(f"<li>{v}</li>" for v in viol)
                st.markdown(f'<div class="violation-box">⚠️ <strong>Constraint Violations:</strong><ul style="margin:0.3rem 0 0 1rem;">{items}</ul></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="ok-box">✅ All engineering constraints satisfied.</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card"><div class="metric-label">Strength</div>
            <div class="metric-value" style="color:#2a2d3e;">— MPa</div></div>
            <div class="metric-card"><div class="metric-label">Water / Binder</div>
            <div class="metric-value" style="color:#2a2d3e;">—</div></div>
            <div class="metric-card"><div class="metric-label">Slump</div>
            <div class="metric-value" style="color:#2a2d3e;">— mm</div></div>
            <div class="metric-card"><div class="metric-label">Cost</div>
            <div class="metric-value" style="color:#2a2d3e;">₹ —/m³</div></div>
            """, unsafe_allow_html=True)

elif st.session_state.active_tab == "Optimize":
    # ════════════════════════ TAB 2 — OPTIMIZE ════════════════════════
    st.subheader("Mix Optimization Engine")
    st.caption("Optuna searches the feasible space to find the cheapest mix that hits your target strength.")
    c1, c2, c3 = st.columns(3)
    with c1:
        target = st.number_input("Target Strength (MPa)", 20.0, 80.0, 40.0)
    with c2:
        trials = st.slider("Optuna Trials", 50, 300, 100)
    with c3:
        multi = st.checkbox("Pareto Mode", value=True, help="Show full cost-vs-error trade-off frontier")
    if st.button("🔍 Run Optimization", key="btn_opt"):
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
                    if mix is None:
                        continue
                    sv  = float(pipeline.predict(mix)[0])
                    cv  = sum(mix[k] * COST[k] for k in COST if k in mix)
                    rows.append({
                        "Cement":         round(p["Cement"], 1),
                        "Slag":           round(p["Slag"], 1),
                        "Fly Ash":        round(p["FlyAsh"], 1),
                        "Water":          round(p["Water"], 1),
                        "SP":             round(p["SP"], 2),
                        "Strength (MPa)": round(sv, 1),
                        "Error (MPa)":    round(abs(sv - target), 2),
                        "Cost (₹/m³)":   round(cv, 0),
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
            '<div class="placeholder-box">Set parameters and click <strong>Run Optimization</strong></div>',
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
                    hover_data=["Cement", "Slag", "Fly Ash", "Water", "SP", "Strength (MPa)"],
                    title=f"Pareto Front — Target: {opt['target']} MPa",
                )
                fig.update_traces(marker=dict(size=11, line=dict(width=1.5, color="white")))
                fig.update_layout(
                    height=420, autosize=False,
                    coloraxis_colorbar=dict(title="Strength<br>(MPa)"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b8fa3", family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)
                best = df.sort_values("Error (MPa)").iloc[0]
                st.markdown(f"""
                <div class="success-card">
                    🏆 <strong>Best trade‑off:</strong> {best['Strength (MPa)']} MPa |
                    ₹ {best['Cost (₹/m³)']:,.0f} /m³ | error {best['Error (MPa)']} MPa
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(df.sort_values("Error (MPa)").reset_index(drop=True),
                             use_container_width=True, hide_index=True)
    else:
        with opt_slot.container():
            mix   = opt["mix"]
            sv    = opt["strength"]
            cv    = opt["cost"]
            viol  = opt["violations"]
            binder = mix["Cement"] + mix["Blast_Furnace_Slag"] + mix["Fly_Ash"]
            st.markdown(f"""
            <div class="success-card" style="text-align:center;">
                🏆 <strong style="font-size:1.2rem;">{sv:.1f} MPa</strong> &nbsp;|&nbsp;
                <strong style="color:#f59e0b;">₹ {cv:,.0f} /m³</strong>
            </div>
            """, unsafe_allow_html=True)
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
                st.markdown('<div class="ok-box">✅ All engineering constraints satisfied.</div>',
                            unsafe_allow_html=True)
else:  # Explain tab
    # ════════════════════════ TAB 3 — SHAP EXPLAIN ════════════════════════
    st.subheader("SHAP Explainability")
    st.caption("Understand exactly how each ingredient drives the predicted strength above or below baseline.")
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
        submit = st.form_submit_button("🔬 Explain")
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
            '<div class="placeholder-box">Fill in the mix above and click <strong>Explain</strong></div>',
            unsafe_allow_html=True,
        )
    elif "error" in sr:
        shap_slot.error(f"SHAP failed: {sr['error']}")
    else:
        with shap_slot.container():
            st.pyplot(sr["fig"], use_container_width=True)
            st.markdown(sr["reasoning"])