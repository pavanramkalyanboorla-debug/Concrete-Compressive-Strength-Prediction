# app/main.py
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os
import pandas as pd
import logging
from typing import List, Optional
import base64
from io import BytesIO
import matplotlib.pyplot as plt           # ← MISSING import added here

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s - %(message)s')
logger = logging.getLogger("ccsp_api")

app = FastAPI(
    title="Concrete Compressive Strength Prediction API",
    description="Predicts concrete compressive strength (MPa) and optimises mix design.",
    version="2.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------
model_path = os.getenv("MODEL_PATH", os.path.join("artifacts", "model.pkl"))
preprocessor_path = os.getenv("PREPROCESSOR_PATH", os.path.join("artifacts", "preprocessor.pkl"))
model = None
preprocessor = None
load_error: Optional[str] = None

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    logger.info("Model loaded from %s", model_path)
except FileNotFoundError as e:
    load_error = f"Model or preprocessor file not found: {e}"
    logger.exception(load_error)
except Exception as e:
    load_error = f"Failed to load model or preprocessor: {e}"
    logger.exception(load_error)

# ------------------------------------------------------------
# Schemas (unchanged)
# ------------------------------------------------------------
class PredictionInput(BaseModel):
    Cement: float = Field(..., gt=0)
    Blast_Furnace_Slag: float = Field(..., ge=0)
    Fly_Ash: float = Field(..., ge=0)
    Water: float = Field(..., gt=0)
    Superplasticizer: float = Field(..., ge=0)
    Coarse_Aggregate: float = Field(..., gt=0)
    Fine_Aggregate: float = Field(..., gt=0)
    Age: int = Field(..., gt=0)

class PredictionResponse(BaseModel):
    predicted_strength_mpa: float
    strength_category: str

class MixRequest(BaseModel):
    target_strength: float = 40.0
    n_trials: int = 100
    multi_objective: bool = False

FEATURE_COLS = [
    'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
    'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age',
    'Water_Binder_Ratio', 'Log_Age', 'Cement_x_Age', 'SCM_Ratio'
]

def get_strength_category(mpa: float) -> str:
    # (unchanged)
    if mpa < 20:
        return "Low Strength (< 20 MPa)"
    elif mpa < 40:
        return "Normal Strength (20–40 MPa)"
    elif mpa < 60:
        return "High Strength (40–60 MPa)"
    else:
        return "Ultra High Strength (> 60 MPa)"

def _feature_engineer(input_df: pd.DataFrame) -> pd.DataFrame:
    # (unchanged)
    df = input_df.copy()
    binder = df['Cement'] + df['Blast_Furnace_Slag'] + df['Fly_Ash']
    if (binder <= 0).any():
        raise ValueError("Binder must be positive")
    df['Water_Binder_Ratio'] = df['Water'] / binder
    df['Log_Age'] = np.log(df['Age'].replace(0, np.nan))
    df['Cement_x_Age'] = df['Cement'] * df['Age']
    df['SCM_Ratio'] = (df['Blast_Furnace_Slag'] + df['Fly_Ash']) / df['Cement'].replace(0, np.nan)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df[FEATURE_COLS]

# ------------------------------------------------------------
# Health / Metadata (unchanged)
# ------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Concrete Compressive Strength Prediction API v2.0"}

@app.get("/health")
def health_check():
    if load_error:
        return {"status": "unhealthy", "details": load_error}
    if model is None or preprocessor is None:
        return {"status": "unhealthy", "details": "Model not loaded"}
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_type": type(model).__name__ if model else None,
        "preprocessor_type": type(preprocessor).__name__ if preprocessor else None,
        "input_features": list(PredictionInput.__fields__.keys()),
        "engineered_features": ["Water_Binder_Ratio", "Log_Age", "Cement_x_Age", "SCM_Ratio"],
    }

# ------------------------------------------------------------
# Prediction endpoints (unchanged)
# ------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict_strength(input_data: PredictionInput):
    logger.info("Predict request: %s", input_data)
    if load_error:
        raise HTTPException(status_code=503, detail=load_error)
    input_df = pd.DataFrame([input_data.dict()])
    engineered = _feature_engineer(input_df)
    scaled = preprocessor.transform(engineered)
    pred = float(model.predict(scaled)[0])
    return PredictionResponse(predicted_strength_mpa=pred,
                              strength_category=get_strength_category(pred))

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(inputs: List[PredictionInput]):
    logger.info("Batch predict: %d items", len(inputs))
    if not inputs:
        raise HTTPException(status_code=422, detail="Input list cannot be empty")
    input_df = pd.DataFrame([item.dict() for item in inputs])
    engineered = _feature_engineer(input_df)
    scaled = preprocessor.transform(engineered)
    preds = model.predict(scaled)
    return [PredictionResponse(predicted_strength_mpa=float(p),
                               strength_category=get_strength_category(float(p)))
            for p in preds]

# ------------------------------------------------------------
# Optimisation endpoint (unchanged)
# ------------------------------------------------------------
from src.optimization import (single_objective_study, multi_objective_study,
                              best_single_mix, _predict, _cost, _physics, _constraints)

@app.post("/optimize")
def optimize_mix(req: MixRequest):
    # ... (exactly as before) ...
    if req.multi_objective:
        study = multi_objective_study(req.target_strength, req.n_trials)
        pareto = []
        for t in study.best_trials:
            p = t.params
            mix = {
                "Cement": p["Cement"], "Blast_Furnace_Slag": p["Slag"],
                "Fly_Ash": p["FlyAsh"], "Water": p["Water"],
                "Superplasticizer": p["SP"],
                "Fine_Aggregate": 700, "Coarse_Aggregate": 1000, "Age": 28
            }
            mix = _physics(mix)
            if mix is None:
                continue
            strength = float(_predict(mix))
            cost = float(_cost(mix))
            violations = _constraints(mix)
            pareto.append({
                "mix": {k: round(v, 2) for k, v in mix.items()
                        if k in ['Cement','Blast_Furnace_Slag','Fly_Ash','Water',
                                 'Superplasticizer','Fine_Aggregate','Coarse_Aggregate']},
                "predicted_strength": round(strength, 2),
                "cost": round(cost, 2),
                "violations": violations,
                "strength_error": round(abs(strength - req.target_strength), 2)
            })
        return {"pareto_front": pareto}
    else:
        study = single_objective_study(req.target_strength, req.n_trials)
        mix = best_single_mix(study)
        strength = float(_predict(mix))
        cost = float(_cost(mix))
        return {
            "mix": {k: round(v, 2) for k, v in mix.items()
                    if k in ['Cement','Blast_Furnace_Slag','Fly_Ash','Water',
                             'Superplasticizer','Fine_Aggregate','Coarse_Aggregate']},
            "predicted_strength": round(strength, 2),
            "cost": round(cost, 2),
            "violations": _constraints(mix)
        }

# ------------------------------------------------------------
# SHAP endpoint – FIXED
# ------------------------------------------------------------
from src.shap_utils import explain_with_shap_fast   # <-- optimised function (see below)

@app.get("/shap")
def get_shap_plot(
    cement: float, slag: float = 0, flyash: float = 0,
    water: float = 180, sp: float = 5, coarse: float = 1000,
    fine: float = 700, age: int = 28
):
    mix = {
        "Cement": cement, "Blast_Furnace_Slag": slag, "Fly_Ash": flyash,
        "Water": water, "Superplasticizer": sp,
        "Coarse_Aggregate": coarse, "Fine_Aggregate": fine, "Age": age
    }
    shap_values, fig = explain_with_shap_fast(mix)   # computes only once
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return {"shap_plot": f"data:image/png;base64,{img_str}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)