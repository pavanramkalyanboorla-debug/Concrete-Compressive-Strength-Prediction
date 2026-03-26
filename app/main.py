from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os
import pandas as pd
import logging
from typing import List, Optional

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s - %(message)s'
)
logger = logging.getLogger("ccsp_api")

app = FastAPI(
    title="Concrete Compressive Strength Prediction API",
    description="Predicts concrete compressive strength (MPa) based on mix composition and age.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and preprocessor (from env vars if provided)
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
    logger.info("Preprocessor loaded from %s", preprocessor_path)
except FileNotFoundError as e:
    load_error = f"Model or preprocessor file not found: {e}"
    logger.exception(load_error)
except Exception as e:
    load_error = f"Failed to load model or preprocessor: {e}"
    logger.exception(load_error)

class PredictionInput(BaseModel):
    Cement: float = Field(..., gt=0, description="Cement content in kg/m³")
    Blast_Furnace_Slag: float = Field(..., ge=0, description="Blast Furnace Slag content in kg/m³")
    Fly_Ash: float = Field(..., ge=0, description="Fly Ash content in kg/m³")
    Water: float = Field(..., gt=0, description="Water content in kg/m³")
    Superplasticizer: float = Field(..., ge=0, description="Superplasticizer content in kg/m³")
    Coarse_Aggregate: float = Field(..., gt=0, description="Coarse Aggregate content in kg/m³")
    Fine_Aggregate: float = Field(..., gt=0, description="Fine Aggregate content in kg/m³")
    Age: int = Field(..., gt=0, description="Age of concrete in days")

class PredictionResponse(BaseModel):
    predicted_strength_mpa: float
    strength_category: str
    feature_importance: dict = {}

def get_strength_category(mpa: float) -> str:
    if mpa < 20:
        return "Low Strength (< 20 MPa)"
    elif mpa < 40:
        return "Normal Strength (20–40 MPa)"
    elif mpa < 60:
        return "High Strength (40–60 MPa)"
    else:
        return "Ultra High Strength (> 60 MPa)"

def _check_and_feature_engineer(input_df: pd.DataFrame) -> pd.DataFrame:
    binder = input_df['Cement'] + input_df['Blast_Furnace_Slag'] + input_df['Fly_Ash']

    if (binder <= 0).any():
        raise ValueError("Binder must be positive for stable Water/Binder ratio")

    input_df = input_df.copy()
    input_df['Water_Binder_Ratio'] = input_df['Water'] / binder
    input_df['Log_Age'] = np.log(input_df['Age'])
    input_df['Cement_x_Age'] = input_df['Cement'] * input_df['Age']
    input_df['SCM_Ratio'] = (input_df['Blast_Furnace_Slag'] + input_df['Fly_Ash']) / input_df['Cement']

    if not np.isfinite(input_df[['Water_Binder_Ratio', 'Log_Age', 'Cement_x_Age', 'SCM_Ratio']]).all().all():
        raise ValueError("Engineered features contain non-finite values")

    return input_df

def _predict_from_df(input_df: pd.DataFrame) -> List[PredictionResponse]:
    if load_error is not None:
        logger.error("Health check failed: %s", load_error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=load_error)

    if model is None or preprocessor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model or preprocessor is not loaded")

    engineered_df = _check_and_feature_engineer(input_df)
    try:
        processed = preprocessor.transform(engineered_df)
        predictions = model.predict(processed)
    except Exception as e:
        logger.exception("Prediction pipeline error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Prediction pipeline error: {e}")

    outputs = []
    default_importance = {f: 0.0 for f in input_df.columns}
    for p in predictions:
        strength = float(p)
        outputs.append(PredictionResponse(predicted_strength_mpa=strength,
                                          strength_category=get_strength_category(strength),
                                          feature_importance=default_importance))

    return outputs

@app.get("/")
def read_root():
    return {"message": "Concrete Compressive Strength Prediction API"}

@app.get("/health")
def health_check():
    if load_error:
        return {"status": "unhealthy", "details": load_error}
    if model is None or preprocessor is None:
        return {"status": "unhealthy", "details": "Model or preprocessor not loaded"}
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    model_version = None
    if model is not None:
        model_version = getattr(model, 'version', None) or getattr(model, '__version__', None)

    features = list(PredictionInput.__fields__.keys())
    engineered_features = ["Water_Binder_Ratio", "Log_Age", "Cement_x_Age", "SCM_Ratio"]

    return {
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_type": type(model).__name__ if model is not None else None,
        "preprocessor_type": type(preprocessor).__name__ if preprocessor is not None else None,
        "model_version": model_version,
        "input_features": features,
        "engineered_features": engineered_features
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_strength(input_data: PredictionInput):
    logger.info("Predict request received: %s", input_data)
    input_df = pd.DataFrame([input_data.dict()])

    outputs = _predict_from_df(input_df)
    return outputs[0]

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(inputs: List[PredictionInput]):
    logger.info("Batch predict request received: %d items", len(inputs))
    if not inputs:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Input list cannot be empty")

    input_df = pd.DataFrame([item.dict() for item in inputs])
    return _predict_from_df(input_df)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

 