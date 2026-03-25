from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os
import pandas as pd

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

# Load the model and preprocessor
model_path = os.path.join("artifacts", "model.pkl")
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Model or preprocessor file not found: {e}")

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
    predicted_strength: float
    strength_category: str

def get_strength_category(mpa: float) -> str:
    if mpa < 20:
        return "Low Strength (< 20 MPa)"
    elif mpa < 40:
        return "Normal Strength (20–40 MPa)"
    elif mpa < 60:
        return "High Strength (40–60 MPa)"
    else:
        return "Ultra High Strength (> 60 MPa)"

@app.get("/")
def read_root():
    return {"message": "Concrete Compressive Strength Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_strength(input_data: PredictionInput):
    try:
        # Create input dataframe
        input_df = pd.DataFrame([input_data.dict()])

        # Compute engineered features
        binder = input_df['Cement'] + input_df['Blast_Furnace_Slag'] + input_df['Fly_Ash']
        input_df['Water_Binder_Ratio'] = input_df['Water'] / binder
        input_df['Log_Age'] = np.log(input_df['Age'])
        input_df['Cement_x_Age'] = input_df['Cement'] * input_df['Age']
        input_df['SCM_Ratio'] = (input_df['Blast_Furnace_Slag'] + input_df['Fly_Ash']) / input_df['Cement']

        # Preprocess the input
        input_scaled = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)

        strength = float(prediction[0])
        category = get_strength_category(strength)

        return PredictionResponse(predicted_strength=strength, strength_category=category)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 