import os
import sys
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.utils.utils import load_object


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
@dataclass
class PredictPipelineConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


# ------------------------------------------------------------
# Feature engineering (mirrors training EXACTLY)
# ------------------------------------------------------------
FEATURE_COLS = [
    'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
    'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age',
    'Water_Binder_Ratio', 'Log_Age', 'Cement_x_Age', 'SCM_Ratio'
]

LOG_COLS = ['Blast_Furnace_Slag', 'Fly_Ash', 'Superplasticizer']  # ← NEW


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the same 12 engineered features used during training.
    NOTE: log1p is NOT applied here – the preprocessor handles it now.
    """
    df = df.copy()
    binder = df['Cement'] + df['Blast_Furnace_Slag'] + df['Fly_Ash']
    df['Water_Binder_Ratio'] = df['Water'] / binder.replace(0, np.nan)
    df['Log_Age'] = np.log(df['Age'].replace(0, np.nan))
    df['Cement_x_Age'] = df['Cement'] * df['Age']
    df['SCM_Ratio'] = (df['Blast_Furnace_Slag'] + df['Fly_Ash']) / df['Cement'].replace(0, np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df[FEATURE_COLS]


# ------------------------------------------------------------
# Prediction Pipeline Class
# ------------------------------------------------------------
class PredictPipeline:
    def __init__(self, config: PredictPipelineConfig = None):
        self.config = config or PredictPipelineConfig()
        self.model = None
        self.preprocessor = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model and preprocessor from disk."""
        try:
            logging.info("Loading model and preprocessor...")
            self.model = load_object(self.config.model_path)
            self.preprocessor = load_object(self.config.preprocessor_path)
            logging.info("Artifacts loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data):
        """
        Main prediction method.
        Accepts: dict, list of dicts, numpy array (8 cols), or DataFrame.
        Returns: numpy array of predicted strengths.
        """
        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            elif isinstance(input_data, np.ndarray):
                cols = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
                        'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age']
                df = pd.DataFrame(input_data, columns=cols)
            elif isinstance(input_data, pd.DataFrame):
                df = input_data.copy()
            else:
                raise ValueError("Unsupported input type.")

            # Ensure required raw columns exist
            required_raw = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
                            'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age']
            missing = set(required_raw) - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns in input: {missing}")

            # 1. Engineer features
            df_eng = engineer_features(df)

            # 2. The preprocessor now handles log1p + scaling in one go
            scaled = self.preprocessor.transform(df_eng)

            # 3. Predict
            predictions = self.model.predict(scaled)
            return predictions

        except Exception as e:
            raise CustomException(e, sys)


# ------------------------------------------------------------
# Convenience function for single prediction
# ------------------------------------------------------------
def predict_strength(cement, blast_furnace_slag, fly_ash, water, superplasticizer,
                     coarse_aggregate, fine_aggregate, age):
    pipeline = PredictPipeline()
    input_dict = {
        'Cement': cement, 'Blast_Furnace_Slag': blast_furnace_slag,
        'Fly_Ash': fly_ash, 'Water': water,
        'Superplasticizer': superplasticizer,
        'Coarse_Aggregate': coarse_aggregate,
        'Fine_Aggregate': fine_aggregate,
        'Age': age
    }
    result = pipeline.predict(input_dict)
    return float(result[0])


# ------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        'Cement': 300, 'Blast_Furnace_Slag': 100, 'Fly_Ash': 100,
        'Water': 180, 'Superplasticizer': 5,
        'Coarse_Aggregate': 1000, 'Fine_Aggregate': 700,
        'Age': 28
    }
    try:
        preds = PredictPipeline().predict(sample)
        print(f"Predicted strength: {preds[0]:.2f} MPa")
    except Exception as e:
        print("Error:", e)
        raise