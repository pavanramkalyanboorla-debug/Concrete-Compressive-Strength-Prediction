# src/pipeline/training_pipeline.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.utils.exceptions import CustomException
from src.utils.logger import logging

if __name__ == "__main__":
    try:
        # 1. Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data loaded: train={train_path}, test={test_path}")

        # 2. Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info(f"Transformation done. Preprocessor saved to {preprocessor_path}")

        # 3. Model Training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model training completed. R2 Score: {r2_score}")

    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        raise CustomException(e, sys)