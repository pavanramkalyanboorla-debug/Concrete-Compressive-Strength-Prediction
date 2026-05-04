import os
import sys
from src.utils.exceptions import CustomException
from src.utils.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # --- CHANGED: now reads the RAW Excel file, NOT the pre‑transformed CSV ---
            raw_path = os.path.join("data", "Dataset2.xlsx")
            col_names = [
                'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
                'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate',
                'Age', 'Concrete_Strength'
            ]
            df = pd.read_excel(raw_path, names=col_names)
            logging.info("Read the raw dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),
                        exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,
                             index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info(f"Error occurred in data ingestion component: {e}")
            raise CustomException(e, sys)