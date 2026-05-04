import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import os
import sys

from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.utils.utils import save_object


# --- NEW: helper to add engineered features ---
def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Water_Binder_Ratio, Log_Age, Cement_x_Age, SCM_Ratio."""
    df = df.copy()
    binder = df['Cement'] + df['Blast_Furnace_Slag'] + df['Fly_Ash']
    df['Water_Binder_Ratio'] = df['Water'] / binder.replace(0, np.nan)
    df['Log_Age'] = np.log(df['Age'].replace(0, np.nan))
    df['Cement_x_Age'] = df['Cement'] * df['Age']
    df['SCM_Ratio'] = (df['Blast_Furnace_Slag'] + df['Fly_Ash']) / df['Cement'].replace(0, np.nan)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Build a ColumnTransformer that:
          1. applies log1p to skewed columns
          2. scales ALL 12 numerical columns
        The final preprocessor is a Pipeline of (log1p → scaler).
        """
        try:
            LOG_COLS = ['Blast_Furnace_Slag', 'Fly_Ash', 'Superplasticizer']
            NUMERICAL_COLS = [
                'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
                'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age',
                'Water_Binder_Ratio', 'Log_Age', 'Cement_x_Age', 'SCM_Ratio'
            ]

            # Step 1: log1p on the 3 skewed columns
            log_transformer = FunctionTransformer(np.log1p, validate=True)

            # Step 2: standard scaler on all 12
            scaler = StandardScaler()

            # Combine them in a pipeline
            preprocessor = Pipeline(steps=[
                ('log1p', ColumnTransformer(
                    transformers=[('log', log_transformer, LOG_COLS)],
                    remainder='passthrough'
                )),
                ('scaler', scaler)
            ])

            logging.info(f"Numerical columns: {NUMERICAL_COLS}")
            logging.info(f"Log1p columns: {LOG_COLS}")
            return preprocessor
        except Exception as e:
            logging.info(f"Error occurred: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # --- NEW: engineer features BEFORE scaling ---
            train_df = _add_engineered_features(train_df)
            test_df = _add_engineered_features(test_df)
            logging.info("Engineered features added")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'Concrete_Strength'
            FEATURE_COLS = [
                'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
                'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age',
                'Water_Binder_Ratio', 'Log_Age', 'Cement_x_Age', 'SCM_Ratio'
            ]

            input_feature_train_df = train_df[FEATURE_COLS]
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df[FEATURE_COLS]
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr, test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info(f"Error occurred: {e}")
            raise CustomException(e, sys)