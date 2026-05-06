"""Compare Abram's Law against your model on the same test set."""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the test data (already split)
df_train = pd.read_csv("artifacts/train.csv")
df_test  = pd.read_csv("artifacts/test.csv")
target   = "Concrete_Strength"

# Helper to add 1/(water/binder) for Abram's Law
def add_inv_wb(df):
    df = df.copy()
    binder = df["Cement"] + df["Blast_Furnace_Slag"] + df["Fly_Ash"]
    df["w_b_ratio"] = df["Water"] / binder
    df["inv_wb"] = 1 / df["w_b_ratio"]
    return df

train_ab = add_inv_wb(df_train)
test_ab  = add_inv_wb(df_test)

# Abram's Law: linear regression on inverse water‑binder ratio
X_tr_ab = train_ab[["inv_wb"]]
y_tr_ab = train_ab[target]
X_te_ab = test_ab[["inv_wb"]]
y_te_ab = test_ab[target]

abrams = LinearRegression().fit(X_tr_ab, y_tr_ab)
y_pred_ab = abrams.predict(X_te_ab)

mae_ab = metrics.mean_absolute_error(y_te_ab, y_pred_ab)
r2_ab  = metrics.r2_score(y_te_ab, y_pred_ab)

# -----------------------------------------------------------------
# Your ML model (full feature engineering)
# -----------------------------------------------------------------

FEATURE_COLS = [
    'Cement','Blast_Furnace_Slag','Fly_Ash','Water','Superplasticizer',
    'Coarse_Aggregate','Fine_Aggregate','Age',
    'Water_Binder_Ratio','Log_Age','Cement_x_Age','SCM_Ratio'
]

def engineer_ml(df):
    df = df.copy()
    binder = df['Cement'] + df['Blast_Furnace_Slag'] + df['Fly_Ash']
    df['Water_Binder_Ratio'] = df['Water'] / binder
    df['Log_Age'] = np.log(df['Age'].replace(0, np.nan))
    df['Cement_x_Age'] = df['Cement'] * df['Age']
    df['SCM_Ratio'] = (df['Blast_Furnace_Slag'] + df['Fly_Ash']) / df['Cement'].replace(0, np.nan)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df[FEATURE_COLS]

model        = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

X_ml_eng    = engineer_ml(df_test)
X_ml_scaled = preprocessor.transform(X_ml_eng)
y_pred_ml   = model.predict(X_ml_scaled)

mae_ml = metrics.mean_absolute_error(df_test[target], y_pred_ml)
r2_ml  = metrics.r2_score(df_test[target], y_pred_ml)

print("======================================================")
print(f"Abram's Law : MAE = {mae_ab:.2f} MPa, R² = {r2_ab:.3f}")
print(f"Your Model  : MAE = {mae_ml:.2f} MPa, R² = {r2_ml:.3f}")
print("======================================================")