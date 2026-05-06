"""
Compute 90% prediction intervals using 5‑fold cross‑validation residuals.
Saves artifacts/uncertainty_params.pkl (model + preprocessor + std).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_predict, KFold

# ------------------------------------------------------------
# 1. Load raw training data
# ------------------------------------------------------------
df_train = pd.read_csv("artifacts/train.csv")
target_col = "Concrete_Strength"
X_raw = df_train.drop(columns=[target_col])
y = df_train[target_col]

# ------------------------------------------------------------
# 2. Feature engineering (match API exactly)
# ------------------------------------------------------------
FEATURE_COLS = [
    'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
    'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age',
    'Water_Binder_Ratio', 'Log_Age', 'Cement_x_Age', 'SCM_Ratio'
]

def _feature_engineer(input_df: pd.DataFrame) -> pd.DataFrame:
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

X_eng = _feature_engineer(X_raw)

# ------------------------------------------------------------
# 3. Load preprocessor and model
# ------------------------------------------------------------
preprocessor = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")
X_scaled = preprocessor.transform(X_eng)

# ------------------------------------------------------------
# 4. Cross‑validated residuals for interval width
# ------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv)
sigma = np.std(y - y_pred_cv)

print(f"Cross‑validated residual std: {sigma:.2f} MPa")
print("→ 90% interval = prediction ± 1.645 × σ")

# ------------------------------------------------------------
# 5. Save all needed parameters
# ------------------------------------------------------------
params = {"model": model, "preprocessor": preprocessor, "cv_std": sigma}
joblib.dump(params, "artifacts/uncertainty_params.pkl")
print("✅ Saved artifacts/uncertainty_params.pkl")