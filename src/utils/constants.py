# src/utils/constants.py

RAW_FEATURES = [
    'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
    'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age'
]

ENGINEERED_FEATURES = [
    'Water_Binder_Ratio',
    'Log_Age',
    'Cement_x_Age',
    'SCM_Ratio'
]

FINAL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

LOG_COLS = ['Blast_Furnace_Slag', 'Fly_Ash', 'Superplasticizer']

CONFIG = {
    "max_wb": 0.5,
    "min_cement": 320,
    "max_cement": 450,
    "max_scm": 0.6,
    "min_paste": 0.26,
    "max_paste": 0.34,
}

COST = {
    "Cement": 7,
    "Fly_Ash": 2,
    "Blast_Furnace_Slag": 3,
    "Water": 0.05,
    "Superplasticizer": 50,
    "Fine_Aggregate": 1.2,
    "Coarse_Aggregate": 1.5
}