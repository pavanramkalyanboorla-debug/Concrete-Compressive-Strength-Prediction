# Physical constants, cost, and engineering limits

SPECIFIC_GRAVITIES = {
    'Cement': 3.15, 'Fly_Ash': 2.2, 'Blast_Furnace_Slag': 2.9,
    'Water': 1.0, 'Fine_Aggregate': 2.65, 'Coarse_Aggregate': 2.70
}
ABSORPTION = {'Fine_Aggregate': 0.02, 'Coarse_Aggregate': 0.01}
MOISTURE = {'Fine_Aggregate': 0.04, 'Coarse_Aggregate': 0.02}
AIR_CONTENT = 0.02

CONFIG = {
    "max_wb": 0.50,
    "min_cement": 320,
    "max_cement": 450,
    "max_scm": 0.6,
    "min_paste": 0.26,
    "max_paste": 0.34,
    "target_slump_mm": (50, 100),
    "fine_aggregate_ratio": 0.4
}

COST = {
    "Cement": 7, "Fly_Ash": 2, "Blast_Furnace_Slag": 3,
    "Water": 0.05, "Superplasticizer": 50,
    "Fine_Aggregate": 1.2, "Coarse_Aggregate": 1.5
}

RAW_FEATURES = ['Cement','Blast_Furnace_Slag','Fly_Ash','Water','Superplasticizer',
                'Coarse_Aggregate','Fine_Aggregate','Age']
ENGINEERED_FEATURES = ['Water_Binder_Ratio','Log_Age','Cement_x_Age','SCM_Ratio']
FINAL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
LOG_TRANSFORM_COLS = ['Blast_Furnace_Slag', 'Fly_Ash', 'Superplasticizer']