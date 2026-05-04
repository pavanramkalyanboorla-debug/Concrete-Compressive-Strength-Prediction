# src/optimization.py
# Optuna-based mix design search: single-objective & Pareto multi-objective.
import numpy as np
import pandas as pd
import optuna
from functools import lru_cache

from src.pipeline.predict_pipeline import PredictPipeline

# ------------------------------------------------------------
# Material costs (₹/kg, approximate)
# ------------------------------------------------------------
COST = {
    "Cement": 7.0,
    "Fly_Ash": 2.0,
    "Blast_Furnace_Slag": 3.0,
    "Water": 0.05,
    "Superplasticizer": 50.0,
    "Fine_Aggregate": 1.2,
    "Coarse_Aggregate": 1.5,
}

# ------------------------------------------------------------
# Engineering constraints
# ------------------------------------------------------------
CONFIG = {
    "max_wb": 0.50,
    "min_cement": 320,
    "max_cement": 450,
    "max_scm": 0.6,
    "min_paste_vol": 0.26,
    "max_paste_vol": 0.34,
}

SPECIFIC_GRAVITIES = {
    'Cement': 3.15, 'Fly_Ash': 2.2, 'Blast_Furnace_Slag': 2.9, 'Water': 1.0,
    'Fine_Aggregate': 2.65, 'Coarse_Aggregate': 2.70
}
AIR_CONTENT = 0.02

_pipeline = PredictPipeline()


def _physics(mix: dict) -> dict:
    """Apply absolute volume method; recalculate aggregate masses."""
    mix = mix.copy()
    binder = mix['Cement'] + mix['Blast_Furnace_Slag'] + mix['Fly_Ash']
    vol = sum(mix[m] / (SPECIFIC_GRAVITIES[m] * 1000)
              for m in ['Cement', 'Fly_Ash', 'Blast_Furnace_Slag', 'Water'])
    remaining = 1.0 - (vol + AIR_CONTENT)
    if remaining <= 0:
        return None
    fine_ratio = 0.4
    total_agg = remaining * SPECIFIC_GRAVITIES['Fine_Aggregate'] * 1000
    mix['Fine_Aggregate'] = total_agg * fine_ratio
    mix['Coarse_Aggregate'] = total_agg * (1 - fine_ratio)
    mix['Paste_Volume'] = vol + AIR_CONTENT
    return mix


def _constraints(mix: dict) -> list:
    """Return list of violations; empty list if all constraints pass."""
    v = []
    binder = mix['Cement'] + mix['Blast_Furnace_Slag'] + mix['Fly_Ash']
    wb = mix['Water'] / binder
    if wb > CONFIG["max_wb"]:
        v.append("w/b too high")
    if mix['Cement'] < CONFIG["min_cement"]:
        v.append("cement too low")
    if mix['Cement'] > CONFIG["max_cement"]:
        v.append("cement too high")
    scm = (mix['Fly_Ash'] + mix['Blast_Furnace_Slag']) / binder
    if scm > CONFIG["max_scm"]:
        v.append("SCM ratio too high")
    pv = mix.get('Paste_Volume', 0)
    if not (CONFIG["min_paste_vol"] <= pv <= CONFIG["max_paste_vol"]):
        v.append("paste volume out of range")
    return v


def _predict(mix: dict) -> float:
    return _pipeline.predict(mix)[0]


def _cost(mix: dict) -> float:
    return sum(mix[k] * COST[k] for k in COST)


# ------------------------------------------------------------
# Single-objective optimisation
# ------------------------------------------------------------
def single_objective_study(target_strength: float = 40, n_trials: int = 120):
    def objective(trial):
        mix = {
            "Cement": trial.suggest_float("Cement", 300, 450),
            "Blast_Furnace_Slag": trial.suggest_float("Slag", 0, 250),
            "Fly_Ash": trial.suggest_float("FlyAsh", 0, 150),
            "Water": trial.suggest_float("Water", 140, 220),
            "Superplasticizer": trial.suggest_float("SP", 0, 25),
            "Fine_Aggregate": 700,
            "Coarse_Aggregate": 1000,
            "Age": 28,
        }
        mix = _physics(mix)
        if mix is None:
            return 1e6
        if _constraints(mix):
            return 1e6

        strength = _predict(mix)
        cost = _cost(mix)
        return abs(strength - target_strength) + 0.001 * cost

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study


# ------------------------------------------------------------
# Multi-objective (Pareto) optimisation
# ------------------------------------------------------------
def multi_objective_study(target_strength: float = 40, n_trials: int = 80):
    def objectives(trial):
        mix = {
            "Cement": trial.suggest_float("Cement", 300, 450),
            "Blast_Furnace_Slag": trial.suggest_float("Slag", 0, 250),
            "Fly_Ash": trial.suggest_float("FlyAsh", 0, 150),
            "Water": trial.suggest_float("Water", 140, 220),
            "Superplasticizer": trial.suggest_float("SP", 0, 25),
            "Fine_Aggregate": 700,
            "Coarse_Aggregate": 1000,
            "Age": 28,
        }
        mix = _physics(mix)
        if mix is None:
            return 1e6, 1e6
        if _constraints(mix):
            return 1e6, 1e6

        strength = _predict(mix)
        cost = _cost(mix)
        return abs(strength - target_strength), cost

    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objectives, n_trials=n_trials)
    return study


def best_single_mix(study):
    p = study.best_params
    mix = {
        "Cement": p["Cement"],
        "Blast_Furnace_Slag": p["Slag"],
        "Fly_Ash": p["FlyAsh"],
        "Water": p["Water"],
        "Superplasticizer": p["SP"],
        "Fine_Aggregate": 700,
        "Coarse_Aggregate": 1000,
        "Age": 28,
    }
    return _physics(mix)