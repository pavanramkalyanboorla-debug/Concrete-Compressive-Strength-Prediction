import numpy as np
from src.constants import CONFIG

def estimate_slump(mix: dict) -> float:
    water = mix['Water']
    sp = mix['Superplasticizer']
    paste = mix['Paste_Volume']
    slump = (water - 120) * 2 + sp * 4 - 200 * paste + 200
    return np.clip(slump, 0, 250)

def enforce_constraints(mix: dict) -> tuple:
    violations = []
    binder = mix['Cement'] + mix['Blast_Furnace_Slag'] + mix['Fly_Ash']
    wb = mix['Water'] / binder

    if wb > CONFIG["max_wb"]:
        return None, ["w/b too high"]
    if mix['Cement'] < CONFIG["min_cement"]:
        return None, ["cement too low"]
    if mix['Cement'] > CONFIG["max_cement"]:
        return None, ["cement too high"]
    scm = (mix['Fly_Ash'] + mix['Blast_Furnace_Slag']) / binder
    if scm > CONFIG["max_scm"]:
        return None, ["SCM too high"]
    pv = mix.get('Paste_Volume', 0)
    if not (CONFIG["min_paste"] <= pv <= CONFIG["max_paste"]):
        return None, ["paste out of range"]
    slump = estimate_slump(mix)
    lo, hi = CONFIG["target_slump_mm"]
    if not (lo <= slump <= hi):
        return None, [f"slump {slump:.0f} mm out of range"]
    return mix, []