from src.constants import *

def apply_physics(mix: dict) -> tuple:
    mix = mix.copy()
    binder = mix['Cement'] + mix['Blast_Furnace_Slag'] + mix['Fly_Ash']

    # Moisture correction
    water_correction = (
        (MOISTURE['Fine_Aggregate'] - ABSORPTION['Fine_Aggregate']) * mix['Fine_Aggregate'] +
        (MOISTURE['Coarse_Aggregate'] - ABSORPTION['Coarse_Aggregate']) * mix['Coarse_Aggregate']
    )
    mix['Water'] += water_correction

    vol = sum(mix[m] / (SPECIFIC_GRAVITIES[m] * 1000)
              for m in ['Cement','Fly_Ash','Blast_Furnace_Slag','Water'])
    remaining = 1.0 - (vol + AIR_CONTENT)
    if remaining <= 0:
        return None, ["Volume overflow"]

    fine_ratio = CONFIG["fine_aggregate_ratio"]
    total_agg_mass = remaining * SPECIFIC_GRAVITIES['Fine_Aggregate'] * 1000
    mix['Fine_Aggregate'] = total_agg_mass * fine_ratio
    mix['Coarse_Aggregate'] = total_agg_mass * (1 - fine_ratio)
    mix['Paste_Volume'] = vol + AIR_CONTENT
    return mix, []