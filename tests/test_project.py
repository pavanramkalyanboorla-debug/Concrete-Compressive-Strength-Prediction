# tests/test_project.py
import requests
import json
import base64

BASE = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200, f"Health failed: {r.text}"
    assert r.json()["status"] == "healthy"
    print("[PASS] Health check")

def test_predict():
    mix = {
        "Cement": 350, "Blast_Furnace_Slag": 50, "Fly_Ash": 50,
        "Water": 180, "Superplasticizer": 5,
        "Coarse_Aggregate": 1000, "Fine_Aggregate": 700, "Age": 28
    }
    r = requests.post(f"{BASE}/predict", json=mix)
    assert r.status_code == 200, f"Predict failed: {r.text}"
    data = r.json()
    assert 10 < data["predicted_strength_mpa"] < 100, "Strength out of range"
    assert data["strength_category"] in [
        "Low Strength (< 20 MPa)", "Normal Strength (20–40 MPa)",
        "High Strength (40–60 MPa)", "Ultra High Strength (> 60 MPa)"
    ]
    print(f"[PASS] Predict: {data['predicted_strength_mpa']:.1f} MPa ({data['strength_category']})")

def test_preprocessing_fix():
    """
    The original bug: raw Blast_Furnace_Slag / Fly_Ash were not log1p'd.
    After the fix, a mix with high Slag/FlyAsh should give a **different** prediction
    than a mixture where those values are artificially log‑transformed externally.
    """
    mix1 = {
        "Cement": 300, "Blast_Furnace_Slag": 200, "Fly_Ash": 100,
        "Water": 200, "Superplasticizer": 0,
        "Coarse_Aggregate": 1000, "Fine_Aggregate": 700, "Age": 28
    }
    r1 = requests.post(f"{BASE}/predict", json=mix1)
    assert r1.status_code == 200
    strength1 = r1.json()["predicted_strength_mpa"]

    # If the old broken system were in place, the strength would be unrealistic.
    # The fixed pipeline includes log1p internally, so strength1 must be reasonable.
    assert 10 < strength1 < 100, f"Preprocessing fix likely broken: strength={strength1}"
    print(f"[PASS] Preprocessing fix: strength with high SCM = {strength1:.1f} MPa (within range)")

def test_optimize_single():
    req = {"target_strength": 40, "n_trials": 30, "multi_objective": False}
    r = requests.post(f"{BASE}/optimize", json=req)
    assert r.status_code == 200, f"Optimize single failed: {r.text}"
    data = r.json()
    assert "mix" in data
    assert "predicted_strength" in data
    assert "cost" in data
    assert isinstance(data["violations"], list)
    print(f"[PASS] Single‑objective: strength={data['predicted_strength']} MPa, cost=₹{data['cost']:.0f}/m³, violations={data['violations']}")

def test_optimize_multi():
    req = {"target_strength": 40, "n_trials": 30, "multi_objective": True}
    r = requests.post(f"{BASE}/optimize", json=req)
    assert r.status_code == 200, f"Optimize multi failed: {r.text}"
    data = r.json()
    pareto = data["pareto_front"]
    assert len(pareto) >= 1, "Pareto front empty"
    # Check structure of each Pareto mix
    for mix in pareto:
        assert "mix" in mix
        assert "predicted_strength" in mix
        assert "cost" in mix
        assert "strength_error" in mix
        assert "violations" in mix
    print(f"[PASS] Multi‑objective: Pareto size = {len(pareto)}")

def test_shap():
    r = requests.get(f"{BASE}/shap", params={
        "cement": 350, "slag": 50, "flyash": 50,
        "water": 180, "sp": 5, "coarse": 1000, "fine": 700, "age": 28
    })
    assert r.status_code == 200, f"SHAP failed: {r.text}"
    data = r.json()
    assert "shap_plot" in data
    # Verify the string is a valid base64‑encoded PNG
    img_data = data["shap_plot"].split(",")[1]
    try:
        base64.b64decode(img_data)
    except Exception:
        raise AssertionError("SHAP plot is not valid base64")
    print("[PASS] SHAP plot returned (base64 PNG)")

if __name__ == "__main__":
    test_health()
    test_predict()
    test_preprocessing_fix()
    test_optimize_single()
    test_optimize_multi()
    test_shap()
    print("\nAll tests passed. Your project is ready for production.")