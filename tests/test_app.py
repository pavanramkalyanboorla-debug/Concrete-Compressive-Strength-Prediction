from fastapi.testclient import TestClient
import pytest

import app.main as main

client = TestClient(main.app)


def test_get_strength_category_boundaries():
    assert main.get_strength_category(10.0) == "Low Strength (< 20 MPa)"
    assert main.get_strength_category(20.0) == "Normal Strength (20–40 MPa)"
    assert main.get_strength_category(39.999) == "Normal Strength (20–40 MPa)"
    assert main.get_strength_category(40.0) == "High Strength (40–60 MPa)"
    assert main.get_strength_category(60.0) == "Ultra High Strength (> 60 MPa)"


class DummyPreprocessor:
    def transform(self, df):
        return df.values


class DummyModel:
    def predict(self, data):
        # Return predictable value per row
        return [50.0] * len(data)


@pytest.fixture(autouse=True)
def monkeypatch_model_preprocessor(monkeypatch):
    monkeypatch.setattr(main, "load_error", None)
    monkeypatch.setattr(main, "model", DummyModel())
    monkeypatch.setattr(main, "preprocessor", DummyPreprocessor())


def test_predict_strength_valid_payload():
    payload = {
        "Cement": 300.0,
        "Blast_Furnace_Slag": 0.0,
        "Fly_Ash": 0.0,
        "Water": 150.0,
        "Superplasticizer": 0.0,
        "Coarse_Aggregate": 1000.0,
        "Fine_Aggregate": 800.0,
        "Age": 28,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_strength_mpa"] == 50.0
    assert data["strength_category"] == "High Strength (40–60 MPa)"


def test_predict_strength_invalid_payload():
    payload = {
        "Cement": -10.0,
        "Blast_Furnace_Slag": 0.0,
        "Fly_Ash": 0.0,
        "Water": 150.0,
        "Superplasticizer": 0.0,
        "Coarse_Aggregate": 1000.0,
        "Fine_Aggregate": 800.0,
        "Age": 28,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_success():
    payload = [
        {
            "Cement": 300.0,
            "Blast_Furnace_Slag": 0.0,
            "Fly_Ash": 0.0,
            "Water": 150.0,
            "Superplasticizer": 0.0,
            "Coarse_Aggregate": 1000.0,
            "Fine_Aggregate": 800.0,
            "Age": 28,
        },
        {
            "Cement": 350.0,
            "Blast_Furnace_Slag": 20.0,
            "Fly_Ash": 10.0,
            "Water": 160.0,
            "Superplasticizer": 3.0,
            "Coarse_Aggregate": 1000.0,
            "Fine_Aggregate": 700.0,
            "Age": 56,
        },
    ]

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["predicted_strength_mpa"] == 50.0
