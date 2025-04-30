import pytest
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from main import app
import joblib
import numpy as np

# Create test client
client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Ensure we have model artifacts for testing"""
    models_dir = Path(__file__).parent.parent / "models"
    if not models_dir.exists():
        models_dir.mkdir(exist_ok=True)
        
    latest_dir = models_dir / "latest"
    if not latest_dir.exists():
        # Train a new model if none exists
        import train
        
    return True

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "model_version" in data
    assert "model_type" in data

def test_get_features():
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert "feature_names" in data
    assert len(data["feature_names"]) == 4

def test_get_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "training_date" in data
    assert "model_type" in data
    assert "metrics" in data

def test_predict_valid_input():
    # Load sample input from the model directory
    model_dir = Path(__file__).parent.parent / "models/latest"
    sample_input = joblib.load(model_dir / "sample_input.pkl")
    
    response = client.post(
        "/predict",
        json={"features": sample_input[0].tolist()}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "model_version" in data
    assert isinstance(data["probability"], float)
    assert 0 <= data["probability"] <= 1

def test_predict_invalid_input():
    # Test with wrong number of features
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]}  # Only 3 features
    )
    assert response.status_code == 400

    # Test with invalid feature values
    response = client.post(
        "/predict",
        json={"features": ["invalid", 2.0, 3.0, 4.0]}  # Non-numeric value
    )
    assert response.status_code == 500 