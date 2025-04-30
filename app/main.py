from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Dict, Any
import json
import os
from pathlib import Path

# Create FastAPI app
app = FastAPI(
    title="Iris Classifier API",
    description="A simple API for iris flower classification with model versioning",
    version="1.0.0"
)

# Load the latest model and feature names
MODEL_DIR = Path("models/latest")
if not MODEL_DIR.exists():
    raise RuntimeError("No trained model found. Please run train.py first.")

model = joblib.load(MODEL_DIR / "model.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
metadata = json.load(open(MODEL_DIR / "metadata.json"))
metrics = json.load(open(MODEL_DIR / "metrics.json"))

# Input data model
class IrisFeatures(BaseModel):
    features: List[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

# Output data model
class IrisPrediction(BaseModel):
    prediction: str
    probability: float
    model_version: str

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Iris Classifier API",
        "model_version": metadata["version"],
        "model_type": metadata["model_type"]
    }

@app.get("/model-info")
def get_model_info():
    """Get detailed information about the current model"""
    return {
        "version": metadata["version"],
        "training_date": metadata["training_date"],
        "model_type": metadata["model_type"],
        "parameters": metadata["parameters"],
        "feature_names": metadata["feature_names"],
        "target_names": metadata["target_names"],
        "metrics": metrics
    }

@app.get("/features")
def get_features():
    return {"feature_names": feature_names}

@app.post("/predict", response_model=IrisPrediction)
def predict(data: IrisFeatures):
    try:
        # Validate input length
        if len(data.features) != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_names)} features, got {len(data.features)}"
            )

        # Make prediction
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = max(model.predict_proba(features)[0])

        return {
            "prediction": metadata["target_names"][prediction],
            "probability": float(probability),
            "model_version": metadata["version"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 