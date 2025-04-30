from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
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

# Initialize model and metadata as None
model = None
feature_names = None
metadata = None
metrics = None

def load_model():
    """Load model and artifacts"""
    global model, feature_names, metadata, metrics
    
    # Get the directory containing this file
    current_dir = Path(__file__).parent
    MODEL_DIR = current_dir / "models/latest"
    
    if not MODEL_DIR.exists():
        # If no model exists, train one
        import train
        
    # Load the model and artifacts
    model = joblib.load(MODEL_DIR / "model.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    metadata = json.load(open(MODEL_DIR / "metadata.json"))
    metrics = json.load(open(MODEL_DIR / "metrics.json"))

# Load model on startup
load_model()

# Input data model
class IrisFeatures(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

    @property
    def feature_count(self):
        return len(self.features)

    def validate_features(self, feature_names):
        if self.feature_count != len(feature_names):
            raise ValidationError(
                f"Expected {len(feature_names)} features, got {self.feature_count}"
            )
        return True

# Output data model
class IrisPrediction(BaseModel):
    prediction: str
    probability: float
    version: str  # Renamed from model_version to avoid warning

    model_config = {
        'protected_namespaces': ()
    }

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
        data.validate_features(feature_names)

        # Make prediction
        try:
            features = np.array(data.features).reshape(1, -1)
            prediction = model.predict(features)[0]
            probability = max(model.predict_proba(features)[0])
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feature values: {str(e)}"
            )

        return {
            "prediction": metadata["target_names"][prediction],
            "probability": float(probability),
            "version": metadata["version"]  # Using renamed field
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the error for debugging
        print(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}") 