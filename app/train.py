from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import json
import os
import shutil
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_save_model():
    try:
        # Load iris dataset
        logger.info("Loading iris dataset...")
        iris = load_iris()
        X, y = iris.data, iris.target

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Create and train the model
        logger.info("Training RandomForestClassifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Create model version directory
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(__file__).parent
        model_dir = base_dir / "models" / version
        latest_dir = base_dir / "models" / "latest"
        
        # Create directories
        model_dir.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)

        # Save model artifacts
        logger.info(f"Saving model artifacts to {model_dir}...")
        joblib.dump(model, model_dir / "model.pkl")
        joblib.dump(list(iris.feature_names), model_dir / "feature_names.pkl")

        # Save evaluation metrics
        metrics = {
            "accuracy": report["accuracy"],
            "macro_avg": report["macro avg"],
            "weighted_avg": report["weighted avg"],
            "confusion_matrix": conf_matrix
        }

        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Save metadata
        metadata = {
            "version": version,
            "training_date": datetime.now().isoformat(),
            "model_type": "RandomForestClassifier",
            "parameters": model.get_params(),
            "feature_names": list(iris.feature_names),
            "target_names": list(iris.target_names)
        }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Save a sample input for testing
        sample_input = X_test[0].reshape(1, -1)
        joblib.dump(sample_input, model_dir / "sample_input.pkl")

        # Print results
        logger.info(f"\nModel evaluation results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro average F1-score: {metrics['macro_avg']['f1-score']:.4f}")
        logger.info(f"Weighted average F1-score: {metrics['weighted_avg']['f1-score']:.4f}")

        # Copy files to latest directory
        for file in ["model.pkl", "feature_names.pkl", "metrics.json", "metadata.json", "sample_input.pkl"]:
            shutil.copy2(model_dir / file, latest_dir / file)
        
        logger.info(f"Successfully copied model files to latest directory")
        return True

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model() 