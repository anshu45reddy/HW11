from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import json
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
model_dir = f"models/{version}"
os.makedirs(model_dir, exist_ok=True)

# Save model artifacts
logger.info(f"Saving model artifacts to {model_dir}...")
joblib.dump(model, f"{model_dir}/model.pkl")
joblib.dump(iris.feature_names, f"{model_dir}/feature_names.pkl")

# Save evaluation metrics
metrics = {
    "accuracy": report["accuracy"],
    "macro_avg": report["macro avg"],
    "weighted_avg": report["weighted avg"],
    "confusion_matrix": conf_matrix
}

with open(f"{model_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save metadata
metadata = {
    "version": version,
    "training_date": datetime.now().isoformat(),
    "model_type": "RandomForestClassifier",
    "parameters": model.get_params(),
    "feature_names": iris.feature_names,
    "target_names": iris.target_names.tolist()
}

with open(f"{model_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# Save a sample input for testing
sample_input = X_test[0].reshape(1, -1)
joblib.dump(sample_input, f"{model_dir}/sample_input.pkl")

# Print results
logger.info(f"\nModel evaluation results:")
logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
logger.info(f"Macro average F1-score: {metrics['macro_avg']['f1-score']:.4f}")
logger.info(f"Weighted average F1-score: {metrics['weighted_avg']['f1-score']:.4f}")

# Create symlink to latest model
latest_dir = "models/latest"
if os.path.exists(latest_dir):
    os.remove(latest_dir)
os.symlink(version, latest_dir, target_is_directory=True)
logger.info(f"Created symlink to latest model version: {version}") 