# Iris Classifier MLOps Project

A comprehensive MLOps project that demonstrates model training, versioning, API deployment, and CI/CD pipeline setup.

## Project Structure
```
mlops/
├── app/
│   ├── train.py        # Model training script
│   ├── main.py         # FastAPI application
│   ├── models/         # Model versions directory
│   │   ├── latest -> YYYYMMDD_HHMMSS/  # Symlink to latest model
│   │   └── YYYYMMDD_HHMMSS/  # Versioned model directory
│   │       ├── model.pkl
│   │       ├── feature_names.pkl
│   │       ├── sample_input.pkl
│   │       ├── metrics.json
│   │       └── metadata.json
│   └── tests/          # Test directory
│       └── test_api.py # API test cases
├── .github/
│   └── workflows/
│       └── ci.yml      # CI pipeline configuration
├── pyproject.toml      # Project configuration
└── requirements.txt    # Project dependencies
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mlops
   ```

2. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   cd app
   python train.py
   ```

4. Start the API:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

- `GET /`: Welcome message and model info
- `GET /model-info`: Detailed model information and metrics
- `GET /features`: List of feature names
- `POST /predict`: Make predictions
  - Input format:
    ```json
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    ```
  - Output format:
    ```json
    {
        "prediction": "setosa",
        "probability": 0.98,
        "model_version": "20240315_123456"
    }
    ```

## Model Versioning

The project implements a simple model versioning system:
- Each model training run creates a new versioned directory under `models/`
- The `latest` symlink always points to the most recent model
- Each model version includes:
  - Trained model
  - Feature names
  - Sample input
  - Evaluation metrics
  - Metadata (version, training date, parameters)

## Testing

1. Run API tests:
   ```bash
   cd app
   pytest tests/ -v
   ```

2. Run tests with coverage:
   ```bash
   pytest tests/ -v --cov=. --cov-report=html
   ```

## CI Pipeline

The CI pipeline includes:
1. Code linting with flake8
2. Security checks with bandit and safety
3. Model training and validation
4. API tests with coverage reporting
5. API startup test
6. Coverage report upload to Codecov

## GitHub Setup

1. Create a new repository on GitHub

2. Push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <repository-url>
   git push -u origin main
   ```

3. GitHub Actions will automatically run the CI pipeline on push and pull requests

4. Required GitHub Secrets:
   - None required for basic setup
   - Add `CODECOV_TOKEN` if using Codecov for coverage reporting

## API Documentation

Access the API documentation at `http://localhost:8000/docs` when running locally. 