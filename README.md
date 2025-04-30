# Iris Classifier MLOps Project

A machine learning project that demonstrates MLOps best practices using an Iris flower classifier. The project includes a FastAPI-based REST API, automated model training, and a robust CI/CD pipeline.

## Features

- FastAPI-based REST API for model predictions
- Automated model training and versioning
- Comprehensive test suite with pytest
- GitHub Actions CI/CD pipeline
- Model performance monitoring
- Error handling and input validation

## Project Structure

```
mlops/
├── app/
│   ├── main.py           # FastAPI application
│   ├── train.py          # Model training script
│   ├── models/           # Model artifacts directory
│   │   └── latest/       # Latest model version
│   └── tests/            # Test files
├── .github/
│   └── workflows/        # GitHub Actions workflows
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/anshu45reddy/HW11.git
cd HW11/mlops
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
cd app
python train.py
```

2. Start the API server:
```bash
cd app
uvicorn main:app --reload
```

3. Access the API documentation at:
```
http://localhost:8000/docs
```

## API Endpoints

- `GET /`: Welcome message and model information
- `GET /health`: Health check endpoint
- `GET /model-info`: Detailed model information and metrics
- `GET /features`: List of model features
- `POST /predict`: Make predictions with the model

### Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Testing

Run the test suite:
```bash
cd app
python -m pytest tests/ -v
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:
1. Runs linting and security checks
2. Trains and validates the model
3. Runs the test suite
4. Tests the API functionality
5. Generates test coverage reports

## License

MIT License

## Author

Chandana Rondla 