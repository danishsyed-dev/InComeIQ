# InComeIQ

A complete end-to-end Machine Learning web application designed to predict whether an individual's income exceeds $50K/yr based on census data.

Built strictly with Python, Flask, XGBoost, Scikit-Learn, and Vanilla CSS, utilizing a modern, maintainable modular architecture.

## 🌟 Key Features

- **Champion Model**: Uses XGBoost (84% accuracy) as the default predictor, automatically trained and selected via hyperparameter grid search against Random Forest, Decision Tree, Logistic Regression, and SVM.
- **Dynamic Feature Importance**: Every prediction generates a dynamic bar chart explaining *why* the model made its decision, utilizing the model's feature importance extraction.
- **Prediction Confidence**: Extracts and displays the exact probability metric calculated by the prediction engine.
- **REST API**: Fully functional endpoints for programmatic access and automated predictions.
- **Prediction History Logs**: Persistent SQLite tracking of all predictions, forming an auditable history trace.
- **Modern UI**: Fully redesigned glassmorphism interface focusing on a premium aesthetic experience.

## 🏗️ Architecture

The project drops standard Jupyter notebook spaghetti code for a robust, object-oriented pipeline design:
- **`config/`**: Centralized path management (`settings.py`) and feature schema mapping (`feature_config.py`).
- **`core/`**: Custom exception routing, persistent rotating file logging, and IO utilities.
- **`data/`**: Modularized ingestion and preprocessing components. Cures outliers with IQR capping and builds the transformation `Pipeline`.
- **`models/`**: Configurable hyperparameter tuning and model selection using cross-validated `GridSearchCV`.
- **`pipelines/`**: Standalone orchestrators bridging the raw features into the model (`train.py` & `predict.py`).
- **`web/`**: Implementation of the Flask App Factory pattern, routing, HTML templates, data validators, and SQLite models.

## 🚀 Quick Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/danishsyed-dev/InComeIQ.git
cd "InComeIQ"

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data & Train

The original dataset is deliberately `.gitignore`d. Run the downloader and training pipeline:

```bash
# Download Adult Census dataset from UCI repository
python download_data.py

# Run the training pipeline (Ingest -> Preprocess -> Train)
python -m pipelines.train
```

*Note: The training pipeline runs GridSearchCV across 5 heavy algorithms. It may take 5-10 minutes depending on your hardware.*

### 3. Run Web Application

```bash
# Start the server (runs on 0.0.0.0:5000)
python run.py
```

Open a browser and navigate to `http://localhost:5000`.

## 🔌 REST API Usage

You can interface programmatically with the predictor using JSON:

**Endpoint:** `POST /api/predict`

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "workclass": 3,
    "education_num": 13,
    "marital_status": 2,
    "occupation": 9,
    "relationship": 0,
    "race": 4,
    "sex": 1,
    "capital_gain": 5000,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": 38
  }'
```

**Response:**
```json
{
  "status": "success",
  "prediction": 1,
  "prediction_label": ">50K",
  "confidence": 0.852,
  "history_id": 1
}
```

**Endpoint:** `GET /api/history?limit=10`
Retrieve the last N predictions made across the app or API.

## 🐳 Docker Deployment

The repository includes a production-ready `Dockerfile` powered by `gunicorn`.

```bash
# Build the image
docker build -t incomeiq .

# Run the container
docker run -p 5000:5000 incomeiq
```
