"""
Centralized configuration — all paths, constants, and hyperparameters.

Every file path in the project is defined here so nothing is hardcoded
elsewhere. Uses pathlib.Path for cross-platform compatibility.
"""

from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data Paths ────────────────────────────────────────────────────────────
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "adult.csv"

# ── Artifact Paths (generated at runtime) ─────────────────────────────────
ARTIFACTS_DIR = BASE_DIR / "artifacts"

INGESTION_DIR = ARTIFACTS_DIR / "data_ingestion"
TRAIN_DATA_PATH = INGESTION_DIR / "train.csv"
TEST_DATA_PATH = INGESTION_DIR / "test.csv"
RAW_ARTIFACT_PATH = INGESTION_DIR / "raw.csv"

PREPROCESSING_DIR = ARTIFACTS_DIR / "preprocessing"
PREPROCESSOR_PATH = PREPROCESSING_DIR / "preprocessor.pkl"

MODEL_DIR = ARTIFACTS_DIR / "model_trainer"
MODEL_PATH = MODEL_DIR / "model.pkl"

# ── Logging ───────────────────────────────────────────────────────────────
LOG_DIR = BASE_DIR / "logs"

# ── Training Config ──────────────────────────────────────────────────────
TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_FOLDS = 5

# ── Target Column ────────────────────────────────────────────────────────
TARGET_COLUMN = "income"

# ── Hyperparameter Grids ─────────────────────────────────────────────────
PARAM_GRIDS = {
    "Random Forest": {
        "class_weight": ["balanced"],
        "n_estimators": [20, 50, 100],
        "max_depth": [5, 8, 10],
        "min_samples_split": [2, 5, 10],
    },
    "Decision Tree": {
        "class_weight": ["balanced"],
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["best", "random"],
        "max_depth": [3, 4, 5, 6],
        "min_samples_split": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 3],
    },
    "Logistic Regression": {
        "class_weight": ["balanced"],
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "saga"],
    },
    "XGBoost Classifier": {
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 7],
    },
    "Support Vector Machine": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "probability": [True], # Important for predict_proba
    },
}
