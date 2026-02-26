"""
Utility functions — model serialization and model evaluation.

Key fixes from the original project:
  - evaluate_model() now correctly iterates ALL models (return moved outside loop)
  - Uses joblib instead of pickle for sklearn model serialization
"""

import os
import sys
from typing import Any, Dict

import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from config.settings import CV_FOLDS
from core.exceptions import CustomException
from core.logging import get_logger

logger = get_logger(__name__)


def save_object(file_path: str, obj: Any) -> None:
    """
    Serialize and save a Python object to disk using joblib.

    Args:
        file_path: Destination path for the serialized object.
        obj: The Python object to serialize (e.g., sklearn model or pipeline).
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)
        logger.info(f"Object saved to {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Load a serialized Python object from disk.

    Args:
        file_path: Path to the serialized object file.

    Returns:
        The deserialized Python object.
    """
    try:
        obj = joblib.load(file_path)
        logger.info(f"Object loaded from {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(
    X_train, y_train,
    X_test, y_test,
    models: Dict[str, Any],
    params: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Train and evaluate multiple models using GridSearchCV.

    For each model, performs hyperparameter tuning via cross-validation,
    refits with the best parameters, and records the test accuracy.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        models: Dict mapping model name → sklearn estimator instance.
        params: Dict mapping model name → hyperparameter grid.

    Returns:
        Dict mapping model name → test accuracy score.

    NOTE: The original project had a critical bug where `return` was inside
    the for loop, causing only the first model to be evaluated. This version
    correctly returns AFTER the loop completes.
    """
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params[model_name]

            logger.info(f"Training {model_name} with GridSearchCV...")

            gs = GridSearchCV(model, param_grid, cv=CV_FOLDS, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Set best params and refit
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            report[model_name] = accuracy
            logger.info(f"{model_name} — Accuracy: {accuracy:.4f}")

        # ✅ FIX: return is now OUTSIDE the loop — all models get evaluated
        return report

    except Exception as e:
        raise CustomException(e, sys)
