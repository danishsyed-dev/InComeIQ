"""
Model Trainer — Trains multiple classifiers and selects the best one.

Compares Random Forest, Decision Tree, and Logistic Regression
using GridSearchCV hyperparameter tuning. The best model (by accuracy)
is saved to disk.
"""

import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config.settings import MODEL_PATH, PARAM_GRIDS
from core.exceptions import CustomException
from core.logging import get_logger
from core.utils import evaluate_model, save_object

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    """Path where the trained model is saved."""
    model_path: str = str(MODEL_PATH)


class ModelTrainer:
    """Trains, evaluates, and saves the best classification model."""

    def __init__(self):
        self.config = TrainerConfig()

    def initiate(self, train_array, test_array) -> dict:
        """
        Train multiple models and select the best one.

        Args:
            train_array: Numpy array with features + target (last column).
            test_array: Numpy array with features + target (last column).

        Returns:
            Dict with model comparison report (model_name → accuracy).
        """
        try:
            logger.info("Model training started")

            # Split arrays into features and target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Define models to compare
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Machine": SVC(),
                "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            }

            # Evaluate all models with GridSearchCV
            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=PARAM_GRIDS,
            )

            # Find the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logger.info(
                f"Best Model: {best_model_name} | "
                f"Accuracy: {best_model_score:.4f}"
            )

            print(f"\n{'='*60}")
            print(f"  Best Model: {best_model_name}")
            print(f"  Accuracy:   {best_model_score:.4f}")
            print(f"{'='*60}\n")

            # Print all model scores for comparison
            print("Model Comparison Report:")
            print("-" * 40)
            for name, score in sorted(
                model_report.items(), key=lambda x: x[1], reverse=True
            ):
                marker = " ← BEST" if name == best_model_name else ""
                print(f"  {name:25s} {score:.4f}{marker}")
            print("-" * 40)

            # Save the best model
            save_object(
                file_path=self.config.model_path,
                obj=best_model,
            )
            logger.info("Best model saved successfully")

            return model_report

        except Exception as e:
            logger.error("Error in model training stage")
            raise CustomException(e, sys)
