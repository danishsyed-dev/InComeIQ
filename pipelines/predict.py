"""
Prediction Pipeline — Loads saved artifacts and makes predictions.

Features:
  - One-time model loading (cached after first prediction)
  - CustomInput class for building DataFrame from form data
"""

import sys
import pandas as pd
from typing import Optional

from config.settings import PREPROCESSOR_PATH, MODEL_PATH
from config.feature_config import NUMERICAL_FEATURES
from core.exceptions import CustomException
from core.logging import get_logger
from core.utils import load_object

import matplotlib
matplotlib.use('Agg') # Needed to prevent GUI errors on server
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

logger = get_logger(__name__)


class PredictionPipeline:
    """
    Loads the saved preprocessor and model, then predicts new inputs.

    The model and preprocessor are loaded once on first call and
    cached in memory for subsequent predictions.
    """

    def __init__(self):
        self._preprocessor = None
        self._model = None

    def _load_artifacts(self) -> None:
        """Load the preprocessor and model from disk (one-time)."""
        if self._preprocessor is None:
            logger.info("Loading preprocessor from disk")
            self._preprocessor = load_object(str(PREPROCESSOR_PATH))

        if self._model is None:
            logger.info("Loading model from disk")
            self._model = load_object(str(MODEL_PATH))

    def predict(self, features: pd.DataFrame) -> tuple[int, Optional[float], Optional[str]]:
        """
        Transform features and return a prediction along with probability and plot path.

        Args:
            features: DataFrame with the 12 input features.

        Returns:
            Tuple of:
                - Prediction result (0 = income ≤ 50K, 1 = income > 50K).
                - Probability of the predicted class (0.0 to 1.0), or None.
                - Filename of the generated feature importance plot, or None.
        """
        try:
            self._load_artifacts()

            scaled = self._preprocessor.transform(features)
            
            # Get hard prediction
            prediction = self._model.predict(scaled)
            pred_class = int(prediction[0])

            # Get probability if model supports it
            probability = None
            if hasattr(self._model, "predict_proba"):
                try:
                    proba_array = self._model.predict_proba(scaled)
                    
                    # predict_proba returns array of shape (n_samples, n_classes)
                    # We want the probability of the *predicted* class
                    class_idx = list(self._model.classes_).index(pred_class)
                    probability = float(proba_array[0][class_idx])
                except Exception as e:
                    logger.warning(f"Could not extract probability: {e}")

            # Generate feature importance plot if model supports it
            feature_plot_path = None
            if hasattr(self._model, "feature_importances_"):
                try:
                    importances = self._model.feature_importances_
                    feature_names = features.columns

                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    sorted_features = [feature_names[i] for i in indices]
                    sorted_importances = importances[indices]

                    # Create static dir if not exists
                    static_dir = os.path.join("web", "static")
                    os.makedirs(static_dir, exist_ok=True)
                    plot_filename = "feature_importance.png"
                    plot_path = os.path.join(static_dir, plot_filename)

                    # Plot
                    plt.figure(figsize=(8, 6), facecolor='#1e1e3f')
                    ax = plt.axes()
                    ax.set_facecolor('#1e1e3f')
                    
                    sns.barplot(
                        x=sorted_importances[:8], # Top 8
                        y=sorted_features[:8],
                        palette="viridis"
                    )
                    
                    plt.title("Top Predictors", color='white', pad=20)
                    plt.xlabel("Relative Importance", color='lightgray')
                    plt.ylabel("")
                    plt.xticks(color='lightgray')
                    plt.yticks(color='white')
                    
                    # Remove borders
                    sns.despine(left=True, bottom=True)
                    
                    plt.tight_layout()
                    plt.savefig(plot_path, bg='transparent', dpi=100)
                    plt.close()
                    
                    feature_plot_path = plot_filename
                except Exception as e:
                    logger.warning(f"Could not generate feature importance plot: {e}")

            logger.info(f"Prediction result: {pred_class}, Probability: {probability}")
            return pred_class, probability, feature_plot_path

        except Exception as e:
            logger.error("Prediction failed")
            raise CustomException(e, sys)


class CustomInput:
    """
    Represents a single user input for income prediction.

    Takes the 12 feature values from the web form and converts
    them into a pandas DataFrame that the model can consume.
    """

    def __init__(
        self,
        age: int,
        workclass: int,
        education_num: int,
        marital_status: int,
        occupation: int,
        relationship: int,
        race: int,
        sex: int,
        capital_gain: int,
        capital_loss: int,
        hours_per_week: int,
        native_country: int,
    ):
        self.age = age
        self.workclass = workclass
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the input values into a single-row DataFrame.

        Returns:
            DataFrame with columns matching the training features.
        """
        try:
            data = {
                "age": [self.age],
                "workclass": [self.workclass],
                "education_num": [self.education_num],
                "marital_status": [self.marital_status],
                "occupation": [self.occupation],
                "relationship": [self.relationship],
                "race": [self.race],
                "sex": [self.sex],
                "capital_gain": [self.capital_gain],
                "capital_loss": [self.capital_loss],
                "hours_per_week": [self.hours_per_week],
                "native_country": [self.native_country],
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
