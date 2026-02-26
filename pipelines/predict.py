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

    def predict(self, features: pd.DataFrame) -> int:
        """
        Transform features and return a prediction.

        Args:
            features: DataFrame with the 12 input features.

        Returns:
            Prediction result (0 = income ≤ 50K, 1 = income > 50K).
        """
        try:
            self._load_artifacts()

            scaled = self._preprocessor.transform(features)
            prediction = self._model.predict(scaled)

            logger.info(f"Prediction result: {prediction[0]}")
            return int(prediction[0])

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
