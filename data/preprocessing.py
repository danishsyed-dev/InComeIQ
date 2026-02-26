"""
Data Preprocessing — Outlier capping, imputation, and feature scaling.

Pipeline stages:
  1. IQR-based outlier capping (clips extreme values to bounds)
  2. Missing value imputation (median strategy)
  3. Feature scaling (StandardScaler)

The fitted preprocessor is saved as a pickle artifact for reuse
during prediction.
"""

import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import PREPROCESSOR_PATH, TARGET_COLUMN
from config.feature_config import NUMERICAL_FEATURES
from core.exceptions import CustomException
from core.logging import get_logger
from core.utils import save_object

logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Path where the fitted preprocessor is saved."""
    preprocessor_path: str = str(PREPROCESSOR_PATH)


class DataPreprocessor:
    """Handles outlier removal, imputation, and scaling."""

    def __init__(self):
        self.config = PreprocessingConfig()

    def _build_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Build the sklearn preprocessing pipeline.

        Creates a Pipeline with:
          - SimpleImputer (fills NaN with median of each column)
          - StandardScaler (normalizes to mean=0, std=1)

        Wraps it in a ColumnTransformer applied to all numerical features.
        """
        logger.info("Building preprocessing pipeline")

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, NUMERICAL_FEATURES),
            ]
        )

        return preprocessor

    def _cap_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Cap outliers using the IQR (Interquartile Range) method.

        Values above Q3 + 1.5*IQR are clipped to the upper limit.
        Values below Q1 - 1.5*IQR are clipped to the lower limit.
        This PRESERVES all rows (no data is deleted).

        Args:
            df: DataFrame to modify.
            column: Column name to cap outliers in.

        Returns:
            DataFrame with outliers capped in the specified column.
        """
        try:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1

            upper_limit = q3 + 1.5 * iqr
            lower_limit = q1 - 1.5 * iqr

            # Cast column to float to avoid LossySetitemError when inserting float limits
            df[column] = df[column].astype(float)

            df.loc[df[column] > upper_limit, column] = upper_limit
            df.loc[df[column] < lower_limit, column] = lower_limit

            return df

        except Exception as e:
            logger.error(f"Error capping outliers in column: {column}")
            raise CustomException(e, sys)

    def initiate(self, train_path: str, test_path: str) -> tuple:
        """
        Run the full preprocessing pipeline.

        Steps:
          1. Read train/test CSVs
          2. Cap outliers using IQR on both sets
          3. Split features and target
          4. Fit preprocessor on train, transform both
          5. Save preprocessor artifact

        Args:
            train_path: Path to the training CSV file.
            test_path: Path to the test CSV file.

        Returns:
            Tuple of (train_array, test_array, preprocessor_path).
        """
        try:
            logger.info("Data preprocessing started")

            # 1. Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 2. Cap outliers on train and test data
            for col in NUMERICAL_FEATURES:
                self._cap_outliers_iqr(train_df, col)
            logger.info("Outliers capped on training data")

            for col in NUMERICAL_FEATURES:
                self._cap_outliers_iqr(test_df, col)
            logger.info("Outliers capped on test data")

            # 3. Split into features (X) and target (y)
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # 4. Build and fit preprocessor
            preprocessor = self._build_preprocessing_pipeline()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # 5. Combine features and target back into arrays
            train_array = np.c_[X_train_transformed, np.array(y_train)]
            test_array = np.c_[X_test_transformed, np.array(y_test)]

            # 6. Save the fitted preprocessor
            save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessor,
            )
            logger.info("Preprocessor saved successfully")

            return (train_array, test_array, self.config.preprocessor_path)

        except Exception as e:
            logger.error("Error in preprocessing stage")
            raise CustomException(e, sys)
