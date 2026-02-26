"""
Data Ingestion — Load the dataset and split into train/test sets.

Reads the raw CSV, validates expected columns exist,
saves artifacts (raw, train, test CSVs), and returns paths.
"""

import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from config.settings import (
    RAW_DATA_FILE,
    RAW_ARTIFACT_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from config.feature_config import NUMERICAL_FEATURES
from core.exceptions import CustomException
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionConfig:
    """Paths where ingestion artifacts are saved."""
    raw_data_source: str = str(RAW_DATA_FILE)
    raw_artifact_path: str = str(RAW_ARTIFACT_PATH)
    train_path: str = str(TRAIN_DATA_PATH)
    test_path: str = str(TEST_DATA_PATH)


class DataIngestion:
    """Handles loading raw data and splitting into train/test sets."""

    def __init__(self):
        self.config = IngestionConfig()

    def initiate(self) -> tuple:
        """
        Load data, validate columns, split into train/test, save artifacts.

        Returns:
            Tuple of (train_path, test_path) for downstream components.
        """
        logger.info("Data ingestion started")

        try:
            # 1. Load raw data
            logger.info(f"Reading data from {self.config.raw_data_source}")
            data = pd.read_csv(self.config.raw_data_source)
            logger.info(f"Dataset shape: {data.shape}")

            # 2. Validate expected columns exist
            expected_columns = NUMERICAL_FEATURES + [TARGET_COLUMN]
            missing = set(expected_columns) - set(data.columns)
            if missing:
                raise ValueError(f"Missing columns in dataset: {missing}")

            logger.info("Column validation passed")

            # 3. Save raw data artifact
            RAW_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(self.config.raw_artifact_path, index=False)

            # 4. Train-test split
            train_set, test_set = train_test_split(
                data,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )
            logger.info(
                f"Train size: {len(train_set)}, Test size: {len(test_set)}"
            )

            # 5. Save train/test artifacts
            train_set.to_csv(self.config.train_path, index=False, header=True)
            test_set.to_csv(self.config.test_path, index=False, header=True)

            logger.info("Data ingestion completed successfully")

            return (self.config.train_path, self.config.test_path)

        except Exception as e:
            logger.error("Error in data ingestion stage")
            raise CustomException(e, sys)
