"""
Training Pipeline — Orchestrates the full model training workflow.

Chains together: Data Ingestion → Preprocessing → Model Training.
Can be run directly: python -m pipelines.train
"""

import sys

from data.ingestion import DataIngestion
from data.preprocessing import DataPreprocessor
from models.trainer import ModelTrainer
from core.exceptions import CustomException
from core.logging import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """Orchestrates the end-to-end training workflow."""

    def __init__(self):
        self.ingestion = DataIngestion()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()

    def run(self) -> dict:
        """
        Execute the full training pipeline.

        Returns:
            Model comparison report (dict of model_name → accuracy).
        """
        try:
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            # Stage 1: Data Ingestion
            logger.info("Stage 1/3: Data Ingestion")
            train_path, test_path = self.ingestion.initiate()

            # Stage 2: Data Preprocessing
            logger.info("Stage 2/3: Data Preprocessing")
            train_arr, test_arr, _ = self.preprocessor.initiate(
                train_path, test_path
            )

            # Stage 3: Model Training
            logger.info("Stage 3/3: Model Training")
            report = self.trainer.initiate(train_arr, test_arr)

            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            return report

        except Exception as e:
            logger.error("Training pipeline failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
