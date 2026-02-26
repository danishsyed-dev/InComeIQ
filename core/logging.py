"""
Logging configuration with rotating file handler.

Creates timestamped log files in the logs/ directory.
Uses RotatingFileHandler to prevent unbounded log growth.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

from config.settings import LOG_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Configured logger with file and console handlers.
    """
    # Ensure logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Log file with timestamp
    log_filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    log_filepath = os.path.join(LOG_DIR, log_filename)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if logger already configured
    if not logger.handlers:
        # File handler with rotation (max 5MB per file, keep 3 backups)
        file_handler = RotatingFileHandler(
            log_filepath,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
        )
        file_handler.setLevel(logging.INFO)

        # Console handler for real-time feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
