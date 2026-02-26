"""
Download the Adult Census Income dataset.

Downloads the dataset from the UCI Machine Learning Repository and
saves it to data/raw/adult.csv with clean column names.

Usage: python download_data.py
"""

import os
import pandas as pd
from config.settings import RAW_DATA_DIR, RAW_DATA_FILE


DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]


def download_and_prepare():
    """Download the Adult Census dataset and save as a clean CSV."""
    print("Downloading Adult Census Income dataset...")

    # Read from UCI repository
    df = pd.read_csv(
        DATASET_URL,
        names=COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        na_values="?",
    )

    print(f"Downloaded {len(df)} rows, {len(df.columns)} columns")

    # Encode the target column: <=50K → 0, >50K → 1
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    # Drop fnlwgt (census weighting — not useful for prediction)
    # Drop education (we keep education_num which is the numeric version)
    df = df.drop(columns=["fnlwgt", "education"])

    # Label-encode categorical columns to integers
    categorical_cols = [
        "workclass", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country",
    ]

    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    print(f"After processing: {df.shape}")
    print(f"Target distribution:\n{df['income'].value_counts()}")

    # Save to disk
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df.to_csv(RAW_DATA_FILE, index=False)
    print(f"\nDataset saved to: {RAW_DATA_FILE}")


if __name__ == "__main__":
    download_and_prepare()
