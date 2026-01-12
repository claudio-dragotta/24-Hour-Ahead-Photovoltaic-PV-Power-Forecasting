"""ARCHIVE: generate_mb_predictions.py

Original: scripts/ensemble/generate_mb_predictions.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from pv_forecasting.logger import get_logger

logger = get_logger(__name__)


def generate_predictions():
    logger.info("ARCHIVED: Generating Multi-Branch predictions (kept for reference)")
    model_dir = Path("outputs/multi_branch/final_v1")
    data_path = Path("outputs/processed.parquet")
    output_dir = Path("outputs/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    with open(model_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    df = pd.read_parquet(data_path)
    print(f"Archived Multi-Branch prediction script available; data rows: {len(df)}")


if __name__ == "__main__":
    generate_predictions()
