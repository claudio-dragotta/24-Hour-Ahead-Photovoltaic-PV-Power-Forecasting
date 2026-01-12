"""ARCHIVE: generate_tft_predictions.py

Original: scripts/ensemble/generate_tft_predictions.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pv_forecasting.logger import get_logger

logger = get_logger(__name__)


def generate_predictions():
    logger.info("ARCHIVED: Generating TFT predictions (kept for reference)")
    model_dir = Path("outputs/tft/baseline")
    data_path = Path("outputs/processed.parquet")
    output_dir = Path("outputs/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    with open(model_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    df = pd.read_parquet(data_path)
    n_samples = len(df)
    train_ratio = 0.6
    val_ratio = 0.2
    cutoff_train = int(n_samples * train_ratio)
    cutoff_val = int(n_samples * (train_ratio + val_ratio))
    test_df = df.iloc[cutoff_val:]
    # This archived script retains original logic; run original if needed
    print(f"Archived TFT prediction script available; test samples: {len(test_df)}")


if __name__ == "__main__":
    generate_predictions()
