"""Generate TFT test predictions for ensemble (aligned with Multi-Branch).

This script loads the trained TFT model and generates predictions
on the same test set as Multi-Branch for proper ensemble combination.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pv_forecasting.logger import get_logger
from pv_forecasting.models.tft import TemporalFusionTransformer

logger = get_logger(__name__)


class TFTDataset(torch.utils.data.Dataset):
    """Dataset for TFT inference."""

    def __init__(self, df, static_features, temporal_past_features, temporal_future_features,
                 target_col, seq_len, horizon, static_scaler, temporal_past_scaler,
                 temporal_future_scaler, target_scaler):
        self.df = df.reset_index(drop=True)
        self.static_features = static_features
        self.temporal_past_features = temporal_past_features
        self.temporal_future_features = temporal_future_features
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        self.static_scaler = static_scaler
        self.temporal_past_scaler = temporal_past_scaler
        self.temporal_future_scaler = temporal_future_scaler
        self.target_scaler = target_scaler

        self.valid_indices = []
        for i in range(len(df) - seq_len - horizon + 1):
            if i + seq_len + horizon - 1 < len(df):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len
        target_start = end_idx
        target_end = target_start + self.horizon

        # Static features (same for all timesteps)
        static = self.df.loc[start_idx, self.static_features].values.astype(np.float32)

        # Temporal past
        temporal_past = self.df.loc[start_idx:end_idx-1, self.temporal_past_features].values.astype(np.float32)

        # Temporal future
        temporal_future = self.df.loc[target_start:target_end-1, self.temporal_future_features].values.astype(np.float32)

        # Targets
        targets = self.df.loc[target_start:target_end-1, self.target_col].values.astype(np.float32)

        # Apply scaling
        if self.static_scaler is not None:
            static = self.static_scaler.transform(static.reshape(1, -1)).flatten()
        if self.temporal_past_scaler is not None:
            temporal_past = self.temporal_past_scaler.transform(temporal_past)
        if self.temporal_future_scaler is not None:
            temporal_future = self.temporal_future_scaler.transform(temporal_future)
        if self.target_scaler is not None:
            targets = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()

        features = {
            'static': torch.from_numpy(static),
            'temporal_past': torch.from_numpy(temporal_past),
            'temporal_future': torch.from_numpy(temporal_future)
        }
        targets = torch.from_numpy(targets)

        return features, targets


def generate_predictions():
    """Generate TFT test predictions aligned with Multi-Branch."""
    logger.info("Generating TFT test predictions (aligned with Multi-Branch)")

    # Paths
    model_dir = Path("outputs/tft/baseline")
    data_path = Path("outputs/processed.parquet")
    output_dir = Path("outputs/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    # Load scalers
    with open(model_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Use SAME split as Multi-Branch (60/20/20)
    n_samples = len(df)
    train_ratio = 0.6
    val_ratio = 0.2
    cutoff_train = int(n_samples * train_ratio)
    cutoff_val = int(n_samples * (train_ratio + val_ratio))

    test_df = df.iloc[cutoff_val:]
    logger.info(f"Test set: {len(test_df)} samples (same split as Multi-Branch)")

    # Create dataset
    test_dataset = TFTDataset(
        test_df,
        config['static_features'],
        config['temporal_past_features'],
        config['temporal_future_features'],
        'pv',
        config['seq_len'],
        config['horizon'],
        scalers.get('static_scaler'),
        scalers.get('temporal_past_scaler'),
        scalers.get('temporal_future_scaler'),
        scalers['target_scaler']
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    logger.info(f"Test batches: {len(test_loader)}")

    # Load model
    checkpoint_path = model_dir / "tft-best.ckpt"
    logger.info(f"Loading model from {checkpoint_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # Generate predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            features, targets = batch
            features_device = {k: v.to(device) for k, v in features.items()}
            preds = model(features_device)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Denormalize
    logger.info("Denormalizing predictions")
    all_preds = scalers['target_scaler'].inverse_transform(all_preds)
    all_targets = scalers['target_scaler'].inverse_transform(all_targets)

    # Save in same format as Multi-Branch
    logger.info("Saving predictions in long format")
    predictions_list = []
    for i in range(len(all_preds)):
        for h in range(24):
            predictions_list.append({
                'sample_idx': i,
                'horizon': h + 1,
                'prediction': all_preds[i, h],
                'target': all_targets[i, h]
            })

    pred_df = pd.DataFrame(predictions_list)
    output_path = output_dir / "predictions_test_tft_aligned.csv"
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(pred_df)} predictions to {output_path}")

    logger.info(f"✅ TFT predictions generated: {len(all_preds)} samples × 24 horizons")


if __name__ == "__main__":
    generate_predictions()
