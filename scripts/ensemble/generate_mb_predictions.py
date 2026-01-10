"""Generate Multi-Branch test predictions for ensemble.

This script loads the trained Multi-Branch model and generates predictions
on the test set, saving them in the same format as TFT for ensemble combination.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pv_forecasting.logger import get_logger
from pv_forecasting.models.multi_branch_tft import MultiBranchTransformer

logger = get_logger(__name__)


class PVForecastingDataset(torch.utils.data.Dataset):
    """Simple dataset for inference."""

    def __init__(self, df, pv_lag_features, weather_lag_features, forecast_features,
                 target_col, seq_len, horizon, pv_scaler, weather_scaler, forecast_scaler, target_scaler):
        self.df = df.reset_index(drop=True)
        self.pv_lag_features = pv_lag_features
        self.weather_lag_features = weather_lag_features
        self.forecast_features = forecast_features
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        self.pv_scaler = pv_scaler
        self.weather_scaler = weather_scaler
        self.forecast_scaler = forecast_scaler
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

        pv_history = self.df.loc[start_idx:end_idx-1, self.pv_lag_features].values.astype(np.float32)
        weather_history = self.df.loc[start_idx:end_idx-1, self.weather_lag_features].values.astype(np.float32)
        weather_forecast = self.df.loc[target_start:target_end-1, self.forecast_features].values.astype(np.float32)
        targets = self.df.loc[target_start:target_end-1, self.target_col].values.astype(np.float32)

        if self.pv_scaler is not None:
            pv_history = self.pv_scaler.transform(pv_history)
        if self.weather_scaler is not None:
            weather_history = self.weather_scaler.transform(weather_history)
        if self.forecast_scaler is not None:
            weather_forecast = self.forecast_scaler.transform(weather_forecast)
        if self.target_scaler is not None:
            targets = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()

        features = {
            'pv_history': torch.from_numpy(pv_history.astype(np.float32)),
            'weather_history': torch.from_numpy(weather_history.astype(np.float32)),
            'weather_forecast': torch.from_numpy(weather_forecast.astype(np.float32))
        }
        targets = torch.from_numpy(targets.astype(np.float32))

        return features, targets


def generate_predictions():
    """Generate Multi-Branch test predictions."""
    logger.info("Generating Multi-Branch test predictions for ensemble")

    # Paths
    model_dir = Path("outputs/multi_branch/final_v1")
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

    # Split (same as training)
    n_samples = len(df)
    train_ratio = 0.6
    val_ratio = 0.2
    cutoff_train = int(n_samples * train_ratio)
    cutoff_val = int(n_samples * (train_ratio + val_ratio))

    test_df = df.iloc[cutoff_val:]
    logger.info(f"Test set: {len(test_df)} samples")

    # Create dataset
    test_dataset = PVForecastingDataset(
        test_df,
        config['pv_lag_features'],
        config['weather_lag_features'],
        config['forecast_features'],
        'pv',
        config['seq_len'],
        config['horizon'],
        scalers['pv_scaler'],
        scalers['weather_scaler'],
        scalers['forecast_scaler'],
        scalers['target_scaler']
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    logger.info(f"Test batches: {len(test_loader)}")

    # Load model
    checkpoint_path = model_dir / "multi-branch-best.ckpt"
    logger.info(f"Loading model from {checkpoint_path}")
    model = MultiBranchTransformer.load_from_checkpoint(checkpoint_path)
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

    # Save in TFT format (long format)
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
    output_path = output_dir / "predictions_test_multi_branch.csv"
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(pred_df)} predictions to {output_path}")

    logger.info(f"✅ Multi-Branch predictions generated: {len(all_preds)} samples × 24 horizons")


if __name__ == "__main__":
    generate_predictions()
