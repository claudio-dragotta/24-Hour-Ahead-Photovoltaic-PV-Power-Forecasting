"""ARCHIVE: generate_final_predictions.py

Original: scripts/inference/generate_final_predictions.py
Moved to scripts/_archived/inference for cleanup. Kept intact for history.
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


def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mase(y_true, y_pred, seasonality=24):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    mae_model = np.mean(np.abs(y_true_flat - y_pred_flat))
    if len(y_true_flat) > seasonality:
        naive_errors = np.abs(y_true_flat[seasonality:] - y_true_flat[:-seasonality])
        mae_naive = np.mean(naive_errors)
    else:
        mae_naive = np.mean(np.abs(y_true_flat))
    if mae_naive == 0:
        return np.nan
    return float(mae_model / mae_naive)


class PVForecastingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        pv_lag_features,
        weather_lag_features,
        forecast_features,
        target_col,
        seq_len,
        horizon,
        pv_scaler,
        weather_scaler,
        forecast_scaler,
        target_scaler,
    ):
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

        pv_history = self.df.loc[start_idx : end_idx - 1, self.pv_lag_features].values.astype(np.float32)
        weather_history = self.df.loc[start_idx : end_idx - 1, self.weather_lag_features].values.astype(np.float32)
        weather_forecast = self.df.loc[target_start : target_end - 1, self.forecast_features].values.astype(np.float32)
        targets = self.df.loc[target_start : target_end - 1, self.target_col].values.astype(np.float32)

        if self.pv_scaler is not None:
            pv_history = self.pv_scaler.transform(pv_history)
        if self.weather_scaler is not None:
            weather_history = self.weather_scaler.transform(weather_history)
        if self.forecast_scaler is not None:
            weather_forecast = self.forecast_scaler.transform(weather_forecast)
        if self.target_scaler is not None:
            targets = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()

        features = {
            "pv_history": torch.from_numpy(pv_history.astype(np.float32)),
            "weather_history": torch.from_numpy(weather_history.astype(np.float32)),
            "weather_forecast": torch.from_numpy(weather_forecast.astype(np.float32)),
        }
        targets = torch.from_numpy(targets.astype(np.float32))

        return features, targets


def generate_predictions():
    logger.info("Generating FINAL predictions (archived script)")
    model_dir = Path("outputs/multi_branch/final_seed2")
    data_path = Path("outputs/processed.parquet")
    with open(model_dir / "config_model.json") as f:
        config = json.load(f)
    with open(model_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    df = pd.read_parquet(data_path)
    n_samples = len(df)
    train_ratio = 0.6
    val_ratio = 0.2
    cutoff_train = int(n_samples * train_ratio)
    cutoff_val = int(n_samples * (train_ratio + val_ratio))
    test_df = df.iloc[cutoff_val:].copy()

    test_dataset = PVForecastingDataset(
        test_df,
        config["pv_lag_features"],
        config["weather_lag_features"],
        config["forecast_features"],
        "pv",
        config["seq_len"],
        config["horizon"],
        scalers["pv_scaler"],
        scalers["weather_scaler"],
        scalers["forecast_scaler"],
        scalers["target_scaler"],
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    checkpoint_path = model_dir / "multi-branch-best.ckpt"
    model = MultiBranchTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    all_preds = scalers["target_scaler"].inverse_transform(all_preds)
    all_targets = scalers["target_scaler"].inverse_transform(all_targets)

    rmse_val = compute_rmse(all_targets.flatten(), all_preds.flatten())
    mase_val = compute_mase(all_targets, all_preds, seasonality=24)

    pred_cols = {f"pred_h{h+1}": all_preds[:, h] for h in range(24)}
    target_cols = {f"actual_h{h+1}": all_targets[:, h] for h in range(24)}
    pred_df_wide = pd.DataFrame({**pred_cols, **target_cols})
    (model_dir / "predictions_test_wide.csv").parent.mkdir(parents=True, exist_ok=True)
    pred_df_wide.to_csv(model_dir / "predictions_test_wide.csv", index=False)

    print(f"Archived script ran: RMSE={rmse_val:.4f}, MASE={mase_val}")


if __name__ == "__main__":
    generate_predictions()
