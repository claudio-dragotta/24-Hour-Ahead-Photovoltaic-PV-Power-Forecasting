#!/usr/bin/env python3
"""Evaluate model on preprocessed test set (already scaled with global MinMaxScaler)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from pv_forecasting.models.multi_branch_tft import MultiBranchTransformer


class TestDataset(Dataset):
    """Dataset for preprocessed test data."""

    def __init__(
        self,
        df: pd.DataFrame,
        pv_lag_features: list,
        weather_lag_features: list,
        forecast_features: list,
        target_col: str,
        seq_len: int,
        horizon: int,
        step: int = 24,
    ):
        self.df = df.reset_index(drop=True)
        self.pv_lag_features = pv_lag_features
        self.weather_lag_features = weather_lag_features
        self.forecast_features = forecast_features
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon

        # Create non-overlapping windows (step=24)
        self.valid_indices = []
        for i in range(0, len(df) - seq_len - horizon + 1, step):
            if i + seq_len + horizon - 1 < len(df):
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len
        target_start = end_idx
        target_end = target_start + self.horizon

        pv_history = self.df.loc[start_idx : end_idx - 1, self.pv_lag_features].values.astype(np.float32)
        weather_history = self.df.loc[start_idx : end_idx - 1, self.weather_lag_features].values.astype(np.float32)
        weather_forecast = self.df.loc[target_start : target_end - 1, self.forecast_features].values.astype(np.float32)
        targets = self.df.loc[target_start : target_end - 1, self.target_col].values.astype(np.float32)

        features = {
            "pv_history": torch.from_numpy(pv_history),
            "weather_history": torch.from_numpy(weather_history),
            "weather_forecast": torch.from_numpy(weather_forecast),
        }
        return features, torch.from_numpy(targets)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0.0

    # Normalized RMSE
    nrmse = rmse / (y_true.max() - y_true.min()) if (y_true.max() - y_true.min()) > 0 else 0.0

    # MBE (Mean Bias Error)
    mbe = np.mean(y_pred - y_true)

    # MASE (Mean Absolute Scaled Error)
    naive_errors = np.abs(np.diff(y_true))
    mase = mae / np.mean(naive_errors) if len(naive_errors) > 0 and np.mean(naive_errors) > 0 else 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "nrmse": float(nrmse),
        "mbe": float(mbe),
        "mase": float(mase),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-parquet", type=str, required=True, help="Preprocessed test parquet")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--step", type=int, default=24, help="Window step (24 for non-overlapping)")
    parser.add_argument("--outdir", type=str, default="outputs/test_eval")
    args = parser.parse_args()

    # Load test data
    df_test = pd.read_parquet(args.test_parquet)
    print(f"Loaded test data: {df_test.shape}")

    # Extract pv_kwp_max for denormalization
    pv_kwp_max = df_test["pv_kwp_max"].iloc[0] if "pv_kwp_max" in df_test.columns else 68.92
    print(f"PV max capacity: {pv_kwp_max:.2f} kW")

    # Define feature groups (matching training)
    pv_lag_features = ["pv_lag1"]
    weather_lag_features = ["ghi_lag1", "temp_lag1"]
    base_forecast = [
        c
        for c in [
            "temp",
            "humidity",
            "wind_speed",
            "ghi",
            "dni",
            "dhi",
            "sp_zenith",
            "sp_azimuth",
            "cs_ghi",
            "cs_dni",
            "cs_dhi",
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "is_weekend",
            "is_holiday",
            "pressure",
            "dew_point",
            "wind_deg",
            "rain_1h",
            "kc",
        ]
        if c in df_test.columns
    ]
    wx_features = [c for c in df_test.columns if c.startswith("wx_")]
    forecast_features = base_forecast + wx_features

    print(f"PV lag features: {len(pv_lag_features)}")
    print(f"Weather lag features: {len(weather_lag_features)}")
    print(f"Forecast features: {len(forecast_features)}")

    # Create dataset
    test_dataset = TestDataset(
        df_test,
        pv_lag_features,
        weather_lag_features,
        forecast_features,
        target_col="pv",
        seq_len=args.seq_len,
        horizon=args.horizon,
        step=args.step,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # Load model
    model = MultiBranchTransformer.load_from_checkpoint(args.ckpt)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded from {args.ckpt}")
    print(f"Using device: {device}")

    # Run inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            # Move to device
            for k in features:
                features[k] = features[k].to(device)
            targets = targets.to(device)

            # Predict (already in normalized [0,1] range)
            preds = model(features)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate
    y_pred_norm = np.concatenate(all_preds, axis=0)  # (N, 24)
    y_true_norm = np.concatenate(all_targets, axis=0)  # (N, 24)

    # Denormalize to kW
    y_pred_kw = y_pred_norm * pv_kwp_max
    y_true_kw = y_true_norm * pv_kwp_max

    # Flatten for metrics
    y_pred_flat = y_pred_kw.flatten()
    y_true_flat = y_true_kw.flatten()

    # Calculate metrics
    metrics = calculate_metrics(y_true_flat, y_pred_flat)

    # Save results
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(outdir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({"y_true_kw": y_true_flat, "y_pred_kw": y_pred_flat})
    pred_df.to_csv(outdir / "test_predictions.csv", index=False)

    # Print results
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {len(test_dataset)} non-overlapping windows")
    print(f"Total predictions: {len(y_pred_flat)} hours")
    print(f"\nMetrics (in kW):")
    print(f"  MAE:   {metrics['mae']:.4f} kW")
    print(f"  RMSE:  {metrics['rmse']:.4f} kW")
    print(f"  R²:    {metrics['r2']:.4f}")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  nRMSE: {metrics['nrmse']:.4f}")
    print(f"  MBE:   {metrics['mbe']:.4f} kW")
    print(f"  MASE:  {metrics['mase']:.4f}")
    print(f"\nResults saved to: {outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
