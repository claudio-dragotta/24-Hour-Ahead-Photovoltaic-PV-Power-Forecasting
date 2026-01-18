#!/usr/bin/env python3
"""Evaluate CNN-BiLSTM model on test set."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def create_sliding_windows(
    df: pd.DataFrame,
    pv_features: List[str],
    weather_hist_features: List[str],
    forecast_features: List[str],
    target_col: str,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows for inference (step=1 for test)."""
    valid_indices = []
    for i in range(len(df) - seq_len - horizon + 1):
        if i + seq_len + horizon - 1 < len(df):
            valid_indices.append(i)

    n_samples = len(valid_indices)
    pv_history = np.zeros((n_samples, seq_len, len(pv_features)), dtype=np.float32)
    weather_history = np.zeros((n_samples, seq_len, len(weather_hist_features)), dtype=np.float32)
    weather_forecast = np.zeros((n_samples, horizon, len(forecast_features)), dtype=np.float32)
    targets = np.zeros((n_samples, horizon), dtype=np.float32)

    for idx, start in enumerate(valid_indices):
        end = start + seq_len
        target_start = end
        target_end = target_start + horizon

        pv_history[idx] = df.iloc[start:end][pv_features].values
        weather_history[idx] = df.iloc[start:end][weather_hist_features].values
        weather_forecast[idx] = df.iloc[target_start:target_end][forecast_features].values
        targets[idx] = df.iloc[target_start:target_end][target_col].values

    return pv_history, weather_history, weather_forecast, targets


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate evaluation metrics."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))

    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    mbe = np.mean(y_pred_flat - y_true_flat)

    # MASE
    naive_errors = np.abs(np.diff(y_true_flat))
    mase = mae / np.mean(naive_errors) if len(naive_errors) > 0 and np.mean(naive_errors) > 0 else 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mbe": float(mbe),
        "mase": float(mase),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="outputs/cnn_bilstm/baseline")
    parser.add_argument(
        "--processed-test",
        type=str,
        default="data/processed/merged/pv_wx_combined.parquet",
        help="Path to preprocessed test parquet (same schema as training)",
    )
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = model_dir / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and scalers
    print(f"Loading model from {model_dir}")
    model = keras.models.load_model(model_dir / "model_best.keras")

    with open(model_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # Load preprocessed test data
    print(f"Loading test data from {args.processed_test}")
    df = pd.read_parquet(args.processed_test)
    df = df.sort_index()
    print(f"Test data shape: {df.shape}")

    # Define feature groups (same as training)
    pv_lag_features = [c for c in df.columns if c.startswith("pv_lag") or c.startswith("pv_roll")]
    if not pv_lag_features:
        pv_lag_features = ["pv"] if "pv" in df.columns else []

    weather_lag_features = [
        c
        for c in df.columns
        if c.startswith(("ghi_lag", "dni_lag", "dhi_lag", "temp_lag")) or ("roll" in c and not c.startswith("pv_roll"))
    ]

    base_forecast = [
        c
        for c in [
            "temp",
            "humidity",
            "wind_speed",
            "clouds",
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
        if c in df.columns
    ]
    wx_features = [c for c in df.columns if c.startswith("wx_")]
    forecast_features = base_forecast + wx_features

    print(f"PV features: {len(pv_lag_features)}")
    print(f"Weather history features: {len(weather_lag_features)}")
    print(f"Forecast features: {len(forecast_features)}")

    # Create windows
    print("Creating sliding windows...")
    pv_history, weather_history, weather_forecast, targets = create_sliding_windows(
        df, pv_lag_features, weather_lag_features, forecast_features, "pv", args.seq_len, args.horizon
    )
    print(f"Test samples: {len(targets)}")

    # Scale data
    def scale_3d(data: np.ndarray, scaler) -> np.ndarray:
        shape = data.shape
        flat = data.reshape(-1, shape[-1])
        scaled = scaler.transform(flat)
        return scaled.reshape(shape)

    pv_scaled = scale_3d(pv_history, scalers["pv_scaler"])
    weather_hist_scaled = scale_3d(weather_history, scalers["weather_hist_scaler"])
    weather_fcst_scaled = scale_3d(weather_forecast, scalers["weather_fcst_scaler"])

    # Predict
    print("Generating predictions...")
    preds_scaled = model.predict(
        [pv_scaled, weather_hist_scaled, weather_fcst_scaled], batch_size=args.batch_size, verbose=1
    )

    # Inverse transform
    preds = scalers["target_scaler"].inverse_transform(preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)

    # Calculate metrics
    metrics = calculate_metrics(targets, preds)
    print(f"\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    # Save predictions
    df_out = pd.DataFrame(
        {
            "y_true_kw": targets.flatten(),
            "y_pred_kw": preds.flatten(),
        }
    )
    df_out.to_csv(out_dir / "test_predictions.csv", index=False)

    # Save metrics
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
