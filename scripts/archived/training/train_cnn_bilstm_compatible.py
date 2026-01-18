#!/usr/bin/env python3
"""Train CNN-BiLSTM fusion model compatible with ensemble pipeline.

This script trains the CNN-BiLSTM architecture using the same data pipeline
as train_multi_branch.py, producing outputs compatible with the ensemble scripts.
"""

from __future__ import annotations

import os

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from pv_forecasting.models.cnn_bilstm import build_cnn_bilstm_fusion_attention

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Train CNN-BiLSTM fusion model")
    ap.add_argument(
        "--processed-path",
        type=str,
        default="data/processed/merged/pv_wx_combined.parquet",
        help="Path to preprocessed parquet file",
    )
    ap.add_argument("--seq-len", type=int, default=24, help="Encoder length in hours")
    ap.add_argument("--horizon", type=int, default=24, help="Prediction horizon in hours")
    ap.add_argument("--window-step", type=int, default=4, help="Sliding window step")
    ap.add_argument("--epochs", type=int, default=200, help="Maximum training epochs")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size")
    ap.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    ap.add_argument("--num-attention-heads", type=int, default=8, help="Number of attention heads")
    ap.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    ap.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--early-stopping-patience", type=int, default=15, help="Early stopping patience")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    ap.add_argument("--outdir", type=str, default="outputs/cnn_bilstm/baseline", help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()


def create_sliding_windows(
    df: pd.DataFrame,
    pv_features: List[str],
    weather_hist_features: List[str],
    forecast_features: List[str],
    target_col: str,
    seq_len: int,
    horizon: int,
    window_step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Create sliding windows for training.

    Returns:
        pv_history: (n_samples, seq_len, n_pv_features)
        weather_history: (n_samples, seq_len, n_weather_features)
        weather_forecast: (n_samples, horizon, n_forecast_features)
        targets: (n_samples, horizon)
        valid_indices: List of starting indices
    """
    valid_indices = []
    for i in range(0, len(df) - seq_len - horizon + 1, window_step):
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

        # PV history
        pv_history[idx] = df.iloc[start:end][pv_features].values

        # Weather history
        weather_history[idx] = df.iloc[start:end][weather_hist_features].values

        # Weather forecast (features for the forecast period)
        weather_forecast[idx] = df.iloc[target_start:target_end][forecast_features].values

        # Targets
        targets[idx] = df.iloc[target_start:target_end][target_col].values

    return pv_history, weather_history, weather_forecast, targets, valid_indices


def main() -> None:
    """Main training function."""
    args = parse_args()
    set_seeds(args.seed)

    logger.info(f"Starting CNN-BiLSTM training with seed={args.seed}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.processed_path}")
    df = pd.read_parquet(args.processed_path)
    df = df.sort_index()

    # Define feature groups (same as train_multi_branch.py)
    # PV lag features
    pv_lag_features = [c for c in df.columns if c.startswith("pv_lag") or c.startswith("pv_roll")]
    if not pv_lag_features:
        # Fallback: use pv column directly
        pv_lag_features = ["pv"] if "pv" in df.columns else []

    # Weather lag features
    weather_lag_features = [
        c
        for c in df.columns
        if c.startswith(("ghi_lag", "dni_lag", "dhi_lag", "temp_lag")) or ("roll" in c and not c.startswith("pv_roll"))
    ]

    # Forecast features (same features used during forecast period)
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

    # Add weather one-hot features
    wx_features = [c for c in df.columns if c.startswith("wx_")]
    forecast_features = base_forecast + wx_features

    logger.info(f"PV lag features: {len(pv_lag_features)}")
    logger.info(f"Weather lag features: {len(weather_lag_features)}")
    logger.info(f"Forecast features: {len(forecast_features)}")

    # Create sliding windows
    logger.info("Creating sliding windows...")
    pv_history, weather_history, weather_forecast, targets, valid_indices = create_sliding_windows(
        df,
        pv_lag_features,
        weather_lag_features,
        forecast_features,
        "pv",
        args.seq_len,
        args.horizon,
        args.window_step,
    )

    logger.info(f"Created {len(valid_indices)} samples")
    logger.info(f"PV history shape: {pv_history.shape}")
    logger.info(f"Weather history shape: {weather_history.shape}")
    logger.info(f"Weather forecast shape: {weather_forecast.shape}")
    logger.info(f"Targets shape: {targets.shape}")

    # Split data (chronological)
    n_train = int(len(valid_indices) * (1 - args.val_ratio))

    pv_train, pv_val = pv_history[:n_train], pv_history[n_train:]
    weather_hist_train, weather_hist_val = weather_history[:n_train], weather_history[n_train:]
    weather_fcst_train, weather_fcst_val = weather_forecast[:n_train], weather_forecast[n_train:]
    targets_train, targets_val = targets[:n_train], targets[n_train:]

    logger.info(f"Train samples: {n_train}, Val samples: {len(valid_indices) - n_train}")

    # Fit scalers on training data
    pv_scaler = StandardScaler()
    weather_hist_scaler = StandardScaler()
    weather_fcst_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Reshape for scaling (flatten time dimension)
    pv_train_flat = pv_train.reshape(-1, pv_train.shape[-1])
    weather_hist_train_flat = weather_hist_train.reshape(-1, weather_hist_train.shape[-1])
    weather_fcst_train_flat = weather_fcst_train.reshape(-1, weather_fcst_train.shape[-1])
    targets_train_flat = targets_train.reshape(-1, 1)

    pv_scaler.fit(pv_train_flat)
    weather_hist_scaler.fit(weather_hist_train_flat)
    weather_fcst_scaler.fit(weather_fcst_train_flat)
    target_scaler.fit(targets_train_flat)

    # Transform data
    def scale_3d(data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        shape = data.shape
        flat = data.reshape(-1, shape[-1])
        scaled = scaler.transform(flat)
        return scaled.reshape(shape)

    pv_train_scaled = scale_3d(pv_train, pv_scaler)
    pv_val_scaled = scale_3d(pv_val, pv_scaler)
    weather_hist_train_scaled = scale_3d(weather_hist_train, weather_hist_scaler)
    weather_hist_val_scaled = scale_3d(weather_hist_val, weather_hist_scaler)
    weather_fcst_train_scaled = scale_3d(weather_fcst_train, weather_fcst_scaler)
    weather_fcst_val_scaled = scale_3d(weather_fcst_val, weather_fcst_scaler)

    targets_train_scaled = target_scaler.transform(targets_train.reshape(-1, 1)).reshape(targets_train.shape)
    targets_val_scaled = target_scaler.transform(targets_val.reshape(-1, 1)).reshape(targets_val.shape)

    # Save scalers
    scalers = {
        "pv_scaler": pv_scaler,
        "weather_hist_scaler": weather_hist_scaler,
        "weather_fcst_scaler": weather_fcst_scaler,
        "target_scaler": target_scaler,
    }
    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    # Build model
    logger.info("Building CNN-BiLSTM fusion model...")
    model = build_cnn_bilstm_fusion_attention(
        pv_seq_len=args.seq_len,
        weather_seq_len=args.seq_len,
        weather_forecast_len=args.horizon,
        weather_hist_features=len(weather_lag_features),
        weather_forecast_features=len(forecast_features),
        horizon=args.horizon,
        embedding_dim=args.embedding_dim,
        num_attention_heads=args.num_attention_heads,
        dropout_rate=args.dropout,
        pv_features=len(pv_lag_features),
    )

    # Recompile with custom learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
    )

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=8,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "model_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        [pv_train_scaled, weather_hist_train_scaled, weather_fcst_train_scaled],
        targets_train_scaled,
        validation_data=(
            [pv_val_scaled, weather_hist_val_scaled, weather_fcst_val_scaled],
            targets_val_scaled,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    model.save(out_dir / "model_final.keras")

    # Generate predictions on validation set
    logger.info("Generating validation predictions...")
    val_preds_scaled = model.predict(
        [pv_val_scaled, weather_hist_val_scaled, weather_fcst_val_scaled],
        batch_size=args.batch_size,
    )

    # Inverse transform predictions
    val_preds = target_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).reshape(val_preds_scaled.shape)
    val_targets = targets_val  # Already in original scale

    # Save predictions in compatible format
    val_preds_df = pd.DataFrame(
        {
            "y_true_kw": val_targets.flatten(),
            "y_pred_kw": val_preds.flatten(),
        }
    )
    val_preds_df.to_csv(out_dir / "val_predictions.csv", index=False)

    # Calculate metrics
    from pv_forecasting.metrics import rmse, mase

    val_rmse = rmse(val_targets.flatten(), val_preds.flatten())
    # For MASE, we need the training series - use training targets
    val_mase = mase(val_targets.flatten(), val_preds.flatten(), targets_train.flatten(), m=24)
    metrics = {"rmse": float(val_rmse), "mase": float(val_mase)}
    logger.info(f"Validation metrics: {metrics}")

    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    logger.info(f"Training complete. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
