"""Inference script for Multi-Branch Transformer.

This script loads a trained Multi-Branch Transformer model and generates
24-hour ahead PV power predictions on new data.

Usage:
    python scripts/inference/predict_multi_branch.py \
        --checkpoint outputs/multi_branch/baseline/multi-branch-best.ckpt \
        --processed-data outputs/processed.parquet \
        --outdir predictions_multi_branch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pv_forecasting.logger import get_logger
from pv_forecasting.models.multi_branch_tft import MultiBranchTransformer

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Multi-Branch Transformer Inference")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.ckpt file)")
    ap.add_argument("--processed-data", type=str, required=True, help="Path to processed parquet file with features")
    ap.add_argument("--outdir", type=str, default="predictions_multi_branch", help="Output directory for predictions")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    ap.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to run inference on"
    )
    return ap.parse_args()


def prepare_features(
    df: pd.DataFrame, seq_len: int = 168, horizon: int = 24
) -> tuple[List[Dict[str, np.ndarray]], List[int]]:
    """Prepare input features for model inference.

    Args:
        df: Processed dataframe with all features.
        seq_len: Encoder sequence length (default: 168 hours).
        horizon: Forecast horizon (default: 24 hours).

    Returns:
        Tuple of (feature_list, origin_indices) where feature_list contains
        dictionaries with 'pv_history', 'weather_history', 'weather_forecast'
        for each valid window.
    """
    # Define feature groups
    pv_lag_features = [c for c in df.columns if c.startswith("pv_lag") or c.startswith("pv_roll")]
    weather_lag_features = [
        c
        for c in df.columns
        if (
            c.startswith("ghi_lag")
            or c.startswith("dni_lag")
            or c.startswith("dhi_lag")
            or ("roll" in c and not c.startswith("pv_roll"))
            or ("var" in c and not c.startswith("pv_var"))
        )
    ]
    forecast_features = [
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
        ]
        if c in df.columns
    ]

    logger.info(
        f"Feature groups: PV lags={len(pv_lag_features)}, "
        f"Weather lags={len(weather_lag_features)}, "
        f"Forecast={len(forecast_features)}"
    )

    # Create sliding windows
    features_list = []
    origin_indices = []

    df_reset = df.reset_index(drop=True)

    for i in range(len(df) - seq_len - horizon + 1):
        start_idx = i
        end_idx = i + seq_len
        forecast_start = end_idx
        forecast_end = forecast_start + horizon

        if forecast_end > len(df_reset):
            break

        # Extract features for each branch
        pv_history = df_reset.loc[start_idx : end_idx - 1, pv_lag_features].values
        weather_history = df_reset.loc[start_idx : end_idx - 1, weather_lag_features].values
        weather_forecast = df_reset.loc[forecast_start : forecast_end - 1, forecast_features].values

        features = {
            "pv_history": pv_history.astype(np.float32),
            "weather_history": weather_history.astype(np.float32),
            "weather_forecast": weather_forecast.astype(np.float32),
        }

        features_list.append(features)
        origin_indices.append(end_idx - 1)

    logger.info(f"Created {len(features_list)} prediction windows")

    return features_list, origin_indices


def main() -> None:
    """Main inference function."""
    args = parse_args()

    # Setup output directory
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Load trained model
    logger.info(f"Loading model from {args.checkpoint}")
    model = MultiBranchTransformer.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)
    logger.info("Model loaded successfully")

    # Load data
    logger.info(f"Loading processed data from {args.processed_data}")
    df = pd.read_parquet(args.processed_data)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")

    # Prepare features
    seq_len = model.seq_len_encoder
    horizon = model.seq_len_decoder
    logger.info(f"Model expects encoder length: {seq_len}h, decoder length: {horizon}h")

    features_list, origin_indices = prepare_features(df, seq_len, horizon)

    # Run inference
    logger.info("Running inference...")
    all_predictions = []

    # Process in batches
    num_batches = (len(features_list) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Inference"):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(features_list))

            batch_features = features_list[start_idx:end_idx]

            # Stack batch
            pv_batch = torch.from_numpy(np.stack([f["pv_history"] for f in batch_features], axis=0)).to(device)
            weather_hist_batch = torch.from_numpy(np.stack([f["weather_history"] for f in batch_features], axis=0)).to(
                device
            )
            weather_fcst_batch = torch.from_numpy(np.stack([f["weather_forecast"] for f in batch_features], axis=0)).to(
                device
            )

            batch_dict = {
                "pv_history": pv_batch,
                "weather_history": weather_hist_batch,
                "weather_forecast": weather_fcst_batch,
            }

            # Predict
            predictions = model(batch_dict)
            all_predictions.append(predictions.cpu().numpy())

    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    logger.info(f"Generated predictions shape: {all_predictions.shape}")

    # Create output DataFrame
    logger.info("Creating output dataframe...")
    rows = []

    for i, origin_idx in enumerate(origin_indices):
        origin_timestamp = df.index[origin_idx]

        for h in range(1, horizon + 1):
            forecast_idx = origin_idx + h
            if forecast_idx < len(df):
                forecast_timestamp = df.index[forecast_idx]
                y_pred = all_predictions[i, h - 1]

                # Also include ground truth if available
                y_true = df.iloc[forecast_idx]["pv"] if "pv" in df.columns else None

                rows.append(
                    {
                        "origin_timestamp_utc": origin_timestamp.isoformat(),
                        "forecast_timestamp_utc": forecast_timestamp.isoformat(),
                        "horizon_h": h,
                        "y_pred": float(y_pred),
                        "y_true": float(y_true) if y_true is not None else None,
                    }
                )

    pred_df = pd.DataFrame(rows)

    # Save predictions
    output_path = out_dir / "predictions.csv"
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

    # Compute summary statistics if ground truth available
    if "y_true" in pred_df.columns and pred_df["y_true"].notna().any():
        valid_mask = pred_df["y_true"].notna()
        y_true = pred_df.loc[valid_mask, "y_true"].values
        y_pred = pred_df.loc[valid_mask, "y_pred"].values

        from pv_forecasting.metrics import mase, rmse

        # Compute metrics (use simple train series for MASE baseline)
        train_series = df["pv"].values[: int(len(df) * 0.6)]

        metrics = {
            "rmse": float(rmse(y_true, y_pred)),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
            "mase": float(mase(y_true, y_pred, train_series, m=24)),
            "num_predictions": len(y_true),
        }

        metrics_path = out_dir / "metrics_summary.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, indent=2, fp=f)

        logger.info(f"Metrics summary:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  MASE: {metrics['mase']:.4f}")
        logger.info(f"Saved metrics to {metrics_path}")

    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
