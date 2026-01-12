"""ARCHIVE: predict_multi_branch.py

Original: scripts/inference/predict_multi_branch.py
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

    return features_list, origin_indices


def main() -> None:
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    model = MultiBranchTransformer.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)
    df = pd.read_parquet(args.processed_data)
    seq_len = model.seq_len_encoder
    horizon = model.seq_len_decoder
    features_list, origin_indices = prepare_features(df, seq_len, horizon)
    all_predictions = []
    num_batches = (len(features_list) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(features_list))
            batch_features = features_list[start_idx:end_idx]
            pv_batch = torch.from_numpy(np.stack([f["pv_history"] for f in batch_features], axis=0)).to(device)
            weather_hist_batch = torch.from_numpy(np.stack([f["weather_history"] for f in batch_features], axis=0)).to(device)
            weather_fcst_batch = torch.from_numpy(
                np.stack([f["weather_forecast"] for f in batch_features], axis=0)
            ).to(device)
            batch_dict = {"pv_history": pv_batch, "weather_history": weather_hist_batch, "weather_forecast": weather_fcst_batch}
            predictions = model(batch_dict)
            all_predictions.append(predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    rows = []
    for i, origin_idx in enumerate(origin_indices):
        origin_timestamp = df.index[origin_idx]
        for h in range(1, horizon + 1):
            forecast_idx = origin_idx + h
            if forecast_idx < len(df):
                forecast_timestamp = df.index[forecast_idx]
                y_pred = all_predictions[i, h - 1]
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
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    print(f"Saved predictions to {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
