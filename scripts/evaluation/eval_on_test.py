#!/usr/bin/env python3
"""Evaluate trained Multi-Branch Transformer on a test set.

Steps:
1) Preprocess test PV/WX data to match the training feature set.
2) Load model checkpoint and scalers produced by `train_multi_branch.py`.
3) Run inference (sliding windows), denormalize predictions, and compare with ground truth.
4) Save predictions CSV and metrics JSON in the chosen output directory.
"""
from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader, Dataset

from pv_forecasting.data import align_hourly
from pv_forecasting.models.multi_branch_tft import MultiBranchTransformer
from pv_forecasting.pipeline import load_and_engineer_features


class InferenceDataset(Dataset):
    """Dataset identical to the training one, for sliding-window inference."""

    def __init__(
        self,
        df: pd.DataFrame,
        pv_lag_features: List[str],
        weather_lag_features: List[str],
        forecast_features: List[str],
        target_col: str,
        seq_len: int,
        horizon: int,
    ):
        self.df = df.reset_index(drop=True)
        self.pv_lag_features = pv_lag_features
        self.weather_lag_features = weather_lag_features
        self.forecast_features = forecast_features
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon

        self.valid_indices = []
        for i in range(len(df) - seq_len - horizon + 1):
            if i + seq_len + horizon - 1 < len(df):
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
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
        targets_tensor = torch.from_numpy(targets)
        return features, targets_tensor


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics.

    Args:
        y_true: Ground truth values (any shape)
        y_pred: Predicted values (same shape as y_true)

    Returns:
        Dictionary with MAE, RMSE, R², MAPE, nRMSE, MBE
    """
    # Flatten arrays for calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Remove any NaN or infinite values
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]

    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))

    # RMSE: Root Mean Square Error
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))

    # R²: Coefficient of determination
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE: Mean Absolute Percentage Error (only where y_true > 0)
    mask_nonzero = y_true_flat > 0
    if np.any(mask_nonzero):
        mape = (
            np.mean(np.abs((y_true_flat[mask_nonzero] - y_pred_flat[mask_nonzero]) / y_true_flat[mask_nonzero])) * 100
        )
    else:
        mape = 0.0

    # nRMSE: Normalized RMSE (normalized by mean of observations)
    mean_true = np.mean(y_true_flat)
    nrmse = (rmse / mean_true * 100) if mean_true > 0 else 0.0

    # MBE: Mean Bias Error
    mbe = np.mean(y_pred_flat - y_true_flat)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "nrmse": float(nrmse),
        "mbe": float(mbe),
    }


def create_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    out_dir: Path,
) -> None:
    """Create visualization plots comparing predictions with ground truth.

    Args:
        y_true: Ground truth values (n_samples, horizon)
        y_pred: Predicted values (n_samples, horizon)
        horizon: Forecast horizon length
        out_dir: Output directory for saving plots
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("PV Power Forecasting Evaluation", fontsize=16, fontweight="bold")

    # 1. Scatter plot: Predictions vs Ground Truth
    ax1 = axes[0, 0]
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    ax1.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10)
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect prediction")
    ax1.set_xlabel("Ground Truth (kW)", fontsize=12)
    ax1.set_ylabel("Predictions (kW)", fontsize=12)
    ax1.set_title("Predictions vs Ground Truth", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Time series: First 100 samples (all horizons flattened)
    ax2 = axes[0, 1]
    n_samples = min(100, len(y_true_flat))
    x_range = np.arange(n_samples)
    ax2.plot(x_range, y_true_flat[:n_samples], label="Ground Truth", linewidth=2, alpha=0.7)
    ax2.plot(x_range, y_pred_flat[:n_samples], label="Predictions", linewidth=2, alpha=0.7)
    ax2.set_xlabel("Sample Index", fontsize=12)
    ax2.set_ylabel("PV Power (kW)", fontsize=12)
    ax2.set_title(f"Time Series Comparison (First {n_samples} points)", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Residuals distribution
    ax3 = axes[1, 0]
    residuals = y_pred_flat - y_true_flat
    ax3.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax3.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero residual")
    ax3.set_xlabel("Residuals (kW)", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)
    ax3.set_title("Residuals Distribution", fontsize=13, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Per-horizon MAE
    ax4 = axes[1, 1]
    per_horizon_mae = []
    for h in range(horizon):
        mae_h = np.mean(np.abs(y_true[:, h] - y_pred[:, h]))
        per_horizon_mae.append(mae_h)
    hours = np.arange(1, horizon + 1)
    ax4.bar(hours, per_horizon_mae, alpha=0.7, edgecolor="black")
    ax4.set_xlabel("Forecast Horizon (hours ahead)", fontsize=12)
    ax4.set_ylabel("MAE (kW)", fontsize=12)
    ax4.set_title("MAE by Forecast Horizon", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = out_dir / "evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {plot_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate Multi-Branch Transformer on test set")
    ap.add_argument(
        "--processed-test", type=str, default=None, help="Optional: preprocessed test parquet (same schema as train)"
    )
    ap.add_argument("--pv-path", type=str, default="data/test/pv_test.xlsx", help="Path to test PV Excel file")
    ap.add_argument(
        "--wx-path", type=str, default="data/test/wx_test.parquet", help="Path to test weather file (parquet or Excel)"
    )
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney")
    ap.add_argument(
        "--ckpt", type=str, default="outputs/multi_branch/baseline/model.ckpt", help="Trained model checkpoint"
    )
    ap.add_argument(
        "--scalers", type=str, default="outputs/multi_branch/baseline/scalers.pkl", help="Scalers pickle from training"
    )
    ap.add_argument("--outdir", type=str, default="outputs/multi_branch/test_eval")
    ap.add_argument("--seq-len", type=int, default=168)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=64)
    return ap.parse_args()


def preprocess_test(pv_path: Path, wx_path: Path, local_tz: str) -> pd.DataFrame:
    """Produce feature-engineered dataframe for test data using the training pipeline."""
    if wx_path.suffix.lower() == ".parquet":
        wx_df = pd.read_parquet(wx_path)
        # Expect a timestamp column; prefer dt_utc if present
        ts_col = "dt_utc" if "dt_utc" in wx_df.columns else wx_df.columns[0]
        wx_df[ts_col] = pd.to_datetime(wx_df[ts_col], utc=True)
        # Save temporary Excel with dt_iso (required by load_wx_xlsx inside pipeline)
        tmp_wx = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        wx_df.to_excel(tmp_wx.name, index=False)
        wx_path_for_pipeline = Path(tmp_wx.name)
    else:
        wx_path_for_pipeline = wx_path

    df = load_and_engineer_features(pv_path, wx_path_for_pipeline, local_tz)
    return df


def main() -> None:
    args = parse_args()
    seed_everything(2)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pv_path = Path(args.pv_path)
    wx_path = Path(args.wx_path)
    ckpt_path = Path(args.ckpt)
    scalers_path = Path(args.scalers)

    if args.processed_test is None:
        if not pv_path.exists() or not wx_path.exists():
            raise SystemExit("PV or WX test file not found.")
    else:
        processed_test_path = Path(args.processed_test)
        if not processed_test_path.exists():
            raise SystemExit("Processed test parquet not found.")

    if not ckpt_path.exists() or not scalers_path.exists():
        raise SystemExit("Checkpoint or scalers not found. Run training first.")

    # Preprocess test set to match training features (or load preprocessed)
    if args.processed_test:
        df = pd.read_parquet(args.processed_test)
    else:
        df = preprocess_test(pv_path, wx_path, args.local_tz)

    target = "pv"
    pv_lag_features = [c for c in df.columns if c.startswith("pv_lag") or c.startswith("pv_roll")]
    weather_lag_features = [
        c
        for c in df.columns
        if (
            c.startswith("ghi_lag")
            or c.startswith("dni_lag")
            or c.startswith("dhi_lag")
            or c.startswith("temp_lag")
            or "roll" in c
            and not c.startswith("pv_roll")
        )
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

    # Load model and scalers
    scalers = pickle.load(open(scalers_path, "rb"))
    model = MultiBranchTransformer.load_from_checkpoint(ckpt_path)
    model.eval()

    # Apply scalers to features/target
    df_scaled = df.copy()
    if pv_lag_features:
        df_scaled[pv_lag_features] = scalers["pv_scaler"].transform(df_scaled[pv_lag_features].values)
    if weather_lag_features:
        df_scaled[weather_lag_features] = scalers["weather_scaler"].transform(df_scaled[weather_lag_features].values)
    if forecast_features:
        df_scaled[forecast_features] = scalers["forecast_scaler_cont"].transform(df_scaled[forecast_features].values)
    df_scaled[target] = scalers["target_scaler"].transform(df_scaled[target].values.reshape(-1, 1))

    dataset = InferenceDataset(
        df_scaled,
        pv_lag_features=pv_lag_features,
        weather_lag_features=weather_lag_features,
        forecast_features=forecast_features,
        target_col=target,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds_list = []
    targets_list = []
    with torch.no_grad():
        for features, targets_batch in loader:
            features_device = {k: v.to(device) for k, v in features.items()}
            preds = model(features_device)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(targets_batch.numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    preds_denorm = scalers["target_scaler"].inverse_transform(preds)
    targets_denorm = scalers["target_scaler"].inverse_transform(targets)

    # Calculate overall metrics
    metrics = calculate_metrics(targets_denorm, preds_denorm)

    # Calculate per-horizon metrics
    per_horizon_metrics = {}
    for h in range(args.horizon):
        h_metrics = calculate_metrics(targets_denorm[:, h], preds_denorm[:, h])
        per_horizon_metrics[f"h+{h+1}"] = h_metrics

    # Create visualizations
    create_visualizations(targets_denorm, preds_denorm, args.horizon, out_dir)

    # Save predictions
    df_out = pd.DataFrame(
        np.hstack([targets_denorm, preds_denorm]),
        columns=[f"y_true_h{h}" for h in range(1, args.horizon + 1)]
        + [f"y_pred_h{h}" for h in range(1, args.horizon + 1)],
    )
    preds_path = out_dir / "test_predictions.csv"
    df_out.to_csv(preds_path, index=False)

    # Save overall metrics
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save per-horizon metrics
    (out_dir / "test_metrics_per_horizon.json").write_text(json.dumps(per_horizon_metrics, indent=2))

    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"\nSaved predictions to {preds_path}")
    print(f"\nOverall Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name.upper():8s}: {metric_value:.4f}")
    print(f"\nPer-horizon metrics saved to: {out_dir / 'test_metrics_per_horizon.json'}")
    print(f"Visualizations saved to: {out_dir / 'evaluation_plots.png'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
