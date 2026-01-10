"""Training script for Multi-Branch Transformer with hierarchical fusion.

This script trains an advanced transformer architecture that processes PV history,
weather history, and future weather forecasts in separate branches before fusing
them adaptively using learned attention mechanisms.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from pv_forecasting.logger import get_logger
from pv_forecasting.metrics import mase, rmse
from pv_forecasting.models.multi_branch_tft import MultiBranchTransformer
from pv_forecasting.pipeline import load_and_engineer_features, persist_processed

logger = get_logger(__name__)


def compute_solar_weights(zenith_deg: pd.Series, min_weight: float = 0.1, gamma: float = 1.5) -> np.ndarray:
    """Compute sample weights based on solar zenith angle.

    Args:
        zenith_deg: Solar zenith angle in degrees.
        min_weight: Minimum weight for nighttime samples.
        gamma: Exponent to boost daytime importance.

    Returns:
        Sample weights normalized to mean=1.0.
    """
    zenith_rad = np.deg2rad(zenith_deg.values)
    cos_zenith = np.cos(zenith_rad)
    weights = np.maximum(cos_zenith, 0.0) ** gamma + min_weight
    weights = weights / weights.mean()
    return weights


class PVForecastingDataset(Dataset):
    """PyTorch dataset for multi-branch transformer training.

    Creates sliding windows and separates features into three branches:
    - PV history: lag features related to PV production
    - Weather history: lag features related to meteorological variables
    - Weather forecast: future weather covariates (if available)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pv_lag_features: List[str],
        weather_lag_features: List[str],
        forecast_features: List[str],
        target_col: str = 'pv',
        seq_len: int = 168,
        horizon: int = 24,
        sample_weights: Optional[np.ndarray] = None,
        pv_scaler: Optional[StandardScaler] = None,
        weather_scaler: Optional[StandardScaler] = None,
        forecast_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None
    ):
        """Initialize dataset with feature separation and normalization.

        Args:
            df: Processed dataframe with all features.
            pv_lag_features: List of PV lag feature names (e.g., ['pv_lag1', 'pv_lag24']).
            weather_lag_features: List of weather lag features (e.g., ['ghi_lag1', 'temp_lag24']).
            forecast_features: List of future weather features (e.g., ['temp', 'ghi']).
            target_col: Target column name (default: 'pv').
            seq_len: Encoder sequence length (default: 168 hours).
            horizon: Forecast horizon (default: 24 hours).
            sample_weights: Optional sample weights for loss computation.
            pv_scaler: StandardScaler for PV features (if None, no scaling).
            weather_scaler: StandardScaler for weather history features.
            forecast_scaler: StandardScaler for forecast features.
            target_scaler: StandardScaler for target normalization.
        """
        self.df = df.reset_index(drop=True)
        self.pv_lag_features = pv_lag_features
        self.weather_lag_features = weather_lag_features
        self.forecast_features = forecast_features
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        self.sample_weights = sample_weights
        self.pv_scaler = pv_scaler
        self.weather_scaler = weather_scaler
        self.forecast_scaler = forecast_scaler
        self.target_scaler = target_scaler

        # Create valid indices for sliding windows
        self.valid_indices = []
        for i in range(len(df) - seq_len - horizon + 1):
            # Ensure we have enough history and future
            if i + seq_len + horizon - 1 < len(df):
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single training sample with multi-branch structure and normalization.

        Returns:
            Tuple of (features_dict, targets):
                - features_dict: Dictionary with keys 'pv_history', 'weather_history', 'weather_forecast'
                - targets: Ground truth PV values for next 24 hours (normalized if scaler provided)
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len
        target_start = end_idx
        target_end = target_start + self.horizon

        # Extract features for each branch
        pv_history = self.df.loc[start_idx:end_idx-1, self.pv_lag_features].values.astype(np.float32)
        weather_history = self.df.loc[start_idx:end_idx-1, self.weather_lag_features].values.astype(np.float32)
        weather_forecast = self.df.loc[target_start:target_end-1, self.forecast_features].values.astype(np.float32)

        # Extract targets
        targets = self.df.loc[target_start:target_end-1, self.target_col].values.astype(np.float32)

        # Apply normalization if scalers are provided
        if self.pv_scaler is not None:
            pv_history = self.pv_scaler.transform(pv_history)
        if self.weather_scaler is not None:
            weather_history = self.weather_scaler.transform(weather_history)
        if self.forecast_scaler is not None:
            weather_forecast = self.forecast_scaler.transform(weather_forecast)
        if self.target_scaler is not None:
            targets = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()

        # Convert to tensors
        features = {
            'pv_history': torch.from_numpy(pv_history.astype(np.float32)),
            'weather_history': torch.from_numpy(weather_history.astype(np.float32)),
            'weather_forecast': torch.from_numpy(weather_forecast.astype(np.float32))
        }
        targets = torch.from_numpy(targets.astype(np.float32))

        return features, targets


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Train Multi-Branch Transformer for 24h-ahead PV forecasting")
    ap.add_argument(
        "--processed-path",
        type=str,
        default="outputs/processed.parquet",
        help="Path to pre-processed parquet file"
    )
    ap.add_argument("--pv-path", type=str, default="data/raw/pv_dataset.xlsx")
    ap.add_argument("--wx-path", type=str, default="data/raw/wx_dataset.xlsx")
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney")
    ap.add_argument("--seq-len", type=int, default=168, help="Encoder length in hours")
    ap.add_argument("--horizon", type=int, default=24, help="Prediction horizon in hours")
    ap.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=256, help="Model hidden dimension")
    ap.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    ap.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers per branch")
    ap.add_argument("--dim-feedforward", type=int, default=1024, help="Feedforward dimension")
    ap.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    ap.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    ap.add_argument("--early-stopping-patience", type=int, default=10)
    ap.add_argument("--dayweight-gamma", type=float, default=1.5, help="Solar weighting exponent")
    ap.add_argument("--dayweight-min", type=float, default=0.1, help="Minimum nighttime weight")
    ap.add_argument("--metrics-zenith-max", type=float, default=90.0, help="Exclude night from metrics")
    ap.add_argument("--outdir", type=str, default="outputs/multi_branch/baseline")
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()
    seed_everything(args.seed)
    logger.info(f"Starting Multi-Branch Transformer training with seed={args.seed}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    processed_path = Path(args.processed_path)
    if processed_path.exists():
        logger.info(f"Loading pre-processed data from {processed_path}")
        df = pd.read_parquet(processed_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    else:
        logger.info("Loading and engineering features from raw data")
        df = load_and_engineer_features(Path(args.pv_path), Path(args.wx_path), args.local_tz)
        persist_processed(df, Path("outputs"))

    target = "pv"

    # Compute solar-based sample weights
    if "sp_zenith" in df.columns:
        sample_weights = compute_solar_weights(
            df["sp_zenith"], min_weight=args.dayweight_min, gamma=args.dayweight_gamma
        )
        logger.info(f"Computed solar weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")
    else:
        sample_weights = None
        logger.warning("sp_zenith not found, using uniform weights")

    # Define feature groups for multi-branch architecture
    pv_lag_features = [c for c in df.columns if c.startswith('pv_lag') or c.startswith('pv_roll')]
    weather_lag_features = [
        c for c in df.columns
        if (c.startswith('ghi_lag') or c.startswith('dni_lag') or c.startswith('dhi_lag') or
            c.startswith('temp_lag') or 'roll' in c and not c.startswith('pv_roll'))
    ]
    forecast_features = [
        c for c in ['temp', 'humidity', 'wind_speed', 'clouds', 'ghi', 'dni', 'dhi',
                    'sp_zenith', 'sp_azimuth', 'cs_ghi', 'cs_dni', 'cs_dhi',
                    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'is_weekend', 'is_holiday']
        if c in df.columns
    ]

    logger.info(f"PV lag features ({len(pv_lag_features)}): {pv_lag_features[:5]}...")
    logger.info(f"Weather lag features ({len(weather_lag_features)}): {weather_lag_features[:5]}...")
    logger.info(f"Forecast features ({len(forecast_features)}): {forecast_features[:5]}...")

    # Chronological 3-way split
    n_samples = len(df)
    cutoff_train = int(n_samples * args.train_ratio)
    cutoff_val = int(n_samples * (args.train_ratio + args.val_ratio))

    logger.info(f"Chronological split:")
    logger.info(f"  Train: 0 to {cutoff_train} ({args.train_ratio:.1%})")
    logger.info(f"  Validation: {cutoff_train+1} to {cutoff_val} ({args.val_ratio:.1%})")
    logger.info(f"  Test: {cutoff_val+1} to {n_samples-1} ({args.test_ratio:.1%})")

    # Fit scalers on TRAINING set only (anti-leakage)
    train_df = df.iloc[:cutoff_train]
    pv_scaler = StandardScaler()
    weather_scaler = StandardScaler()
    forecast_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit on training data
    pv_scaler.fit(train_df[pv_lag_features].values)
    weather_scaler.fit(train_df[weather_lag_features].values)
    forecast_scaler.fit(train_df[forecast_features].values)
    target_scaler.fit(train_df[target].values.reshape(-1, 1))

    logger.info(f"Target scaling: mean={target_scaler.mean_[0]:.2f}, std={target_scaler.scale_[0]:.2f}")

    # Create datasets with scalers
    train_dataset = PVForecastingDataset(
        df.iloc[:cutoff_train],
        pv_lag_features,
        weather_lag_features,
        forecast_features,
        target,
        args.seq_len,
        args.horizon,
        sample_weights[:cutoff_train] if sample_weights is not None else None,
        pv_scaler=pv_scaler,
        weather_scaler=weather_scaler,
        forecast_scaler=forecast_scaler,
        target_scaler=target_scaler
    )
    val_dataset = PVForecastingDataset(
        df.iloc[cutoff_train:cutoff_val],
        pv_lag_features,
        weather_lag_features,
        forecast_features,
        target,
        args.seq_len,
        args.horizon,
        pv_scaler=pv_scaler,
        weather_scaler=weather_scaler,
        forecast_scaler=forecast_scaler,
        target_scaler=target_scaler
    )
    test_dataset = PVForecastingDataset(
        df.iloc[cutoff_val:],
        pv_lag_features,
        weather_lag_features,
        forecast_features,
        target,
        args.seq_len,
        args.horizon,
        pv_scaler=pv_scaler,
        weather_scaler=weather_scaler,
        forecast_scaler=forecast_scaler,
        target_scaler=target_scaler
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Build model
    model = MultiBranchTransformer(
        n_pv_features=len(pv_lag_features),
        n_hist_weather_features=len(weather_lag_features),
        n_forecast_weather_features=len(forecast_features),
        seq_len_encoder=args.seq_len,
        seq_len_decoder=args.horizon,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    logger.info(f"Model: d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}")

    # Training callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min", verbose=True),
        ModelCheckpoint(dirpath=str(out_dir), filename="multi-branch-best", monitor="val_loss", save_top_k=1, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using accelerator: {accelerator}")
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("Training completed")

    # Load best model
    best_path = callbacks[1].best_model_path  # type: ignore
    if best_path:
        logger.info(f"Loading best model from {best_path}")
        model = MultiBranchTransformer.load_from_checkpoint(best_path)
    else:
        logger.warning("No best checkpoint found, using final model")

    # Generate predictions on TEST set
    logger.info("Generating test predictions...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            features, targets = batch
            # Move features to device
            features_device = {k: v.to(device) for k, v in features.items()}
            preds = model(features_device)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 24)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 24)

    # Denormalize predictions and targets back to original scale
    logger.info("Denormalizing predictions...")
    all_preds = target_scaler.inverse_transform(all_preds)  # (N, 24)
    all_targets = target_scaler.inverse_transform(all_targets)  # (N, 24)

    # Compute metrics per horizon
    logger.info("Computing metrics...")
    metrics = []
    train_series = df.iloc[:cutoff_train][target].values

    for h in range(1, args.horizon + 1):
        y_true = all_targets[:, h-1]
        y_pred = all_preds[:, h-1]

        # Compute naive baseline (24h persistence)
        naive_baseline = np.roll(y_true, 24)
        naive_baseline[:24] = np.nan  # First 24 can't have naive forecast

        # Filter out NaN
        valid_mask = ~np.isnan(naive_baseline)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        naive_valid = naive_baseline[valid_mask]

        if len(y_true_valid) > 0:
            rmse_model = rmse(y_true_valid, y_pred_valid)
            rmse_naive = rmse(y_true_valid, naive_valid)
            mase_model = mase(y_true_valid, y_pred_valid, train_series=train_series, m=24)
            mase_naive = mase(y_true_valid, naive_valid, train_series=train_series, m=24)

            metrics.append({
                "horizon_h": h,
                "rmse_model": float(rmse_model),
                "rmse_naive": float(rmse_naive),
                "mase_model": float(mase_model),
                "mase_naive": float(mase_naive),
            })

    # Summary metrics
    if metrics:
        metric_summary = {
            "rmse_model_avg": float(np.mean([m["rmse_model"] for m in metrics])),
            "rmse_naive_avg": float(np.mean([m["rmse_naive"] for m in metrics])),
            "mase_model_avg": float(np.mean([m["mase_model"] for m in metrics])),
            "mase_naive_avg": float(np.mean([m["mase_naive"] for m in metrics])),
        }
    else:
        metric_summary = {}
        logger.error("No metrics computed")

    # Save results
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "metrics_summary.json").write_text(json.dumps(metric_summary, indent=2))
    logger.info(f"Average RMSE: {metric_summary.get('rmse_model_avg', 0):.4f}")
    logger.info(f"Average MASE: {metric_summary.get('mase_model_avg', 0):.4f}")

    # Save config
    config = {
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "pv_lag_features": pv_lag_features,
        "weather_lag_features": weather_lag_features,
        "forecast_features": forecast_features,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Save scalers for inference
    scalers = {
        "pv_scaler": pv_scaler,
        "weather_scaler": weather_scaler,
        "forecast_scaler": forecast_scaler,
        "target_scaler": target_scaler
    }
    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    logger.info("Saved scalers for inference")

    logger.info("Multi-Branch Transformer training completed successfully")


if __name__ == "__main__":
    main()
