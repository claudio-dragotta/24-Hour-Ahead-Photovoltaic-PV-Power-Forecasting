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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

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
        target_col: str = "pv",
        seq_len: int = 168,
        horizon: int = 24,
        sample_weights: Optional[np.ndarray] = None,
        pv_scaler: Optional[StandardScaler] = None,
        weather_scaler: Optional[StandardScaler] = None,
        forecast_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
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
        pv_history = self.df.loc[start_idx : end_idx - 1, self.pv_lag_features].values.astype(np.float32)
        weather_history = self.df.loc[start_idx : end_idx - 1, self.weather_lag_features].values.astype(np.float32)
        weather_forecast = self.df.loc[target_start : target_end - 1, self.forecast_features].values.astype(np.float32)

        # Extract targets
        targets = self.df.loc[target_start : target_end - 1, self.target_col].values.astype(np.float32)

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
            "pv_history": torch.from_numpy(pv_history.astype(np.float32)),
            "weather_history": torch.from_numpy(weather_history.astype(np.float32)),
            "weather_forecast": torch.from_numpy(weather_forecast.astype(np.float32)),
        }
        targets = torch.from_numpy(targets.astype(np.float32))

        return features, targets


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Train Multi-Branch Transformer for 24h-ahead PV forecasting")
    ap.add_argument(
        "--processed-path", type=str, default="/home/claudio/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/data/processed/merged/pv_wx_combined.parquet", help="Path to pre-processed parquet file"
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
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of train data reserved for validation (chronological last fraction)")
    return ap.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()
    seed_everything(2)
    logger.info("Starting Multi-Branch Transformer training with seed=2 (fixed)")

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

    # Base forecast features
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
    # Add one-hot weather columns (wx_*)
    wx_features = [c for c in df.columns if c.startswith("wx_")]
    forecast_features = base_forecast + wx_features

    logger.info(f"PV lag features ({len(pv_lag_features)}): {pv_lag_features[:5]}...")
    logger.info(f"Weather lag features ({len(weather_lag_features)}): {weather_lag_features[:5]}...")
    logger.info(f"Forecast features ({len(forecast_features)}): {forecast_features[:5]}...")
    if wx_features:
        logger.info(f"  Including {len(wx_features)} weather one-hot features")


    # Solo train: usa tutto il dataset per il training
    from sklearn.preprocessing import MinMaxScaler
    pv_scaler = StandardScaler()
    weather_scaler = StandardScaler()
    forecast_scaler_cont = MinMaxScaler()
    target_scaler = MinMaxScaler()

    pv_scaler.fit(df[pv_lag_features].values)
    weather_scaler.fit(df[weather_lag_features].values)
    forecast_scaler_cont.fit(df[base_forecast].values)
    target_scaler.fit(df[target].values.reshape(-1, 1))

    logger.info(f"Target scaling: min={target_scaler.data_min_[0]:.2f}, max={target_scaler.data_max_[0]:.2f}")

    full_dataset = PVForecastingDataset(
        df,
        pv_lag_features,
        weather_lag_features,
        forecast_features,
        target,
        args.seq_len,
        args.horizon,
        sample_weights if sample_weights is not None else None,
        pv_scaler=pv_scaler,
        weather_scaler=weather_scaler,
        forecast_scaler=forecast_scaler_cont,
        target_scaler=target_scaler,
    )
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    if n_val > 0:
        train_indices = list(range(0, n_train))
        val_indices = list(range(n_train, n_total))
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        logger.info(f"Dataset sizes -> train: {len(train_dataset)}, val: {len(val_dataset)} (val_ratio={args.val_ratio})")
    else:
        train_dataset = full_dataset
        val_dataset = None
        logger.info("Dataset sizes -> train only (val_ratio=0)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_dataset else None
    logger.info(f"Train batches: {len(train_loader)}; Val batches: {len(val_loader) if val_loader else 0}")


    # Build model
    model = MultiBranchTransformer(
        n_pv_features=len(pv_lag_features),
        n_hist_weather_features=len(weather_lag_features),
        n_forecast_weather_features=len(forecast_features),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        seq_len_encoder=args.seq_len,
        seq_len_decoder=args.horizon,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        default_root_dir=out_dir,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    # Train su train e (opzionale) val
    logger.info("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader if val_loader else None)
    logger.info("Training completed")


    # Save scalers for inference
    scalers = {
        "pv_scaler": pv_scaler,
        "weather_scaler": weather_scaler,
        "forecast_scaler_cont": forecast_scaler_cont,
        "target_scaler": target_scaler,
    }
    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    logger.info("Saved scalers for inference")


    # Log per data leakage: controlla se target è tra le feature di input
    input_features = set(pv_lag_features + weather_lag_features + forecast_features)
    if target in input_features:
        logger.warning(f"POSSIBILE DATA LEAKAGE: la colonna target '{target}' è tra le feature di input!")
    else:
        logger.info("Nessun data leakage rilevato tra le feature di input.")
    logger.info("Multi-Branch Transformer training completed successfully")


if __name__ == "__main__":
    main()
