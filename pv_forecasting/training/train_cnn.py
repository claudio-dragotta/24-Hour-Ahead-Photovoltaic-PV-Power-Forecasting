"""CNN-BiLSTM training module.

Provides train_cnn_bilstm() function that can be called from CLI or imported.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..data import save_history
from ..logger import get_logger
from ..metrics import mase, rmse
from ..models import build_cnn_bilstm
from ..pipeline import load_and_engineer_features, persist_processed
from ..window import chronological_split, make_windows

logger = get_logger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_cnn_bilstm(config: Config) -> None:
    """Train CNN-BiLSTM model using provided configuration.

    Args:
        config: Configuration object with all hyperparameters.
    """
    set_seed(config.model.seed)

    out_dir = config.output.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training CNN-BiLSTM model")
    logger.info(f"Horizon: {config.model.horizon}h, Seq length: {config.model.seq_len}h")
    logger.info(f"Output directory: {out_dir}")

    # Feature engineering (standardized naming, physics features, lag/rolling)
    logger.info("Loading and engineering features...")
    df = load_and_engineer_features(
        config.data.pv_path,
        config.data.wx_path,
        config.data.local_tz,
        lag_hours=config.data.lag_hours,
        rolling_hours=config.data.rolling_hours,
        include_solar=config.data.include_solar,
        include_clearsky=config.data.include_clearsky,
        dropna=config.data.dropna,
    )
    logger.info(f"Data shape after feature engineering: {df.shape}")

    # Save processed
    if config.output.save_predictions:
        persist_processed(df, out_dir)
        logger.info(f"Processed data saved to {out_dir / 'processed.parquet'}")

    # Scale features (fit on train only)
    feature_cols = [c for c in df.columns if c not in {"pv", "time_idx", "series_id"}]
    logger.info(f"Number of features: {len(feature_cols)}")

    # Window first on unscaled to compute MASE denominator later on the training target series
    logger.info("Creating sliding windows...")
    X_raw, Y_raw, origins = make_windows(
        df[[*feature_cols, "pv"]],
        target_col="pv",
        seq_len=config.model.seq_len,
        horizon=config.model.horizon,
    )
    logger.info(f"Windows created: X shape={X_raw.shape}, Y shape={Y_raw.shape}")

    # Chronological split indexes
    logger.info("Splitting data chronologically...")
    (Xtr_raw, Ytr, origins_tr), (Xval_raw, Yval, origins_val), (Xte_raw, Yte, origins_te) = chronological_split(
        X_raw, Y_raw, np.array(origins), train_ratio=config.model.train_ratio, val_ratio=config.model.val_ratio
    )
    logger.info(f"Train: {Xtr_raw.shape[0]}, Val: {Xval_raw.shape[0]}, Test: {Xte_raw.shape[0]} samples")

    # Fit scaler on train portion features only
    logger.info("Fitting StandardScaler on training data...")
    n_features = len(feature_cols) + 1  # including pv as last column
    scaler = StandardScaler()
    # Extract all feature rows used by train windows (excluding pv column)
    Xtr_feats = Xtr_raw[:, :, : len(feature_cols)].reshape(-1, len(feature_cols))
    scaler.fit(Xtr_feats)

    # Apply scaler to all splits
    def apply_scale(X):
        """Apply StandardScaler to feature columns only (not target)."""
        Xf = X.copy()
        n = Xf.shape[0]
        X_feats = Xf[:, :, : len(feature_cols)].reshape(-1, len(feature_cols))
        X_feats = scaler.transform(X_feats)
        Xf[:, :, : len(feature_cols)] = X_feats.reshape(n, Xf.shape[1], len(feature_cols))
        return Xf

    Xtr = apply_scale(Xtr_raw)
    Xval = apply_scale(Xval_raw)
    Xte = apply_scale(Xte_raw)
    logger.info("Data scaling completed")

    # Build and train model
    logger.info("Building CNN-BiLSTM model...")
    model = build_cnn_bilstm(input_shape=(config.model.seq_len, n_features), horizon=config.model.horizon)
    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config.model.early_stopping_patience,
            restore_best_weights=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "model_best.keras"), save_best_only=True, monitor="val_loss"
        ),
    ]

    logger.info(f"Starting training for {config.model.epochs} epochs...")
    history = model.fit(
        Xtr,
        Ytr,
        validation_data=(Xval, Yval),
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        verbose=2,
        callbacks=callbacks,
    )
    logger.info("Training completed!")

    # Save scaler and history
    if config.output.save_model:
        dump(scaler, out_dir / "scalers.joblib")
        logger.info(f"Scaler saved to {out_dir / 'scalers.joblib'}")

    if config.output.save_history:
        save_history(history.history, out_dir)
        logger.info(f"Training history saved to {out_dir / 'history.json'}")

    # Predictions and metrics
    logger.info("Generating test predictions...")
    Yhat = model.predict(Xte, verbose=0)
    logger.info(f"Predictions shape: {Yhat.shape}")

    # Naive baseline: PV(t) = PV(t-24)
    # In sliding windows, the last 24 PV values of the input correspond to previous day
    naive = Xte_raw[:, -config.model.horizon :, -1]

    # Compute metrics (flattened across horizons)
    logger.info("Computing evaluation metrics...")
    y_true_flat = Yte.reshape(-1)
    y_pred_flat = Yhat.reshape(-1)
    y_naive_flat = naive.reshape(-1)

    # MASE denominator from training target series (not windowed)
    train_cutoff = int(len(df) * config.model.train_ratio)
    train_series = df["pv"].values[:train_cutoff]

    rmse_model = rmse(y_true_flat, y_pred_flat)
    rmse_naive = rmse(y_true_flat, y_naive_flat)
    mase_model = mase(y_true_flat, y_pred_flat, train_series=train_series, m=24)
    mase_naive = mase(y_true_flat, y_naive_flat, train_series=train_series, m=24)

    # Log metrics
    logger.info("=" * 60)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 60)
    logger.info(f"Model RMSE:    {rmse_model:.6f}")
    logger.info(f"Naive RMSE:    {rmse_naive:.6f}")
    logger.info(f"Model MASE:    {mase_model:.6f}")
    logger.info(f"Naive MASE:    {mase_naive:.6f}")
    logger.info(f"RMSE Improvement: {(1 - rmse_model/rmse_naive)*100:.2f}%")
    logger.info(f"MASE Improvement: {(1 - mase_model/mase_naive)*100:.2f}%")
    logger.info("=" * 60)

    # Save metrics to JSON
    import json

    metrics = {
        "model": {
            "rmse": float(rmse_model),
            "mase": float(mase_model),
        },
        "naive_baseline": {
            "rmse": float(rmse_naive),
            "mase": float(mase_naive),
        },
        "improvement": {
            "rmse_percent": float((1 - rmse_model / rmse_naive) * 100),
            "mase_percent": float((1 - mase_model / mase_naive) * 100),
        },
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {out_dir / 'metrics.json'}")

    # Export predictions (long format)
    if config.output.save_predictions:
        logger.info("Exporting predictions...")
        rows = []
        for i in range(len(origins_te)):
            origin = pd.Timestamp(origins_te[i]).tz_convert("UTC")
            base_ts = origin + pd.Timedelta(hours=1)
            for h in range(config.model.horizon):
                rows.append(
                    {
                        "origin_timestamp_utc": origin.isoformat(),
                        "forecast_timestamp_utc": (base_ts + pd.Timedelta(hours=h)).isoformat(),
                        "horizon_h": h + 1,
                        "y_true": float(Yte[i, h]),
                        "y_pred": float(Yhat[i, h]),
                    }
                )
        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(out_dir / "predictions_test.csv", index=False)
        logger.info(f"Predictions exported to {out_dir / 'predictions_test.csv'}")

    logger.info("Training pipeline completed successfully!")
    logger.info(f"All outputs saved to: {out_dir}")
