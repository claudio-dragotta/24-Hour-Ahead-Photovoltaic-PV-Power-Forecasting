from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump
from sklearn.preprocessing import StandardScaler

# Enable mixed precision training for 2-3x faster GPU computation
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

from pv_forecasting.data import save_history
from pv_forecasting.metrics import mase, rmse
from pv_forecasting.model import build_cnn_bilstm
from pv_forecasting.pipeline import load_and_engineer_features, persist_processed
from pv_forecasting.window import chronological_split, make_windows


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_solar_weights(zenith_deg: pd.Series, min_weight: float = 0.1) -> np.ndarray:
    """Compute sample weights based on solar zenith angle.

    Higher weights during daytime (low zenith) and lower weights at night (high zenith).
    Uses cosine of zenith angle as it represents solar irradiance intensity.

    Args:
        zenith_deg: Solar zenith angle in degrees (0° = sun overhead, 90° = horizon)
        min_weight: Minimum weight to assign (prevents zero weights at night)

    Returns:
        Array of sample weights normalized to mean=1.0
    """
    zenith_rad = np.deg2rad(zenith_deg.values)
    # cos(zenith) ranges from 1 (sun overhead) to 0 (horizon) to negative (night)
    cos_zenith = np.cos(zenith_rad)
    # Clip to [0, 1] and add minimum weight
    weights = np.maximum(cos_zenith, 0.0) + min_weight
    # Normalize to mean=1.0 to preserve overall scale
    weights = weights / weights.mean()
    return weights


def parse_args():
    ap = argparse.ArgumentParser(description="24h-ahead PV Forecasting with CNN-BiLSTM")
    ap.add_argument(
        "--processed-path",
        type=str,
        default="outputs/processed.parquet",
        help="path to pre-processed parquet file (preferred)",
    )
    ap.add_argument("--pv-path", type=str, default="data/raw/pv_dataset.xlsx")
    ap.add_argument("--wx-path", type=str, default="data/raw/wx_dataset.xlsx")
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney")
    ap.add_argument("--seq-len", type=int, default=168)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--outdir", type=str, default="outputs_cnn")
    ap.add_argument(
        "--use-future-meteo",
        action="store_true",
        help="Use future weather data as features (simulated NWP mode)",
    )
    return ap.parse_args()


def add_future_meteo_features(df: pd.DataFrame, horizon: int, future_cols: list) -> pd.DataFrame:
    """Add future meteorological features shifted by forecast horizon.
    
    For each horizon h (1 to 24), creates features like ghi_fut_h1, temp_fut_h2, etc.
    These represent the actual weather values at the forecast time, simulating
    perfect NWP forecasts as done in MATNet paper.
    
    Args:
        df: DataFrame with weather columns
        horizon: Forecast horizon (typically 24)
        future_cols: List of weather column names to shift
        
    Returns:
        DataFrame with additional future meteo columns
    """
    data = df.copy()
    added_cols = []
    for h in range(1, horizon + 1):
        for col in future_cols:
            if col in data.columns:
                new_col = f"{col}_fut_h{h}"
                data[new_col] = data[col].shift(-h)
                added_cols.append(new_col)
    print(f"Added {len(added_cols)} future meteo features for horizons 1-{horizon}")
    return data


def main():
    args = parse_args()
    set_seed(42)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data: prefer processed parquet for speed
    processed_path = Path(args.processed_path)
    if processed_path.exists():
        print(f"Loading pre-processed data from {processed_path}...")
        df = pd.read_parquet(processed_path)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features from cache")
    else:
        print(f"Processed parquet not found, loading and engineering features from raw data...")
        # Feature engineering (standardized naming, physics features, lag/rolling)
        df = load_and_engineer_features(
            Path(args.pv_path),
            Path(args.wx_path),
            args.local_tz,
            lag_hours=[1, 24, 168],
            rolling_hours=[3, 6],
        )
        # Save processed
        persist_processed(df, out_dir)
        print(f"Saved processed data to outputs/processed.parquet")

    # Add future meteo features if requested (simulated NWP mode)
    if args.use_future_meteo:
        future_cols = ["ghi", "dni", "dhi", "temp", "humidity", "clouds_all", "wind_speed"]
        print(f"\n{'='*60}")
        print(f"FUTURE METEO MODE ENABLED (simulated NWP)")
        print(f"Adding future weather features: {future_cols}")
        print(f"{'='*60}\n")
        df = add_future_meteo_features(df, args.horizon, future_cols)
        # Drop rows with NaN from the future shift
        df = df.dropna()
        print(f"After adding future meteo: {len(df)} samples, {len(df.columns)} features")
    else:
        print(f"\nStandard mode: no future meteo features")

    # Scale features (fit on train only)
    feature_cols = [c for c in df.columns if c not in {"pv", "time_idx", "series_id"}]
    full_feature_df = df[feature_cols]
    target_series = df["pv"]

    # Window first on unscaled to compute MASE denominator later on the training target series
    X_raw, Y_raw, origins = make_windows(
        df[[*feature_cols, "pv"]], target_col="pv", seq_len=args.seq_len, horizon=args.horizon
    )

    # Chronological split indexes
    (Xtr_raw, Ytr, origins_tr), (Xval_raw, Yval, origins_val), (Xte_raw, Yte, origins_te) = chronological_split(
        X_raw, Y_raw, np.array(origins)
    )

    # Fit scaler on train portion features only
    # Recover the aligned feature frames for the train window span
    n_features = len(feature_cols) + 1  # including pv as last column
    # Prepare a feature scaler for all features except the last column (pv)
    scaler = StandardScaler()
    # To fit scaler, extract all feature rows used by train windows
    # Using the fact that Xtr_raw contains feature+pv; drop last column
    Xtr_feats = Xtr_raw[:, :, : len(feature_cols)].reshape(-1, len(feature_cols))
    scaler.fit(Xtr_feats)

    # Apply scaler to all splits
    def apply_scale(X):
        Xf = X.copy()
        n = Xf.shape[0]
        X_feats = Xf[:, :, : len(feature_cols)].reshape(-1, len(feature_cols))
        X_feats = scaler.transform(X_feats)
        Xf[:, :, : len(feature_cols)] = X_feats.reshape(n, Xf.shape[1], len(feature_cols))
        return Xf

    Xtr = apply_scale(Xtr_raw)
    Xval = apply_scale(Xval_raw)
    Xte = apply_scale(Xte_raw)

    # Compute solar-based sample weights for train and val sets
    if "sp_zenith" in df.columns:
        # Create a mapping from timestamp to sp_zenith
        zenith_map = df.set_index(df.index)["sp_zenith"]

        # Extract zenith values for each window origin
        train_zenith = pd.Series([zenith_map.loc[ts] for ts in origins_tr], index=range(len(origins_tr)))
        val_zenith = pd.Series([zenith_map.loc[ts] for ts in origins_val], index=range(len(origins_val)))

        train_weights = compute_solar_weights(train_zenith)
        val_weights = compute_solar_weights(val_zenith)

        print(f"\n{'='*60}")
        print(f"Training CNN-BiLSTM with solar-weighted samples")
        print(f"Train weights: min={train_weights.min():.3f}, max={train_weights.max():.3f}, mean={train_weights.mean():.3f}")
        print(f"Val weights: min={val_weights.min():.3f}, max={val_weights.max():.3f}, mean={val_weights.mean():.3f}")
        print(f"{'='*60}\n")
    else:
        train_weights = None
        val_weights = None
        print(f"\n{'='*60}")
        print(f"Training CNN-BiLSTM (no solar weights)")
        print(f"{'='*60}\n")

    # Build and train model
    model = build_cnn_bilstm(input_shape=(args.seq_len, n_features), horizon=args.horizon)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=3, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "model_best.keras"), save_best_only=True, monitor="val_loss"
        ),
    ]

    print(f"Training CNN-BiLSTM model...")
    print(f"Model parameters: {model.count_params():,}")
    print(f"Input shape: {args.seq_len} timesteps × {n_features} features")
    print(f"Output: {args.horizon} horizons")
    print(f"Epochs: {args.epochs} (with early stopping patience={15})")
    print(f"Batch size: {args.batch_size}")
    print()

    # Prepare validation_data with sample weights if available
    if val_weights is not None:
        validation_data = (Xval, Yval, val_weights)
    else:
        validation_data = (Xval, Yval)

    history = model.fit(
        Xtr,
        Ytr,
        validation_data=validation_data,
        sample_weight=train_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    # Save scaler and history
    dump(scaler, out_dir / "scalers.joblib")
    save_history(history.history, out_dir)

    # Predictions on VALIDATION set (for ensemble weight optimization - no data leakage)
    Yhat_val = model.predict(Xval, verbose=0)
    val_rows = []
    for i in range(len(origins_val)):
        origin = pd.Timestamp(origins_val[i]).tz_convert("UTC")
        base_ts = origin + pd.Timedelta(hours=1)
        for h in range(args.horizon):
            val_rows.append(
                {
                    "origin_timestamp_utc": origin.isoformat(),
                    "forecast_timestamp_utc": (base_ts + pd.Timedelta(hours=h)).isoformat(),
                    "horizon_h": h + 1,
                    "y_true": float(Yval[i, h]),
                    "y_pred": float(Yhat_val[i, h]),
                }
            )
    val_pred_df = pd.DataFrame(val_rows)
    val_pred_df.to_csv(out_dir / "predictions_val_cnn.csv", index=False)
    print(f"✓ Saved {len(val_pred_df)} validation predictions")

    # Predictions on TEST set (for final evaluation only)
    Yhat = model.predict(Xte, verbose=0)

    # Naive baseline: PV(t) = PV(t-24). In sliding windows, the last 24 PV values of the input correspond to previous day.
    # Extract naive from the raw Xte_raw last 24 time steps of the PV column (last feature column)
    naive = Xte_raw[:, -args.horizon :, -1]

    # Compute metrics (flattened across horizons)
    y_true_flat = Yte.reshape(-1)
    y_pred_flat = Yhat.reshape(-1)
    y_naive_flat = naive.reshape(-1)

    # MASE denominator from training target series (not windowed): use first 70% of the PV series
    train_cutoff = int(len(df) * 0.7)
    train_series = df["pv"].values[:train_cutoff]
    rmse_model = rmse(y_true_flat, y_pred_flat)
    rmse_naive = rmse(y_true_flat, y_naive_flat)
    mase_model = mase(y_true_flat, y_pred_flat, train_series=train_series, m=24)
    mase_naive = mase(y_true_flat, y_naive_flat, train_series=train_series, m=24)

    print(
        {
            "rmse_model": rmse_model,
            "rmse_naive": rmse_naive,
            "mase_model": mase_model,
            "mase_naive": mase_naive,
        }
    )

    # Export TEST predictions (long format)
    rows = []
    for i in range(len(origins_te)):
        origin = pd.Timestamp(origins_te[i]).tz_convert("UTC")
        base_ts = origin + pd.Timedelta(hours=1)
        for h in range(args.horizon):
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
    pred_df.to_csv(out_dir / "predictions_test_cnn.csv", index=False)
    print(f"✓ Saved {len(pred_df)} test predictions")


if __name__ == "__main__":
    main()
