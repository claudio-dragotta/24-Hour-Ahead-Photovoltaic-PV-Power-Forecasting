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
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

from pv_forecasting.data import save_history
from pv_forecasting.metrics import mase, rmse
from pv_forecasting.models import build_cnn_bilstm_fusion_attention
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
    cos_zenith = np.cos(zenith_rad)
    weights = np.maximum(cos_zenith, 0.0) + min_weight
    weights = weights / weights.mean()
    return weights


def add_future_meteo_features(df: pd.DataFrame, horizon: int, future_cols: list) -> pd.DataFrame:
    """Add future meteorological features shifted by forecast horizon.

    Creates features like ghi_fut_h1, temp_fut_h2, etc. for each horizon.
    Simulates perfect NWP forecasts for weather features.

    Args:
        df: DataFrame with weather columns
        horizon: Forecast horizon (typically 24)
        future_cols: List of weather column names to shift

    Returns:
        DataFrame with additional future meteo columns
    """
    available_cols = [col for col in future_cols if col in df.columns]

    shifted_dfs = []
    for h in range(1, horizon + 1):
        shifted = df[available_cols].shift(-h)
        shifted.columns = [f"{col}_fut_h{h}" for col in available_cols]
        shifted_dfs.append(shifted)

    future_df = pd.concat(shifted_dfs, axis=1)
    result = pd.concat([df, future_df], axis=1)

    added_cols = len(available_cols) * horizon
    print(f"Added {added_cols} future meteo features for horizons 1-{horizon}")
    return result


def make_windows_fusion(
    df: pd.DataFrame,
    pv_col: str,
    weather_cols: list,
    future_weather_cols: list,
    seq_len: int,
    horizon: int,
):
    """Create sliding windows for 3-branch fusion model.

    Args:
        df: DataFrame with all features
        pv_col: Name of PV column
        weather_cols: List of historical weather column names
        future_weather_cols: List of future weather column names (e.g., ghi_fut_h1, ...)
        seq_len: Length of lookback window
        horizon: Forecast horizon

    Returns:
        Tuple of (X_pv, X_weather_hist, X_weather_forecast, Y, origins)
    """
    n_samples = len(df) - seq_len - horizon + 1

    # Prepare arrays
    X_pv = np.zeros((n_samples, seq_len, 1))
    X_weather_hist = np.zeros((n_samples, seq_len, len(weather_cols)))
    X_weather_forecast = np.zeros((n_samples, horizon, len(weather_cols)))
    Y = np.zeros((n_samples, horizon))
    origins = []

    for i in range(n_samples):
        # PV history: just the 'pv' column
        X_pv[i] = df[pv_col].values[i : i + seq_len].reshape(-1, 1)

        # Weather history: historical weather features
        X_weather_hist[i] = df[weather_cols].values[i : i + seq_len]

        # Weather forecast: future weather features
        # Extract features for each horizon h (ghi_fut_h1, dni_fut_h1, ..., temp_fut_h1, etc.)
        forecast_start_idx = i + seq_len - 1  # Forecast starts at last historical point
        forecast_features = np.zeros((horizon, len(weather_cols)))

        for h in range(1, horizon + 1):
            for j, base_col in enumerate(weather_cols):
                fut_col = f"{base_col}_fut_h{h}"
                if fut_col in df.columns:
                    forecast_features[h - 1, j] = df[fut_col].iloc[forecast_start_idx]

        X_weather_forecast[i] = forecast_features

        # Target: PV production for next 'horizon' hours
        Y[i] = df[pv_col].values[i + seq_len : i + seq_len + horizon]

        # Origin timestamp
        origins.append(df.index[i + seq_len - 1])

    return X_pv, X_weather_hist, X_weather_forecast, Y, origins


def parse_args():
    ap = argparse.ArgumentParser(description="24h-ahead PV Forecasting with CNN-BiLSTM Multi-Level Fusion + Attention")
    ap.add_argument(
        "--processed-path",
        type=str,
        default="outputs/processed.parquet",
        help="path to pre-processed parquet file (preferred)",
    )
    ap.add_argument("--pv-path", type=str, default="data/raw/pv_dataset.xlsx")
    ap.add_argument("--wx-path", type=str, default="data/raw/wx_dataset.xlsx")
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney")
    ap.add_argument("--seq-len", type=int, default=168, help="lookback sequence length (hours)")
    ap.add_argument("--horizon", type=int, default=24, help="forecast horizon (hours)")
    ap.add_argument("--epochs", type=int, default=200, help="max epochs")
    ap.add_argument("--batch-size", type=int, default=32, help="batch size")
    ap.add_argument("--embedding-dim", type=int, default=128, help="embedding dimension")
    ap.add_argument("--num-heads", type=int, default=8, help="number of attention heads")
    ap.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    ap.add_argument("--outdir", type=str, default="outputs_baseline/cnn")
    ap.add_argument(
        "--no-future-meteo",
        dest="use_future_meteo",
        action="store_false",
        help="Disable future weather features (enabled by default)",
    )
    ap.set_defaults(use_future_meteo=True)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    processed_path = Path(args.processed_path)
    if processed_path.exists():
        print(f"Loading pre-processed data from {processed_path}...")
        df = pd.read_parquet(processed_path)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features from cache")
    else:
        print(f"Processed parquet not found, loading and engineering features from raw data...")
        df = load_and_engineer_features(
            Path(args.pv_path),
            Path(args.wx_path),
            args.local_tz,
            lag_hours=[1, 24, 168],
            rolling_hours=[3, 6],
        )
        persist_processed(df, out_dir.parent)
        print(f"Saved processed data to outputs/processed.parquet")

    # Define base weather columns (before adding future features)
    base_weather_cols = ["ghi", "dni", "dhi", "temp", "humidity", "clouds_all", "wind_speed"]

    # Add future meteo features if enabled
    if args.use_future_meteo:
        print(f"\n{'='*60}")
        print(f"FUTURE METEO MODE ENABLED (simulated NWP)")
        print(f"Adding future weather features: {base_weather_cols}")
        print(f"{'='*60}\n")
        df = add_future_meteo_features(df, args.horizon, base_weather_cols)
        df = df.dropna()
        print(f"After adding future meteo: {len(df)} samples, {len(df.columns)} features")
    else:
        print(f"\nStandard mode: no future meteo features")

    # Identify column groups for the 3 branches
    pv_col = "pv"

    # Historical weather columns (only base features, no future ones)
    weather_hist_cols = [
        col
        for col in df.columns
        if col in base_weather_cols or any(col.startswith(f"{bc}_") for bc in base_weather_cols if not "fut" in col)
    ]
    # Filter to only existing base weather cols
    weather_hist_cols = [col for col in base_weather_cols if col in df.columns]

    # Future weather columns (ghi_fut_h1, ..., ghi_fut_h24, etc.)
    future_weather_cols = [col for col in df.columns if "_fut_h" in col]

    print(f"\n{'='*60}")
    print(f"MULTI-LEVEL FUSION ARCHITECTURE")
    print(f"Branch 1 (PV history): {pv_col}")
    print(f"Branch 2 (Weather history): {len(weather_hist_cols)} features")
    print(f"Branch 3 (Weather forecast): {len(future_weather_cols)} features")
    print(f"{'='*60}\n")

    # Create windowed data for 3 branches
    X_pv, X_weather_hist, X_weather_forecast, Y, origins = make_windows_fusion(
        df,
        pv_col=pv_col,
        weather_cols=weather_hist_cols,
        future_weather_cols=future_weather_cols,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )

    # Chronological split
    n_samples = len(Y)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    X_pv_train, X_pv_val, X_pv_test = (
        X_pv[:train_end],
        X_pv[train_end:val_end],
        X_pv[val_end:],
    )
    X_wh_train, X_wh_val, X_wh_test = (
        X_weather_hist[:train_end],
        X_weather_hist[train_end:val_end],
        X_weather_hist[val_end:],
    )
    X_wf_train, X_wf_val, X_wf_test = (
        X_weather_forecast[:train_end],
        X_weather_forecast[train_end:val_end],
        X_weather_forecast[val_end:],
    )
    Y_train, Y_val, Y_test = Y[:train_end], Y[train_end:val_end], Y[val_end:]
    origins_train = origins[:train_end]
    origins_val = origins[train_end:val_end]
    origins_test = origins[val_end:]

    # Scale features (fit on train only)
    # Scale PV history
    pv_scaler = StandardScaler()
    X_pv_train_scaled = pv_scaler.fit_transform(X_pv_train.reshape(-1, 1)).reshape(X_pv_train.shape)
    X_pv_val_scaled = pv_scaler.transform(X_pv_val.reshape(-1, 1)).reshape(X_pv_val.shape)
    X_pv_test_scaled = pv_scaler.transform(X_pv_test.reshape(-1, 1)).reshape(X_pv_test.shape)

    # Scale weather history
    wh_scaler = StandardScaler()
    X_wh_train_scaled = wh_scaler.fit_transform(X_wh_train.reshape(-1, X_wh_train.shape[-1])).reshape(X_wh_train.shape)
    X_wh_val_scaled = wh_scaler.transform(X_wh_val.reshape(-1, X_wh_val.shape[-1])).reshape(X_wh_val.shape)
    X_wh_test_scaled = wh_scaler.transform(X_wh_test.reshape(-1, X_wh_test.shape[-1])).reshape(X_wh_test.shape)

    # Scale weather forecast
    wf_scaler = StandardScaler()
    X_wf_train_scaled = wf_scaler.fit_transform(X_wf_train.reshape(-1, X_wf_train.shape[-1])).reshape(X_wf_train.shape)
    X_wf_val_scaled = wf_scaler.transform(X_wf_val.reshape(-1, X_wf_val.shape[-1])).reshape(X_wf_val.shape)
    X_wf_test_scaled = wf_scaler.transform(X_wf_test.reshape(-1, X_wf_test.shape[-1])).reshape(X_wf_test.shape)

    # Compute solar weights
    if "sp_zenith" in df.columns:
        zenith_map = df.set_index(df.index)["sp_zenith"]
        train_zenith = pd.Series([zenith_map.loc[ts] for ts in origins_train])
        val_zenith = pd.Series([zenith_map.loc[ts] for ts in origins_val])

        train_weights = compute_solar_weights(train_zenith)
        val_weights = compute_solar_weights(val_zenith)

        print(f"\n{'='*60}")
        print(f"Training with solar-weighted samples")
        print(f"Train weights: min={train_weights.min():.3f}, max={train_weights.max():.3f}")
        print(f"Val weights: min={val_weights.min():.3f}, max={val_weights.max():.3f}")
        print(f"{'='*60}\n")
    else:
        train_weights = None
        val_weights = None

    # Build model
    model = build_cnn_bilstm_fusion_attention(
        pv_seq_len=args.seq_len,
        weather_seq_len=args.seq_len,
        weather_forecast_len=args.horizon,
        weather_features=len(weather_hist_cols),
        horizon=args.horizon,
        embedding_dim=args.embedding_dim,
        num_attention_heads=args.num_heads,
        dropout_rate=args.dropout,
    )

    print(f"\n{'='*60}")
    print(f"MODEL: CNN-BiLSTM with Multi-Level Fusion + Attention")
    print(f"{'='*60}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Architecture:")
    print(f"  - Branch 1: PV history ({args.seq_len} × 1)")
    print(f"  - Branch 2: Weather history ({args.seq_len} × {len(weather_hist_cols)})")
    print(f"  - Branch 3: Weather forecast ({args.horizon} × {len(weather_hist_cols)})")
    print(f"  - Embedding dim: {args.embedding_dim}")
    print(f"  - Attention heads: {args.num_heads}")
    print(f"  - Dropout: {args.dropout}")
    print(f"Output: {args.horizon} horizons")
    print(f"Epochs: {args.epochs} (early stopping patience=20)")
    print(f"Batch size: {args.batch_size}\n")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=3, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "model_best.keras"), save_best_only=True, monitor="val_loss"
        ),
    ]

    # Prepare validation data
    if val_weights is not None:
        validation_data = ([X_pv_val_scaled, X_wh_val_scaled, X_wf_val_scaled], Y_val, val_weights)
    else:
        validation_data = ([X_pv_val_scaled, X_wh_val_scaled, X_wf_val_scaled], Y_val)

    # Train
    history = model.fit(
        [X_pv_train_scaled, X_wh_train_scaled, X_wf_train_scaled],
        Y_train,
        validation_data=validation_data,
        sample_weight=train_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    # Save scalers and history
    dump({"pv": pv_scaler, "weather_hist": wh_scaler, "weather_forecast": wf_scaler}, out_dir / "scalers.joblib")
    save_history(history.history, out_dir)

    # Validation predictions
    Yhat_val = model.predict([X_pv_val_scaled, X_wh_val_scaled, X_wf_val_scaled], verbose=0)
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
                    "y_true": float(Y_val[i, h]),
                    "y_pred": float(Yhat_val[i, h]),
                }
            )
    val_pred_df = pd.DataFrame(val_rows)
    val_pred_df.to_csv(out_dir / "predictions_val_cnn_fusion.csv", index=False)
    print(f"Saved {len(val_pred_df)} validation predictions")

    # Test predictions
    Yhat_test = model.predict([X_pv_test_scaled, X_wh_test_scaled, X_wf_test_scaled], verbose=0)

    # Naive baseline: PV(t) = PV(t-24)
    naive = X_pv_test[:, -args.horizon :, 0]

    # Metrics
    y_true_flat = Y_test.reshape(-1)
    y_pred_flat = Yhat_test.reshape(-1)
    y_naive_flat = naive.reshape(-1)

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

    # Export test predictions
    test_rows = []
    for i in range(len(origins_test)):
        origin = pd.Timestamp(origins_test[i]).tz_convert("UTC")
        base_ts = origin + pd.Timedelta(hours=1)
        for h in range(args.horizon):
            test_rows.append(
                {
                    "origin_timestamp_utc": origin.isoformat(),
                    "forecast_timestamp_utc": (base_ts + pd.Timedelta(hours=h)).isoformat(),
                    "horizon_h": h + 1,
                    "y_true": float(Y_test[i, h]),
                    "y_pred": float(Yhat_test[i, h]),
                }
            )
    test_pred_df = pd.DataFrame(test_rows)
    test_pred_df.to_csv(out_dir / "predictions_test_cnn_fusion.csv", index=False)
    print(f"Saved {len(test_pred_df)} test predictions")


if __name__ == "__main__":
    main()
