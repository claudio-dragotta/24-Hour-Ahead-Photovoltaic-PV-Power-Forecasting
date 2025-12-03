from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from pv_forecasting.data import save_history
from pv_forecasting.pipeline import load_and_engineer_features, persist_processed
from pv_forecasting.window import make_windows, chronological_split
from pv_forecasting.model import build_cnn_bilstm
from pv_forecasting.metrics import rmse, mase


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args():
    ap = argparse.ArgumentParser(description="24h-ahead PV Forecasting")
    ap.add_argument("--pv-path", type=str, default="data/raw/pv_dataset.xlsx")
    ap.add_argument("--wx-path", type=str, default="data/raw/wx_dataset.xlsx")
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney")
    ap.add_argument("--seq-len", type=int, default=168)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", type=str, default="outputs")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Scale features (fit on train only)
    feature_cols = [c for c in df.columns if c not in {"pv", "time_idx", "series_id"}]
    full_feature_df = df[feature_cols]
    target_series = df["pv"]

    # Window first on unscaled to compute MASE denominator later on the training target series
    X_raw, Y_raw, origins = make_windows(df[[*feature_cols, "pv"]], target_col="pv", seq_len=args.seq_len, horizon=args.horizon)

    # Chronological split indexes
    (Xtr_raw, Ytr, origins_tr), (Xval_raw, Yval, origins_val), (Xte_raw, Yte, origins_te) = chronological_split(X_raw, Y_raw, np.array(origins))

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

    # Build and train model
    model = build_cnn_bilstm(input_shape=(args.seq_len, n_features), horizon=args.horizon)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(out_dir / "model_best.keras"), save_best_only=True, monitor="val_loss"),
    ]
    history = model.fit(
        Xtr, Ytr,
        validation_data=(Xval, Yval),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    # Save scaler and history
    dump(scaler, out_dir / "scalers.joblib")
    save_history(history.history, out_dir)

    # Predictions and metrics
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

    print({
        "rmse_model": rmse_model,
        "rmse_naive": rmse_naive,
        "mase_model": mase_model,
        "mase_naive": mase_naive,
    })

    # Export predictions (long format)
    rows = []
    for i in range(len(origins_te)):
        origin = pd.Timestamp(origins_te[i]).tz_convert("UTC")
        base_ts = origin + pd.Timedelta(hours=1)
        for h in range(args.horizon):
            rows.append({
                "origin_timestamp_utc": origin.isoformat(),
                "forecast_timestamp_utc": (base_ts + pd.Timedelta(hours=h)).isoformat(),
                "horizon_h": h + 1,
                "y_true": float(Yte[i, h]),
                "y_pred": float(Yhat[i, h]),
            })
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / "predictions_test.csv", index=False)


if __name__ == "__main__":
    main()
