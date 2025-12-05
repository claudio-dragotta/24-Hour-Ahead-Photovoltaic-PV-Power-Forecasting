"""Temporal Fusion Transformer (TFT) training script for 24h-ahead PV forecasting.

This module implements state-of-the-art TFT training using PyTorch Forecasting,
supporting both scenarios with and without future meteorological data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pv_forecasting.logger import get_logger
from pv_forecasting.metrics import mase, rmse
from pv_forecasting.pipeline import load_and_engineer_features, persist_processed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for TFT training.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description="train tft (pytorch forecasting) for 24h-ahead pv forecasting")
    ap.add_argument(
        "--processed-path",
        type=str,
        default="outputs/processed.parquet",
        help="path to pre-processed parquet file (preferred)",
    )
    ap.add_argument("--pv-path", type=str, default="data/raw/pv_dataset.xlsx", help="fallback: path to PV dataset xlsx")
    ap.add_argument(
        "--wx-path", type=str, default="data/raw/wx_dataset.xlsx", help="fallback: path to weather dataset xlsx"
    )
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney", help="local timezone for data")
    ap.add_argument("--seq-len", type=int, default=168, help="encoder length in hours (default: 168 = 7 days)")
    ap.add_argument("--horizon", type=int, default=24, help="prediction horizon in hours")
    ap.add_argument("--epochs", type=int, default=100, help="maximum training epochs")
    ap.add_argument("--batch-size", type=int, default=64, help="training batch size")
    ap.add_argument("--hidden-size", type=int, default=64, help="TFT hidden size")
    ap.add_argument("--attention-heads", type=int, default=4, help="number of attention heads")
    ap.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    ap.add_argument("--learning-rate", type=float, default=1e-3, help="initial learning rate")
    ap.add_argument(
        "--outdir", type=str, default="outputs_tft", help="output directory for checkpoints and predictions"
    )
    ap.add_argument(
        "--use-future-meteo", action="store_true", help="use future weather covariates if available (NWP mode)"
    )
    ap.add_argument("--train-ratio", type=float, default=0.6, help="training set ratio (chronological split)")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="validation set ratio (chronological split)")
    ap.add_argument("--test-ratio", type=float, default=0.2, help="test set ratio (chronological split)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    return ap.parse_args()


def main() -> None:
    """Main training function for TFT model."""
    args = parse_args()
    seed_everything(args.seed)
    logger.info(f"starting tft training with seed={args.seed}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"output directory: {out_dir}")

    # Load data: prefer processed parquet for speed
    processed_path = Path(args.processed_path)
    if processed_path.exists():
        logger.info(f"loading pre-processed data from {processed_path}")
        df = pd.read_parquet(processed_path)
        logger.info(f"loaded {len(df)} samples with {len(df.columns)} features from cache")
    else:
        logger.info(f"processed parquet not found, loading and engineering features from raw data")
        df = load_and_engineer_features(Path(args.pv_path), Path(args.wx_path), args.local_tz)
        persist_processed(df, Path("outputs"))
        logger.info(f"saved processed data to outputs/processed.parquet")

    target = "pv"
    logger.info(f"target variable: {target}")

    # Known future covariates: time features, solar position, clear-sky estimates
    known_future = [
        c
        for c in [
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "sp_zenith",
            "sp_azimuth",
            "cs_ghi",
            "cs_dni",
            "cs_dhi",
        ]
        if c in df.columns
    ]

    raw_meteo = [c for c in ["ghi", "dni", "dhi", "temp", "humidity", "clouds", "wind_speed"] if c in df.columns]

    # Past unknown covariates: lags, rolling features, clearness index
    past_unknown = [
        c
        for c in df.columns
        if c.startswith(("pv_lag", "ghi_lag", "dni_lag", "dhi_lag"))
        or c.endswith(("roll3h", "roll6h", "roll12h", "roll24h"))
        or c == "kc"
    ]

    # Future meteo mode: include raw weather as known future (NWP scenario)
    if args.use_future_meteo:
        logger.info("using future meteorological data as known covariates (NWP mode)")
        for c in raw_meteo:
            if c not in known_future:
                known_future.append(c)
    else:
        logger.info("standard mode: weather variables treated as past unknown")
        past_unknown.extend(raw_meteo)

    logger.info(f"known future covariates ({len(known_future)}): {sorted(known_future)[:5]}...")
    logger.info(f"past unknown covariates ({len(past_unknown)}): {sorted(past_unknown)[:5]}...")

    max_encoder_length = args.seq_len
    max_prediction_length = args.horizon
    logger.info(f"encoder length: {max_encoder_length}h, prediction horizon: {max_prediction_length}h")

    # Chronological train/validation/test split (3-way)
    max_time_idx = df["time_idx"].max()
    cutoff_train = int(max_time_idx * args.train_ratio)
    cutoff_val = int(max_time_idx * (args.train_ratio + args.val_ratio))

    logger.info(f"chronological 3-way split:")
    logger.info(f"  train: 0 to {cutoff_train} ({args.train_ratio:.1%})")
    logger.info(f"  validation: {cutoff_train+1} to {cutoff_val} ({args.val_ratio:.1%})")
    logger.info(f"  test: {cutoff_val+1} to {max_time_idx} ({args.test_ratio:.1%})")

    # Training dataset
    logger.info("creating training dataset...")
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff_train],
        time_idx="time_idx",
        target=target,
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=sorted(set(known_future)),
        time_varying_unknown_reals=[target] + sorted(set(past_unknown)),
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation="standard"),
        allow_missing_timesteps=True,
    )

    # Validation dataset (for early stopping)
    logger.info("creating validation dataset...")
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[(df["time_idx"] > cutoff_train) & (df["time_idx"] <= cutoff_val)],
        predict=False,
        stop_randomization=True,
    )

    # Test dataset (for final evaluation - NEVER seen during training!)
    logger.info("creating test dataset...")
    test_dataset = TimeSeriesDataSet.from_dataset(
        training, df[df["time_idx"] > cutoff_val], predict=False, stop_randomization=True
    )

    train_loader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)
    logger.info(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}, test batches: {len(test_loader)}")

    # Build Temporal Fusion Transformer model
    logger.info("building temporal fusion transformer model...")
    loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_heads,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_size // 2,
        loss=loss,
        optimizer="adam",
        reduce_on_plateau_patience=3,
    )
    logger.info(
        f"model config: hidden={args.hidden_size}, heads={args.attention_heads}, "
        f"dropout={args.dropout}, lr={args.learning_rate}"
    )

    # Training callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=True),
        ModelCheckpoint(dirpath=str(out_dir), filename="tft-best", monitor="val_loss", save_top_k=1, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # PyTorch Lightning Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger.info(f"using accelerator: {accelerator}")
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    logger.info("starting training...")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("training completed")

    # Load best model checkpoint
    best_path = callbacks[1].best_model_path  # type: ignore[index]
    if best_path:
        logger.info(f"loading best model from {best_path}")
        tft = TemporalFusionTransformer.load_from_checkpoint(best_path)
    else:
        logger.warning("no best checkpoint found, using final model")

    # Generate predictions on TEST set (never seen during training!)
    logger.info("generating predictions on TEST set...")
    preds, x = tft.predict(test_loader, return_x=True, mode="prediction")
    preds_np = preds.detach().cpu().numpy()  # shape (N, horizon)
    decoder_time_idx = x["decoder_time_idx"].detach().cpu().numpy()
    encoder_time_idx = x["encoder_time_idx"].detach().cpu().numpy()
    decoder_target = x["decoder_target"].detach().cpu().numpy()

    # Build timestamp lookup and naive baseline
    time_lookup: Dict[int, pd.Timestamp] = {int(ti): ts for ti, ts in zip(df["time_idx"].astype(int), df.index)}
    naive_map = {h: df["pv"].shift(24 - h).reset_index(drop=True) for h in range(1, args.horizon + 1)}
    train_series = df.loc[df["time_idx"] <= cutoff_train, "pv"].values

    # Export predictions in long format with timestamps
    logger.info("exporting predictions in long format...")
    rows = []
    for i in range(preds_np.shape[0]):
        for h_idx in range(preds_np.shape[1]):
            horizon_h = h_idx + 1
            origin_ti = int(encoder_time_idx[i, -1])
            forecast_ti = int(decoder_time_idx[i, h_idx])
            origin_ts = time_lookup.get(origin_ti)
            forecast_ts = time_lookup.get(forecast_ti)
            y_true = float(decoder_target[i, h_idx])
            y_pred = float(preds_np[i, h_idx])
            naive_series = naive_map[horizon_h]
            naive_val = float(naive_series.iloc[forecast_ti]) if forecast_ti < len(naive_series) else np.nan

            rows.append(
                {
                    "time_idx": forecast_ti,
                    "origin_timestamp_utc": origin_ts.isoformat() if origin_ts is not None else None,
                    "forecast_timestamp_utc": forecast_ts.isoformat() if forecast_ts is not None else None,
                    "horizon_h": horizon_h,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_naive": naive_val,
                }
            )

    pred_df = pd.DataFrame(rows)
    pred_path = out_dir / "predictions_test_tft.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"saved {len(pred_df)} predictions to {pred_path}")

    # Compute metrics per horizon
    logger.info("computing per-horizon metrics...")
    metrics = []
    for h in range(1, args.horizon + 1):
        sub = pred_df[pred_df["horizon_h"] == h]
        naive_vals = sub["y_naive"].values
        valid_mask = ~np.isnan(naive_vals)
        y_true = sub["y_true"].values[valid_mask]
        y_pred = sub["y_pred"].values[valid_mask]
        naive_valid = naive_vals[valid_mask]

        rmse_model = rmse(y_true, y_pred)
        rmse_naive = rmse(y_true, naive_valid)
        mase_model = mase(y_true, y_pred, train_series=train_series, m=24)
        mase_naive = mase(y_true, naive_valid, train_series=train_series, m=24)

        metrics.append(
            {
                "horizon_h": h,
                "rmse_model": float(rmse_model),
                "rmse_naive": float(rmse_naive),
                "mase_model": float(mase_model),
                "mase_naive": float(mase_naive),
            }
        )

    # Summary metrics
    metric_summary = {
        "rmse_model_avg": float(np.mean([m["rmse_model"] for m in metrics])),
        "rmse_naive_avg": float(np.mean([m["rmse_naive"] for m in metrics])),
        "mase_model_avg": float(np.mean([m["mase_model"] for m in metrics])),
        "mase_naive_avg": float(np.mean([m["mase_naive"] for m in metrics])),
    }

    metrics_path = out_dir / "metrics_test_tft.json"
    (out_dir / "metrics_test_tft.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"saved per-horizon metrics to {metrics_path}")

    summary_path = out_dir / "metrics_summary.json"
    (out_dir / "metrics_summary.json").write_text(json.dumps(metric_summary, indent=2))
    logger.info(f"saved summary metrics to {summary_path}")
    logger.info(
        f"average RMSE: {metric_summary['rmse_model_avg']:.4f}, "
        f"average MASE: {metric_summary['mase_model_avg']:.4f}"
    )

    # Save training configuration
    cfg = {
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "attention_heads": args.attention_heads,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "use_future_meteo": args.use_future_meteo,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "best_model_path": str(best_path) if best_path else None,
        "known_future": sorted(set(known_future)),
        "past_unknown": sorted(set(past_unknown)),
    }
    config_path = out_dir / "config_tft.json"
    (out_dir / "config_tft.json").write_text(json.dumps(cfg, indent=2))
    logger.info(f"saved training config to {config_path}")
    logger.info("tft training pipeline completed successfully")


if __name__ == "__main__":
    main()
