from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer

from pv_forecasting.metrics import mase, rmse
from pv_forecasting.pipeline import load_and_engineer_features, persist_processed


def parse_args():
    ap = argparse.ArgumentParser(description="Train TFT (PyTorch Forecasting) for 24h-ahead PV")
    ap.add_argument("--pv-path", type=str, default="data/raw/pv_dataset.xlsx")
    ap.add_argument("--wx-path", type=str, default="data/raw/wx_dataset.xlsx")
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney")
    ap.add_argument("--seq-len", type=int, default=168)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", type=str, default="outputs_tft")
    ap.add_argument("--use-future-meteo", action="store_true", help="Use future weather covariates if available (NWP)")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation share (chronological split)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_engineer_features(Path(args.pv_path), Path(args.wx_path), args.local_tz)
    persist_processed(df, Path("outputs"))

    target = "pv"

    # Known future covariates: time & solar & clearsky
    known_future = [
        c for c in [
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "sp_zenith",
            "sp_azimuth",
            "cs_ghi",
            "cs_dni",
            "cs_dhi",
        ] if c in df.columns
    ]

    raw_meteo = [c for c in ["ghi", "dni", "dhi", "temp", "humidity", "clouds", "wind_speed"] if c in df.columns]
    past_unknown = [
        c
        for c in df.columns
        if c.startswith(("pv_lag", "ghi_lag", "dni_lag", "dhi_lag"))
        or c.endswith("roll3h")
        or c.endswith("roll6h")
        or c.endswith("roll12h")
        or c.endswith("roll24h")
        or c == "kc"
    ]

    # if user states future meteo is available, include raw meteo as known future
    if args.use_future_meteo:
        for c in raw_meteo:
            if c not in known_future:
                known_future.append(c)
    else:
        past_unknown.extend(raw_meteo)

    max_encoder_length = args.seq_len
    max_prediction_length = args.horizon

    # split train/val chronologically
    train_ratio = 1 - args.val_ratio
    cutoff = int(df["time_idx"].max() * train_ratio)

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff],
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

    validation = TimeSeriesDataSet(
        df[df["time_idx"] > cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known_future,
        time_varying_unknown_reals=[target] + past_unknown,
        target_normalizer=training.target_normalizer,
        allow_missing_timesteps=True,
    )

    train_loader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    # build model
    loss = QuantileLoss([0.1, 0.5, 0.9])
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=32,
        loss=loss,
        optimizer="adam",
        reduce_on_plateau_patience=3,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, mode="min"),
        ModelCheckpoint(dirpath=str(out_dir), filename="tft-best", monitor="val_loss", save_top_k=1, mode="min")
    ]

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        log_every_n_steps=50,
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # best model
    best_path = callbacks[1].best_model_path  # type: ignore[index]
    if best_path:
        tft = TemporalFusionTransformer.load_from_checkpoint(best_path)

    # predict on validation window and save long-format with ground truth
    preds, x = tft.predict(val_loader, return_x=True)
    preds_np = preds.detach().cpu().numpy()  # shape (N, horizon)
    decoder_time_idx = x["decoder_time_idx"].detach().cpu().numpy()
    encoder_time_idx = x["encoder_time_idx"].detach().cpu().numpy()
    decoder_target = x["decoder_target"].detach().cpu().numpy()

    time_lookup: Dict[int, pd.Timestamp] = {int(ti): ts for ti, ts in zip(df["time_idx"].astype(int), df.index)}
    naive_map = {h: df["pv"].shift(24 - h).reset_index(drop=True) for h in range(1, args.horizon + 1)}
    train_series = df.loc[df["time_idx"] <= cutoff, "pv"].values

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

            rows.append({
                "time_idx": forecast_ti,
                "origin_timestamp_utc": origin_ts.isoformat() if origin_ts is not None else None,
                "forecast_timestamp_utc": forecast_ts.isoformat() if forecast_ts is not None else None,
                "horizon_h": horizon_h,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_naive": naive_val,
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / "predictions_val_tft.csv", index=False)

    metrics = []
    for h in range(1, args.horizon + 1):
        sub = pred_df[pred_df["horizon_h"] == h]
        naive_vals = sub["y_naive"].values
        valid_mask = ~np.isnan(naive_vals)
        y_true = sub["y_true"].values[valid_mask]
        y_pred = sub["y_pred"].values[valid_mask]
        naive_valid = naive_vals[valid_mask]
        metrics.append({
            "horizon_h": h,
            "rmse_model": rmse(y_true, y_pred),
            "rmse_naive": rmse(y_true, naive_valid),
            "mase_model": mase(y_true, y_pred, train_series=train_series, m=24),
            "mase_naive": mase(y_true, naive_valid, train_series=train_series, m=24),
        })
    metric_summary = {
        "rmse_model_avg": float(np.mean([m["rmse_model"] for m in metrics])),
        "rmse_naive_avg": float(np.mean([m["rmse_naive"] for m in metrics])),
        "mase_model_avg": float(np.mean([m["mase_model"] for m in metrics])),
        "mase_naive_avg": float(np.mean([m["mase_naive"] for m in metrics])),
    }
    (out_dir / "metrics_val_tft.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "metrics_summary.json").write_text(json.dumps(metric_summary, indent=2))

    # also export a simple deterministic series (p50) aligned to timestamps if needed
    # Save config
    cfg = {
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "use_future_meteo": args.use_future_meteo,
        "val_ratio": args.val_ratio,
        "best_model_path": best_path,
        "known_future": sorted(set(known_future)),
        "past_unknown": sorted(set(past_unknown)),
    }
    (out_dir / "config_tft.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
