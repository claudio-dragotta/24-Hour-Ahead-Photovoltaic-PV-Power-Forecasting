"""ARCHIVE: tft_parallel_grid_search.py

Original: scripts/training/tft_parallel_grid_search.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from pv_forecasting.logger import get_logger

logger = get_logger(__name__)


def train_tft_trial(config: Dict, checkpoint_dir=None):
    os.environ.setdefault("PL_DISABLE_FORK", "1")
    torch.set_num_threads(1)
    seed_everything(2)
    processed_path = Path("/home/claudio/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/outputs/processed.parquet")
    if not processed_path.exists():
        raise FileNotFoundError("Processed data not found; run preprocessing first.")
    df = pd.read_parquet(processed_path)
    if "sp_zenith" in df.columns:
        df["sample_weight"] = 1.0
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="pv",
        group_ids=["series_id"],
        max_encoder_length=168,
        max_prediction_length=24,
        time_varying_known_reals=[],
        time_varying_unknown_reals=["pv"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        allow_missing_timesteps=True,
    )
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = training.to_dataloader(train=False, batch_size=64, num_workers=0)
    loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    tft = None
    callbacks = [EarlyStopping(monitor="val_loss", patience=8, mode="min")]
    trainer = Trainer(
        max_epochs=config.get("max_epochs", 50), callbacks=callbacks, enable_checkpointing=False, logger=False
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    tune.report(val_loss=0.0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Parallel grid search for TFT with Ray Tune")
    ap.add_argument("--output-dir", type=str, default="outputs/tft/grid_search")
    ap.add_argument("--num-samples", type=int, default=150)
    ap.add_argument("--max-concurrent", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--gpus-per-trial", type=float, default=0.33)
    return ap.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    search_space = {
        "hidden_size": tune.choice([32, 48, 64, 96, 128]),
        "attention_heads": tune.choice([2, 4, 8]),
        "dropout": tune.uniform(0.1, 0.5),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "hidden_continuous_size": tune.choice([16, 32, 48, 64]),
        "max_epochs": args.max_epochs,
    }
    scheduler = ASHAScheduler(metric="val_loss", mode="min", max_t=args.max_epochs, grace_period=1, reduction_factor=3)
    reporter = CLIReporter(metric_columns=["val_loss", "training_iteration"], max_report_frequency=30)
    analysis = tune.run(
        train_tft_trial,
        config=search_space,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 2, "gpu": args.gpus_per_trial},
        max_concurrent_trials=args.max_concurrent,
        storage_path=str(output_dir),
        name="tft_random_search",
    )
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df_results = analysis.results_df
    df_results.to_csv(results_dir / "grid_search_results.csv", index=False)


if __name__ == "__main__":
    main()
