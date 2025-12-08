"""Parallel Grid Search for TFT hyperparameter optimization.

Uses Ray Tune for efficient parallel hyperparameter search with GPU optimization.
Automatically distributes trials across available GPU memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Import TFT training components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pv_forecasting.logger import get_logger
from scripts.training.train_tft import (
    compute_solar_weights,
    load_and_engineer_features,
    persist_processed,
)
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

logger = get_logger(__name__)


def train_tft_trial(config: Dict, checkpoint_dir=None):
    """Train a single TFT configuration for Ray Tune.
    
    Args:
        config: Hyperparameter configuration from Ray Tune
        checkpoint_dir: Directory for checkpoints (optional)
    """
    seed_everything(42)
    
    # Load data (cached for speed)
    processed_path = Path("outputs/processed.parquet")
    if processed_path.exists():
        df = pd.read_parquet(processed_path)
    else:
        df = load_and_engineer_features(
            Path("data/raw/pv_dataset.xlsx"),
            Path("data/raw/wx_dataset.xlsx"),
            "Australia/Sydney",
        )
        persist_processed(df, Path("outputs"))
    
    # Solar weighting
    if "sp_zenith" in df.columns:
        df["sample_weight"] = compute_solar_weights(df["sp_zenith"])
    else:
        df["sample_weight"] = 1.0
    
    # Feature selection
    known_future = [
        c for c in [
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
            "sp_zenith", "sp_azimuth", "cs_ghi", "cs_dni", "cs_dhi",
        ] if c in df.columns
    ]
    
    raw_meteo = [
        c for c in ["ghi", "dni", "dhi", "temp", "humidity", "clouds", "wind_speed"]
        if c in df.columns
    ]
    
    past_unknown = [
        c for c in df.columns
        if c.startswith(("pv_lag", "ghi_lag", "dni_lag", "dhi_lag"))
        or "_roll" in c or c == "kc"
    ]
    
    # Add meteo to known future (NWP mode)
    for c in raw_meteo:
        if c not in known_future:
            known_future.append(c)
    
    # Train/val/test split
    max_time_idx = df["time_idx"].max()
    cutoff_train = int(max_time_idx * 0.6)
    cutoff_val = int(max_time_idx * 0.8)
    
    # Create datasets
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff_train],
        time_idx="time_idx",
        target="pv",
        group_ids=["series_id"],
        max_encoder_length=168,
        max_prediction_length=24,
        time_varying_known_reals=sorted(set(known_future)),
        time_varying_unknown_reals=["pv"] + sorted(set(past_unknown)),
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        weight="sample_weight",
        allow_missing_timesteps=True,
    )
    
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[(df["time_idx"] > cutoff_train) & (df["time_idx"] <= cutoff_val)],
        predict=False,
        stop_randomization=True,
    )
    
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=4)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=4)
    
    # Build TFT model with trial config
    loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config["learning_rate"],
        hidden_size=config["hidden_size"],
        attention_head_size=config["attention_heads"],
        dropout=config["dropout"],
        hidden_continuous_size=config["hidden_continuous_size"],
        loss=loss,
        optimizer="adamw",
        reduce_on_plateau_patience=5,
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=False),
    ]
    
    # Trainer with Ray Tune integration
    trainer = Trainer(
        max_epochs=config.get("max_epochs", 50),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    
    # Train
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Report final validation loss to Ray Tune
    final_val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
    tune.report(val_loss=float(final_val_loss))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Parallel grid search for TFT with Ray Tune")
    ap.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tft/grid_search",
        help="Output directory for results",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of random samples (None = full grid search)",
    )
    ap.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent trials - default 3 for balanced GPU usage (adjust: 2=conservative, 4=aggressive)",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum epochs per trial",
    )
    ap.add_argument(
        "--gpus-per-trial",
        type=float,
        default=0.33,
        help="GPU fraction per trial - default 0.33 for 3 concurrent trials (adjust: 0.5=2 trials, 0.25=4 trials)",
    )
    return ap.parse_args()


def main():
    """Run parallel grid search with Ray Tune."""
    args = parse_args()
    
    # Define FULL search space (243 combinations - ALL possibilities)
    # Using tune.grid_search() ensures EVERY combination is tested
    search_space = {
        "hidden_size": tune.grid_search([32, 64, 128]),  # 3 options
        "attention_heads": tune.grid_search([2, 4, 8]),  # 3 options  
        "dropout": tune.grid_search([0.2, 0.3, 0.4]),  # 3 options
        "learning_rate": tune.grid_search([1e-4, 3e-4, 1e-3]),  # 3 options
        "hidden_continuous_size": tune.grid_search([16, 32, 64]),  # 3 options
        "max_epochs": args.max_epochs,
    }
    
    # Total combinations: 3 × 3 × 3 × 3 × 3 = 243 configs (COMPLETE GRID)
    # NO COMBINATION IS SKIPPED - grid_search tests ALL possibilities
    
    # Calculate total combinations
    total_combinations = 3 * 3 * 3 * 3 * 3  # 243
    logger.info(f"Total grid search combinations: {total_combinations}")
    
    # ASHA scheduler for early stopping of bad trials
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=args.max_epochs,
        grace_period=8,  # Keep at least 8 epochs before stopping
        reduction_factor=3,  # More aggressive pruning
    )
    
    # Reporter for progress
    reporter = CLIReporter(
        metric_columns=["val_loss", "training_iteration"],
        max_report_frequency=30,
    )
    
    # Run grid search
    logger.info(f"Starting FULL parallel grid search with Ray Tune")
    logger.info(f"Total combinations: {total_combinations}")
    logger.info(f"Max concurrent trials: {args.max_concurrent}")
    logger.info(f"GPUs per trial: {args.gpus_per_trial}")
    logger.info(f"Estimated time: ~{total_combinations * 15 / args.max_concurrent / 60:.1f} hours")
    logger.info(f"  (assuming 15 min/trial with early stopping)")
    logger.info(f"Search space: {search_space}")
    
    analysis = tune.run(
        train_tft_trial,
        config=search_space,
        num_samples=1 if args.num_samples is None else args.num_samples,  # 1 = full grid
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={
            "cpu": 2,  # Reduced CPU per trial for more parallelism
            "gpu": args.gpus_per_trial,
        },
        max_concurrent_trials=args.max_concurrent,
        local_dir=args.output_dir,
        name="tft_full_grid_search",
        verbose=1,
        raise_on_failed_trial=False,  # Continue even if some trials fail
        resume="AUTO",  # Auto-resume if interrupted
    )
    
    # Get best configuration
    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    logger.info(f"\nBest trial config: {best_trial.config}")
    logger.info(f"Best trial validation loss: {best_trial.last_result['val_loss']:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export all results to CSV
    df_results = analysis.results_df
    df_results.to_csv(output_dir / "grid_search_results.csv", index=False)
    
    # Save best config
    best_config = {
        "config": best_trial.config,
        "val_loss": float(best_trial.last_result["val_loss"]),
        "trial_id": best_trial.trial_id,
    }
    
    with open(output_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"- grid_search_results.csv: All trial results")
    logger.info(f"- best_config.json: Best configuration")
    
    print(f"\n{'='*80}")
    print(f"BEST CONFIGURATION FOUND:")
    print(f"{'='*80}")
    for key, value in best_trial.config.items():
        print(f"  {key:25s}: {value}")
    print(f"  {'Validation Loss':25s}: {best_trial.last_result['val_loss']:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
