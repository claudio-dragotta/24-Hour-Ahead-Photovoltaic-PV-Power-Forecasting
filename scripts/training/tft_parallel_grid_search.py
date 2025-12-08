"""Parallel Grid Search for TFT hyperparameter optimization.

Uses Ray Tune for efficient parallel hyperparameter search with GPU optimization.
Automatically distributes trials across available GPU memory.
"""

from __future__ import annotations

import argparse
import json
import os
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
    import sys
    os.environ.setdefault("PL_DISABLE_FORK", "1")  # avoid Lightning forking inside Ray workers
    torch.set_num_threads(1)
    print(f"[TRIAL DEBUG] Starting trial with config: {config}", flush=True)
    sys.stdout.flush()
    
    seed_everything(42)
    print("[TRIAL DEBUG] Seed set", flush=True)
    
    # CRITICAL: Use absolute path - Ray workers run in different directories
    # Always use pre-cached processed.parquet to avoid data loading issues
    processed_path = Path("/home/claudio/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/outputs/processed.parquet")
    
    print(f"[TRIAL DEBUG] Checking path: {processed_path}", flush=True)
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Run training once to generate outputs/processed.parquet first."
        )
    
    print("[TRIAL DEBUG] Loading data...", flush=True)
    df = pd.read_parquet(processed_path)
    print(f"[TRIAL DEBUG] Data loaded: {len(df)} rows", flush=True)
    
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
    
    print(f"[TRIAL DEBUG] Creating datasets (train cutoff={cutoff_train}, val cutoff={cutoff_val})", flush=True)
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
    print("[TRIAL DEBUG] Training dataset created", flush=True)
    
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[(df["time_idx"] > cutoff_train) & (df["time_idx"] <= cutoff_val)],
        predict=False,
        stop_randomization=True,
    )
    print("[TRIAL DEBUG] Validation dataset created", flush=True)
    
    # CRITICAL FIX: num_workers=0 to avoid deadlock in Ray workers
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    print("[TRIAL DEBUG] DataLoaders created (num_workers=0 to avoid Ray deadlock)", flush=True)
    
    # Build TFT model with trial config
    print(f"[TRIAL DEBUG] Building TFT model with config: hidden={config['hidden_size']}, heads={config['attention_heads']}", flush=True)
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
    print(f"[TRIAL DEBUG] TFT model created ({sum(p.numel() for p in tft.parameters())} params)", flush=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=False),
    ]
    
    print("[TRIAL DEBUG] Creating Trainer...", flush=True)
    # Trainer with Ray Tune integration
    # CRITICAL: Use accelerator="auto" to let PyTorch Lightning handle GPU assignment
    # Ray Tune manages GPU allocation, Lightning should auto-detect
    trainer = Trainer(
        max_epochs=config.get("max_epochs", 50),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="single_device",  # avoid DDP/spawn under Ray
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    print(f"[TRIAL DEBUG] Trainer created", flush=True)
    
    # Train
    print("[TRIAL DEBUG] Starting training...", flush=True)
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("[TRIAL DEBUG] Training complete!", flush=True)
    
    # Report final validation loss to Ray Tune
    final_val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
    print(f"[TRIAL DEBUG] Final val_loss={final_val_loss}", flush=True)
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
        default=150,
        help="Number of random samples (default=150, optimal balance)",
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
    
    # Convert to absolute path
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define RANDOM search space (sample best configs efficiently)
    # Random search is MORE EFFICIENT than grid search:
    # - With 100-150 trials, covers the space well
    # - Finds optimal configs faster (proven by research)
    # - ASHA early stopping works better with diverse trials
    search_space = {
        "hidden_size": tune.choice([32, 48, 64, 96, 128]),  # 5 options
        "attention_heads": tune.choice([2, 4, 8]),  # 3 options  
        "dropout": tune.uniform(0.1, 0.5),  # continuous range
        "learning_rate": tune.loguniform(1e-5, 1e-3),  # log scale
        "hidden_continuous_size": tune.choice([16, 32, 48, 64]),  # 4 options
        "max_epochs": args.max_epochs,
    }
    
    # Default: 150 trials (unless user specifies --num-samples)
    num_trials = args.num_samples if args.num_samples else 150
    logger.info(f"Random search with {num_trials} trial samples")
    
    # ASHA scheduler for early stopping of bad trials
    grace = max(1, min(8, args.max_epochs // 2))  # ensure >0
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=args.max_epochs,
        grace_period=grace,
        reduction_factor=3,  # More aggressive pruning
    )
    
    # Reporter for progress
    reporter = CLIReporter(
        metric_columns=["val_loss", "training_iteration"],
        max_report_frequency=30,
    )
    
    # Run random search
    logger.info(f"Starting RANDOM parallel search with Ray Tune")
    logger.info(f"Total trials: {num_trials}")
    logger.info(f"Max concurrent trials: {args.max_concurrent}")
    logger.info(f"GPUs per trial: {args.gpus_per_trial}")
    logger.info(f"Estimated time: ~{num_trials * 12 / args.max_concurrent / 60:.1f} hours")
    logger.info(f"  (assuming 12 min/trial with ASHA early stopping)")
    logger.info(f"Search space: {search_space}")
    
    analysis = tune.run(
        train_tft_trial,
        config=search_space,
        num_samples=num_trials,  # Random sampling
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={
            "cpu": 2,  # Reduced CPU per trial for more parallelism
            "gpu": args.gpus_per_trial,
        },
        max_concurrent_trials=args.max_concurrent,
        storage_path=str(output_dir),
        name="tft_random_search",
        verbose=1,
        raise_on_failed_trial=False,  # Continue even if some trials fail
        resume=False,  # avoid resuming broken state; set to "AUTO" if you need resume
    )
    
    # Get best configuration (if any trial succeeded)
    best_trial = None
    try:
        best_trial = analysis.get_best_trial("val_loss", "min", "last")
    except Exception:
        logger.error("No successful trials with val_loss; check error logs.")
    if best_trial is None or "val_loss" not in best_trial.last_result:
        logger.error("Best trial missing val_loss; skipping result export.")
        return
    logger.info(f"\nBest trial config: {best_trial.config}")
    logger.info(f"Best trial validation loss: {best_trial.last_result['val_loss']:.4f}")
    
    # Save results
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Export all results to CSV
    df_results = analysis.results_df
    df_results.to_csv(results_dir / "grid_search_results.csv", index=False)
    
    # Save best config
    best_config = {
        "config": best_trial.config,
        "val_loss": float(best_trial.last_result["val_loss"]),
        "trial_id": best_trial.trial_id,
    }
    
    with open(results_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    
    logger.info(f"\nResults saved to {results_dir}")
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
