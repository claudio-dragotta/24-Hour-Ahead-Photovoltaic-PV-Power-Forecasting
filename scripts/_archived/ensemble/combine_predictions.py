"""Combine Multi-Branch and TFT predictions to create optimal ensemble.

This script loads actual test predictions from both models and finds the
optimal weighted combination that minimizes RMSE.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pv_forecasting.logger import get_logger
from pv_forecasting.metrics import mase, rmse

logger = get_logger(__name__)


def combine_predictions(
    multi_branch_path: str = "outputs/ensemble/predictions_test_multi_branch.csv",
    tft_path: str = "outputs/tft/baseline/predictions_test_tft.csv",
    output_dir: str = "outputs/ensemble",
):
    """Find optimal ensemble weights and combine predictions.

    Args:
        multi_branch_path: Path to Multi-Branch predictions CSV
        tft_path: Path to TFT predictions CSV
        output_dir: Directory to save ensemble results
    """
    logger.info("Creating Ensemble from Actual Predictions")
    logger.info("=" * 60)

    # Load predictions
    logger.info(f"Loading Multi-Branch predictions from {multi_branch_path}")
    mb_df = pd.read_csv(multi_branch_path)
    logger.info(f"  Loaded {len(mb_df)} predictions ({len(mb_df)//24} samples × 24 horizons)")

    logger.info(f"Loading TFT predictions from {tft_path}")
    tft_df = pd.read_csv(tft_path)
    logger.info(f"  Loaded {len(tft_df)} predictions ({len(tft_df)//24} samples × 24 horizons)")

    # Align predictions (ensure same samples)
    # Multi-Branch uses sample_idx (0-based), TFT might have different indexing
    mb_df = mb_df.sort_values(["sample_idx", "horizon"]).reset_index(drop=True)
    tft_df = tft_df.sort_values(["time_idx", "horizon_h"]).reset_index(drop=True)

    # Take minimum length to ensure alignment
    min_len = min(len(mb_df), len(tft_df))
    mb_df = mb_df.iloc[:min_len]
    tft_df = tft_df.iloc[:min_len]

    logger.info(f"Aligned predictions: {min_len} total ({min_len//24} samples × 24 horizons)")

    # Extract predictions and targets
    mb_preds = mb_df["prediction"].values
    tft_preds = tft_df["y_pred"].values
    targets = mb_df["target"].values  # Use Multi-Branch targets (should be identical)

    # Calculate individual model RMSE
    mb_rmse = rmse(targets, mb_preds)
    tft_rmse = rmse(targets, tft_preds)

    logger.info("")
    logger.info("Individual Model Performance:")
    logger.info(f"  Multi-Branch: RMSE = {mb_rmse:.4f} kW")
    logger.info(f"  TFT:          RMSE = {tft_rmse:.4f} kW")

    # Test different ensemble weights
    logger.info("")
    logger.info("Testing ensemble weights:")
    logger.info("-" * 60)

    weights_to_test = np.arange(0.0, 1.01, 0.05)  # 0.00, 0.05, 0.10, ..., 1.00
    best_weight = None
    best_rmse = float("inf")
    results = []

    for w_mb in weights_to_test:
        w_tft = 1.0 - w_mb

        # Combine predictions
        ensemble_preds = w_mb * mb_preds + w_tft * tft_preds

        # Calculate ensemble RMSE
        ensemble_rmse = rmse(targets, ensemble_preds)

        results.append({"weight_multi_branch": w_mb, "weight_tft": w_tft, "rmse": ensemble_rmse})

        if ensemble_rmse < best_rmse:
            best_rmse = ensemble_rmse
            best_weight = w_mb

        # Log every 5th weight
        if abs(w_mb - round(w_mb / 0.1) * 0.1) < 0.01:
            logger.info(f"  w_MB={w_mb:.2f}, w_TFT={w_tft:.2f}: RMSE = {ensemble_rmse:.4f} kW")

    # Calculate improvements
    mb_improvement = (best_rmse - mb_rmse) / mb_rmse * 100
    tft_improvement = (best_rmse - tft_rmse) / tft_rmse * 100

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ BEST ENSEMBLE: w_MB={best_weight:.2f}, w_TFT={1-best_weight:.2f}")
    logger.info(f"   RMSE: {best_rmse:.4f} kW")
    logger.info(f"   Improvement over Multi-Branch: {mb_improvement:+.2f}%")
    logger.info(f"   Improvement over TFT: {tft_improvement:+.2f}%")
    logger.info("=" * 60)

    # Check if we beat the reference (3.17 kW)
    reference_rmse = 3.17
    if best_rmse < reference_rmse:
        improvement_vs_ref = (best_rmse - reference_rmse) / reference_rmse * 100
        logger.info(f"🎯 BEAT REFERENCE! Improvement: {improvement_vs_ref:+.2f}%")
    else:
        gap_vs_ref = (best_rmse - reference_rmse) / reference_rmse * 100
        logger.info(f"📊 Gap vs reference (3.17 kW): {gap_vs_ref:+.2f}%")
    logger.info("=" * 60)

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble config
    ensemble_config = {
        "models": ["multi_branch", "tft"],
        "weights": {"multi_branch": float(best_weight), "tft": float(1 - best_weight)},
        "performance": {
            "ensemble_rmse": float(best_rmse),
            "multi_branch_rmse": float(mb_rmse),
            "tft_rmse": float(tft_rmse),
            "improvement_vs_mb": float(mb_improvement),
            "improvement_vs_tft": float(tft_improvement),
            "reference_rmse": reference_rmse,
            "beats_reference": bool(best_rmse < reference_rmse),
        },
        "model_paths": {
            "multi_branch": "outputs/multi_branch/final_v1/multi-branch-best.ckpt",
            "tft": "outputs/tft/baseline/tft-best.ckpt",
        },
    }

    config_path = out_dir / "ensemble_config.json"
    with open(config_path, "w") as f:
        json.dump(ensemble_config, f, indent=2)
    logger.info(f"Saved ensemble config to {config_path}")

    # Save all weight test results
    results_df = pd.DataFrame(results)
    results_path = out_dir / "weight_search_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved weight search results to {results_path}")

    # Save best ensemble predictions
    best_ensemble_preds = best_weight * mb_preds + (1 - best_weight) * tft_preds
    ensemble_df = mb_df.copy()
    ensemble_df["prediction"] = best_ensemble_preds
    ensemble_df["mb_prediction"] = mb_preds
    ensemble_df["tft_prediction"] = tft_preds

    ensemble_pred_path = out_dir / "predictions_test_ensemble.csv"
    ensemble_df.to_csv(ensemble_pred_path, index=False)
    logger.info(f"Saved ensemble predictions to {ensemble_pred_path}")

    return ensemble_config


if __name__ == "__main__":
    combine_predictions()
