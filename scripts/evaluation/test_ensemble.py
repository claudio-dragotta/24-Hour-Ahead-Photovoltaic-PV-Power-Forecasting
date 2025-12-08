"""Test ensemble on held-out test set using pre-optimized weights.

This script applies the ensemble weights (optimized on validation set)
to the test set predictions to evaluate final performance on unseen data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pv_forecasting.metrics import mase, rmse


def load_ensemble_weights(weights_path: Path) -> tuple[list[str], list[float]]:
    """Load ensemble weights from JSON file.

    Returns:
        Tuple of (model_names, weights).
    """
    with open(weights_path) as f:
        weights_data = json.load(f)

    model_names = weights_data["model_names"]
    weights = weights_data["weights"]

    print(f"[info] loaded ensemble weights from {weights_path}")
    print(f"[info] models: {model_names}")
    print(f"[info] weights: {[f'{w:.3f}' for w in weights]}")

    return model_names, weights


def load_test_predictions(pred_path: Path, model_name: str) -> pd.DataFrame:
    """Load test predictions for a single model."""
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions not found: {pred_path}")

    df = pd.read_csv(pred_path)
    print(f"[info] loaded {len(df)} test predictions from {model_name}")

    required_cols = ["origin_timestamp_utc", "forecast_timestamp_utc", "horizon_h", "y_true", "y_pred"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"missing required columns in {model_name} predictions")

    return df


def combine_ensemble_predictions(prediction_dfs: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    """Combine predictions from multiple models using ensemble weights.

    Args:
        prediction_dfs: List of prediction DataFrames (one per model).
        weights: Ensemble weights (must sum to 1.0).

    Returns:
        DataFrame with ensemble predictions.
    """
    # Start with first model's predictions
    ensemble_df = prediction_dfs[0].copy()
    ensemble_df = ensemble_df.rename(columns={"y_pred": "y_pred_0"})

    # Merge other models' predictions
    for i, df in enumerate(prediction_dfs[1:], start=1):
        df_merge = df[["origin_timestamp_utc", "forecast_timestamp_utc", "horizon_h", "y_pred"]].copy()
        df_merge = df_merge.rename(columns={"y_pred": f"y_pred_{i}"})

        ensemble_df = ensemble_df.merge(
            df_merge, on=["origin_timestamp_utc", "forecast_timestamp_utc", "horizon_h"], how="inner"
        )

    # Compute weighted average
    pred_cols = [f"y_pred_{i}" for i in range(len(prediction_dfs))]
    preds_array = ensemble_df[pred_cols].values  # shape (N, num_models)
    weights_array = np.array(weights)

    ensemble_df["y_pred_ensemble"] = preds_array @ weights_array

    return ensemble_df


def compute_metrics(df: pd.DataFrame, train_series: np.ndarray) -> dict:
    """Compute evaluation metrics for ensemble predictions.

    Args:
        df: DataFrame with y_true and y_pred_ensemble columns.
        train_series: Training series for MASE computation.

    Returns:
        Dictionary of metrics.
    """
    y_true = df["y_true"].values
    y_pred = df["y_pred_ensemble"].values

    # Overall metrics
    rmse_val = rmse(y_true, y_pred)
    mase_val = mase(y_true, y_pred, train_series=train_series, m=24)

    # Per-horizon metrics
    per_horizon = []
    for h in range(1, 25):
        mask = df["horizon_h"] == h
        if mask.sum() == 0:
            continue

        y_true_h = y_true[mask]
        y_pred_h = y_pred[mask]

        per_horizon.append(
            {
                "horizon_h": h,
                "rmse": float(rmse(y_true_h, y_pred_h)),
                "mase": float(mase(y_true_h, y_pred_h, train_series=train_series, m=24)),
            }
        )

    return {"overall": {"rmse": float(rmse_val), "mase": float(mase_val)}, "per_horizon": per_horizon}


def main():
    parser = argparse.ArgumentParser(description="test ensemble on held-out test set")

    # Ensemble weights
    parser.add_argument("--weights", type=str, required=True, help="path to ensemble_weights.json file")

    # Model predictions (test set) - New simple args
    parser.add_argument("--lgbm", type=str, default=None, help="path to LightGBM test predictions")
    parser.add_argument("--cnn", type=str, default=None, help="path to CNN-BiLSTM test predictions")
    parser.add_argument("--tft", type=str, default=None, help="path to TFT test predictions")

    # Model predictions (test set) - Old args for backward compatibility
    parser.add_argument("--lgbm-baseline", type=str, default=None)
    parser.add_argument("--lgbm-lag72", type=str, default=None)
    parser.add_argument("--cnn-baseline", type=str, default=None)
    parser.add_argument("--cnn-lag72", type=str, default=None)
    parser.add_argument("--tft-baseline", type=str, default=None)
    parser.add_argument("--tft-lag72", type=str, default=None)

    # Training data for MASE
    parser.add_argument(
        "--train-data",
        type=str,
        default="outputs_baseline/processed.parquet",
        help="path to training data (for MASE computation)",
    )

    parser.add_argument("--outdir", type=str, default="outputs_ensemble", help="output directory")

    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ENSEMBLE TEST SET EVALUATION")
    print("=" * 60)

    # Load ensemble weights
    model_names, weights = load_ensemble_weights(Path(args.weights))

    # Map model names to test prediction files
    # Supports both old naming (with -Baseline/-Lag72) and new simple naming
    model_pred_map = {
        # New simple names (used by ensemble.py)
        "LightGBM": args.lgbm or args.lgbm_baseline or args.lgbm_lag72,
        "CNN-BiLSTM": args.cnn or args.cnn_baseline or args.cnn_lag72,
        "TFT": args.tft or args.tft_baseline or args.tft_lag72,
        # Old names (backward compatibility)
        "LightGBM-Baseline": args.lgbm_baseline,
        "LightGBM-Lag72": args.lgbm_lag72,
        "CNN-BiLSTM-Baseline": args.cnn_baseline,
        "CNN-BiLSTM-Lag72": args.cnn_lag72,
        "TFT-Baseline": args.tft_baseline,
        "TFT-Lag72": args.tft_lag72,
    }

    # Load test predictions for selected models
    prediction_dfs = []
    for model_name in model_names:
        pred_path_str = model_pred_map.get(model_name)
        if pred_path_str is None:
            raise ValueError(f"no test predictions provided for {model_name}")

        pred_path = Path(pred_path_str)
        df = load_test_predictions(pred_path, model_name)
        prediction_dfs.append(df)

    # Combine predictions using ensemble weights
    print(f"\n[info] combining {len(prediction_dfs)} model predictions...")
    ensemble_df = combine_ensemble_predictions(prediction_dfs, weights)

    # Load training data for MASE
    print(f"\n[info] loading training data from {args.train_data}...")
    train_df = pd.read_parquet(args.train_data)
    train_series = train_df["pv"].values[: int(len(train_df) * 0.6)]  # First 60% is train

    # Compute metrics
    print(f"\n[info] computing test set metrics...")
    metrics = compute_metrics(ensemble_df, train_series)

    # Save results
    ensemble_pred_path = out_dir / "predictions_test_ensemble.csv"
    ensemble_df.to_csv(ensemble_pred_path, index=False)
    print(f"[info] saved ensemble test predictions to {ensemble_pred_path}")

    metrics_path = out_dir / "metrics_test_ensemble.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, indent=2, fp=f)
    print(f"[info] saved test metrics to {metrics_path}")

    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS (NEVER SEEN DURING TRAINING/ENSEMBLE OPTIMIZATION)")
    print("=" * 60)
    print(f"RMSE: {metrics['overall']['rmse']:.4f}")
    print(f"MASE: {metrics['overall']['mase']:.4f}")
    print("=" * 60)

    print("\nPer-horizon performance:")
    for h_metrics in metrics["per_horizon"][:5]:  # Show first 5
        h = h_metrics["horizon_h"]
        print(f"  h={h:2d}: RMSE={h_metrics['rmse']:.4f}, MASE={h_metrics['mase']:.4f}")
    print("  ...")
    for h_metrics in metrics["per_horizon"][-5:]:  # Show last 5
        h = h_metrics["horizon_h"]
        print(f"  h={h:2d}: RMSE={h_metrics['rmse']:.4f}, MASE={h_metrics['mase']:.4f}")

    print("\n[info] ensemble test evaluation completed successfully")


if __name__ == "__main__":
    main()
