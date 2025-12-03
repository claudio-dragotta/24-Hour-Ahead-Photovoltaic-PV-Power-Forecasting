"""Ensemble optimization script for combining multiple PV forecasting models.

This module implements weighted ensemble combining predictions from:
- Temporal Fusion Transformer (TFT)
- LightGBM multi-horizon model
- CNN-BiLSTM (optional)

The ensemble weights are optimized using grid search or Optuna to minimize RMSE.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(
        description="ensemble optimization for PV forecasting models"
    )
    ap.add_argument(
        "--tft",
        type=str,
        default=None,
        help="path to TFT predictions CSV (predictions_test_tft.csv)"
    )
    ap.add_argument(
        "--lgbm",
        type=str,
        default=None,
        help="path to LightGBM predictions CSV (predictions_test_lgbm.csv)"
    )
    ap.add_argument(
        "--bilstm",
        type=str,
        default=None,
        help="path to CNN-BiLSTM predictions CSV (optional)"
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="outputs_ensemble",
        help="output directory for ensemble results"
    )
    ap.add_argument(
        "--method",
        type=str,
        choices=["grid", "optuna"],
        default="grid",
        help="optimization method (grid search or optuna)"
    )
    ap.add_argument(
        "--grid-steps",
        type=int,
        default=11,
        help="number of steps for grid search (default: 11 => 0.0, 0.1, ..., 1.0)"
    )
    ap.add_argument(
        "--optuna-trials",
        type=int,
        default=100,
        help="number of trials for optuna optimization"
    )
    ap.add_argument(
        "--metric",
        type=str,
        choices=["rmse", "mae", "mase"],
        default="rmse",
        help="metric to optimize (default: rmse)"
    )
    return ap.parse_args()


def load_predictions(path: Optional[str], name: str) -> Optional[pd.DataFrame]:
    """Load prediction CSV file.

    Args:
        path: Path to CSV file.
        name: Model name for logging.

    Returns:
        DataFrame with predictions or None if path not provided.
    """
    if path is None:
        print(f"[info] {name} predictions not provided, skipping")
        return None

    path_obj = Path(path)
    if not path_obj.exists():
        print(f"[warning] {name} predictions not found at {path}, skipping")
        return None

    df = pd.read_csv(path_obj)
    print(f"[info] loaded {len(df)} predictions from {name} ({path})")

    # Ensure required columns exist
    required = ["origin_timestamp_utc", "forecast_timestamp_utc", "horizon_h", "y_true", "y_pred"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"{name} predictions missing required columns: {required}")

    return df


def align_predictions(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Align multiple prediction DataFrames on (origin, forecast, horizon).

    Args:
        *dfs: Variable number of prediction DataFrames.

    Returns:
        Aligned DataFrame with columns: origin_timestamp_utc, forecast_timestamp_utc,
        horizon_h, y_true, pred_0, pred_1, pred_2, ...
    """
    dfs_valid = [df for df in dfs if df is not None]
    if len(dfs_valid) == 0:
        raise ValueError("no valid prediction dataframes to align")

    # Start with first DataFrame
    base = dfs_valid[0].copy()
    base = base.rename(columns={"y_pred": "pred_0"})

    # Merge remaining DataFrames
    for i, df in enumerate(dfs_valid[1:], start=1):
        df_merge = df[["origin_timestamp_utc", "forecast_timestamp_utc", "horizon_h", "y_pred"]].copy()
        df_merge = df_merge.rename(columns={"y_pred": f"pred_{i}"})

        base = base.merge(
            df_merge,
            on=["origin_timestamp_utc", "forecast_timestamp_utc", "horizon_h"],
            how="inner"
        )

    print(f"[info] aligned predictions: {len(base)} samples across {len(dfs_valid)} models")
    return base


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_series: np.ndarray,
    m: int = 24
) -> float:
    """Compute Mean Absolute Scaled Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        train_series: Training series for scaling.
        m: Seasonal period.

    Returns:
        MASE value.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    naive_mae = np.mean(np.abs(np.diff(train_series, n=m)))
    return float(mae / naive_mae) if naive_mae > 0 else float("inf")


def ensemble_predict(
    aligned: pd.DataFrame,
    weights: List[float],
    num_models: int
) -> np.ndarray:
    """Compute ensemble predictions as weighted average.

    Args:
        aligned: Aligned DataFrame with pred_0, pred_1, ... columns.
        weights: List of weights for each model.
        num_models: Number of models.

    Returns:
        Ensemble predictions array.
    """
    pred_cols = [f"pred_{i}" for i in range(num_models)]
    preds = aligned[pred_cols].values  # shape (N, num_models)
    weights_arr = np.array(weights)
    return preds @ weights_arr


def grid_search_ensemble(
    aligned: pd.DataFrame,
    num_models: int,
    grid_steps: int,
    metric: str
) -> Tuple[List[float], float]:
    """Find optimal ensemble weights using grid search.

    Args:
        aligned: Aligned predictions DataFrame.
        num_models: Number of models to ensemble.
        grid_steps: Number of steps for grid (e.g., 11 => 0.0, 0.1, ..., 1.0).
        metric: Metric to optimize ("rmse", "mae", "mase").

    Returns:
        Tuple of (best_weights, best_metric_value).
    """
    print(f"[info] starting grid search with {grid_steps} steps per model...")

    y_true = aligned["y_true"].values
    weight_grid = np.linspace(0, 1, grid_steps)

    best_weights = None
    best_metric = float("inf")

    # Generate all weight combinations that sum to 1.0
    total_combinations = 0
    valid_combinations = 0

    for weights_tuple in product(weight_grid, repeat=num_models):
        total_combinations += 1
        weights_list = list(weights_tuple)
        weight_sum = sum(weights_list)

        # Only consider weights that sum to approximately 1.0
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            continue

        valid_combinations += 1
        y_pred = ensemble_predict(aligned, weights_list, num_models)

        if metric == "rmse":
            metric_val = compute_rmse(y_true, y_pred)
        elif metric == "mae":
            metric_val = compute_mae(y_true, y_pred)
        else:
            raise ValueError(f"metric {metric} not supported for grid search (use rmse or mae)")

        if metric_val < best_metric:
            best_metric = metric_val
            best_weights = weights_list

    print(f"[info] evaluated {valid_combinations}/{total_combinations} valid weight combinations")
    print(f"[info] best {metric}: {best_metric:.6f}")
    print(f"[info] best weights: {[f'{w:.3f}' for w in best_weights]}")

    return best_weights, best_metric


def optuna_ensemble(
    aligned: pd.DataFrame,
    num_models: int,
    n_trials: int,
    metric: str
) -> Tuple[List[float], float]:
    """Find optimal ensemble weights using Optuna.

    Args:
        aligned: Aligned predictions DataFrame.
        num_models: Number of models to ensemble.
        n_trials: Number of Optuna trials.
        metric: Metric to optimize ("rmse", "mae", "mase").

    Returns:
        Tuple of (best_weights, best_metric_value).
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is not installed. Install with: pip install optuna")

    print(f"[info] starting optuna optimization with {n_trials} trials...")

    y_true = aligned["y_true"].values

    def objective(trial: optuna.Trial) -> float:
        # Sample weights from Dirichlet-like distribution (sum to 1)
        weights = []
        remaining = 1.0
        for i in range(num_models - 1):
            w = trial.suggest_float(f"w{i}", 0.0, remaining)
            weights.append(w)
            remaining -= w
        weights.append(remaining)  # Last weight ensures sum = 1.0

        y_pred = ensemble_predict(aligned, weights, num_models)

        if metric == "rmse":
            return compute_rmse(y_true, y_pred)
        elif metric == "mae":
            return compute_mae(y_true, y_pred)
        else:
            raise ValueError(f"metric {metric} not supported")

    study = optuna.create_study(direction="minimize", study_name="ensemble_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    best_weights = []
    remaining = 1.0
    for i in range(num_models - 1):
        w = best_trial.params[f"w{i}"]
        best_weights.append(w)
        remaining -= w
    best_weights.append(remaining)

    best_metric = best_trial.value

    print(f"[info] best {metric}: {best_metric:.6f}")
    print(f"[info] best weights: {[f'{w:.3f}' for w in best_weights]}")

    return best_weights, best_metric


def main() -> None:
    """Main ensemble optimization function."""
    args = parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] output directory: {out_dir}")

    # Load predictions from all models
    tft_df = load_predictions(args.tft, "TFT")
    lgbm_df = load_predictions(args.lgbm, "LightGBM")
    bilstm_df = load_predictions(args.bilstm, "CNN-BiLSTM")

    # Collect valid models
    model_dfs = [df for df in [tft_df, lgbm_df, bilstm_df] if df is not None]
    model_names = []
    if tft_df is not None:
        model_names.append("TFT")
    if lgbm_df is not None:
        model_names.append("LightGBM")
    if bilstm_df is not None:
        model_names.append("CNN-BiLSTM")

    if len(model_dfs) < 2:
        raise ValueError("at least 2 models required for ensemble (provide --tft, --lgbm, and/or --bilstm)")

    print(f"[info] ensembling {len(model_dfs)} models: {', '.join(model_names)}")

    # Align predictions
    aligned = align_predictions(*model_dfs)

    # Optimize ensemble weights
    if args.method == "grid":
        best_weights, best_metric = grid_search_ensemble(
            aligned,
            num_models=len(model_dfs),
            grid_steps=args.grid_steps,
            metric=args.metric
        )
    elif args.method == "optuna":
        best_weights, best_metric = optuna_ensemble(
            aligned,
            num_models=len(model_dfs),
            n_trials=args.optuna_trials,
            metric=args.metric
        )
    else:
        raise ValueError(f"unknown optimization method: {args.method}")

    # Generate ensemble predictions
    y_pred_ensemble = ensemble_predict(aligned, best_weights, len(model_dfs))
    aligned["y_pred_ensemble"] = y_pred_ensemble

    # Save ensemble predictions
    ensemble_pred_path = out_dir / "predictions_val_ensemble.csv"
    aligned.to_csv(ensemble_pred_path, index=False)
    print(f"[info] saved ensemble predictions to {ensemble_pred_path}")

    # Save ensemble weights
    weights_dict = {
        "model_names": model_names,
        "weights": [float(w) for w in best_weights],
        "optimization_method": args.method,
        "optimization_metric": args.metric,
        f"best_{args.metric}": float(best_metric),
    }
    weights_path = out_dir / "ensemble_weights.json"
    weights_path.write_text(json.dumps(weights_dict, indent=2))
    print(f"[info] saved ensemble weights to {weights_path}")

    # Compute final metrics for all models + ensemble
    print("\n[info] final metrics comparison:")
    y_true = aligned["y_true"].values

    # Individual model metrics
    for i, name in enumerate(model_names):
        y_pred = aligned[f"pred_{i}"].values
        rmse_val = compute_rmse(y_true, y_pred)
        mae_val = compute_mae(y_true, y_pred)
        print(f"  {name:12s} - RMSE: {rmse_val:.6f}, MAE: {mae_val:.6f}")

    # Ensemble metrics
    rmse_ens = compute_rmse(y_true, y_pred_ensemble)
    mae_ens = compute_mae(y_true, y_pred_ensemble)
    print(f"  {'Ensemble':12s} - RMSE: {rmse_ens:.6f}, MAE: {mae_ens:.6f}")

    # Save metrics summary
    metrics_summary = {
        "individual_models": {
            name: {
                "rmse": float(compute_rmse(y_true, aligned[f"pred_{i}"].values)),
                "mae": float(compute_mae(y_true, aligned[f"pred_{i}"].values)),
            }
            for i, name in enumerate(model_names)
        },
        "ensemble": {
            "rmse": float(rmse_ens),
            "mae": float(mae_ens),
        }
    }
    metrics_path = out_dir / "metrics_ensemble.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2))
    print(f"[info] saved metrics summary to {metrics_path}")

    print("\n[info] ensemble optimization completed successfully")


if __name__ == "__main__":
    main()
