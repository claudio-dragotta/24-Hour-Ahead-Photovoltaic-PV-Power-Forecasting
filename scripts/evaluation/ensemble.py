"""Ensemble optimization script for combining multiple PV forecasting models.

This module implements weighted ensemble combining predictions from:
- Temporal Fusion Transformer (TFT)
- LightGBM multi-horizon model
- CNN-BiLSTM

The ensemble weights are optimized using grid search or Optuna to minimize RMSE.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations, product
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
        description="ensemble optimization for PV forecasting models (supports up to 3 models)"
    )

    # Model prediction paths - VALIDATION (for weight optimization)
    ap.add_argument("--lgbm-val", type=str, default=None, help="path to LightGBM VALIDATION predictions CSV")
    ap.add_argument("--cnn-val", type=str, default=None, help="path to CNN-BiLSTM VALIDATION predictions CSV")
    ap.add_argument("--tft-val", type=str, default=None, help="path to TFT VALIDATION predictions CSV")
    
    # Model prediction paths - TEST (for final evaluation)
    ap.add_argument("--lgbm-test", type=str, default=None, help="path to LightGBM TEST predictions CSV")
    ap.add_argument("--cnn-test", type=str, default=None, help="path to CNN-BiLSTM TEST predictions CSV")
    ap.add_argument("--tft-test", type=str, default=None, help="path to TFT TEST predictions CSV")
    
    # Legacy arguments (for backward compatibility)
    ap.add_argument("--lgbm", type=str, default=None, help="[DEPRECATED] use --lgbm-val and --lgbm-test")
    ap.add_argument("--cnn", type=str, default=None, help="[DEPRECATED] use --cnn-val and --cnn-test")
    ap.add_argument("--tft", type=str, default=None, help="[DEPRECATED] use --tft-val and --tft-test")
    
    ap.add_argument("--outdir", type=str, default="outputs_ensemble", help="output directory for ensemble results")
    ap.add_argument(
        "--method",
        type=str,
        choices=["grid", "optuna", "exhaustive"],
        default="grid",
        help="optimization method: grid, optuna, or exhaustive (tries ALL subsets)",
    )
    ap.add_argument(
        "--exhaustive-min-models",
        type=int,
        default=1,
        help="minimum number of models for exhaustive search (default: 1, includes single models)",
    )
    ap.add_argument(
        "--exhaustive-max-models",
        type=int,
        default=3,
        help="maximum number of models for exhaustive search (default: 3)",
    )
    ap.add_argument(
        "--grid-steps", type=int, default=11, help="number of steps for grid search (default: 11 => 0.0, 0.1, ..., 1.0)"
    )
    ap.add_argument("--optuna-trials", type=int, default=100, help="number of trials for optuna optimization")
    ap.add_argument(
        "--metric", type=str, choices=["rmse", "mae", "mase"], default="rmse", help="metric to optimize (default: rmse)"
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

        base = base.merge(df_merge, on=["origin_timestamp_utc", "forecast_timestamp_utc", "horizon_h"], how="inner")

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


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, train_series: np.ndarray, m: int = 24) -> float:
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


def ensemble_predict(aligned: pd.DataFrame, weights: List[float], num_models: int) -> np.ndarray:
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
    aligned: pd.DataFrame, num_models: int, grid_steps: int, metric: str
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


def optuna_ensemble(aligned: pd.DataFrame, num_models: int, n_trials: int, metric: str) -> Tuple[List[float], float]:
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


def exhaustive_search_ensemble(
    all_model_dfs: List[pd.DataFrame],
    all_model_names: List[str],
    min_models: int,
    max_models: int,
    metric: str,
    grid_steps: int = 11,
) -> Tuple[List[int], List[float], float, List[str]]:
    """Try ALL possible combinations of models and find the best ensemble.

    This function:
    1. Generates all subsets of models (from min_models to max_models)
    2. For each subset, optimizes weights using grid search
    3. Returns the absolute best combination

    Args:
        all_model_dfs: List of all model DataFrames.
        all_model_names: List of all model names.
        min_models: Minimum number of models to try.
        max_models: Maximum number of models to try.
        metric: Metric to optimize ("rmse", "mae").
        grid_steps: Number of steps for grid search per combination.

    Returns:
        Tuple of (best_indices, best_weights, best_metric, best_model_names).
    """
    print(f"\n{'='*60}")
    print(f"EXHAUSTIVE ENSEMBLE SEARCH")
    print(f"{'='*60}")
    print(f"Total models available: {len(all_model_dfs)}")
    print(f"Model names: {', '.join(all_model_names)}")
    print(f"Testing combinations: {min_models} to {max_models} models")
    print(f"Optimization metric: {metric}")
    print(f"{'='*60}\n")

    best_overall_metric = float("inf")
    best_overall_indices = None
    best_overall_weights = None
    best_overall_names = None

    total_combinations = sum(
        len(list(combinations(range(len(all_model_dfs)), k))) for k in range(min_models, max_models + 1)
    )

    print(f"[info] total subset combinations to test: {total_combinations}\n")

    combination_results = []
    current_combo = 0

    for num_models in range(min_models, max_models + 1):
        print(f"\n{'='*60}")
        print(f"Testing {num_models}-model combinations")
        print(f"{'='*60}")

        for indices in combinations(range(len(all_model_dfs)), num_models):
            current_combo += 1
            subset_names = [all_model_names[i] for i in indices]
            subset_dfs = [all_model_dfs[i] for i in indices]

            print(f"\n[{current_combo}/{total_combinations}] Combination: {', '.join(subset_names)}")

            # Align predictions for this subset
            aligned = align_predictions(*subset_dfs)

            # Optimize weights using grid search
            weights, metric_val = grid_search_ensemble(
                aligned, num_models=num_models, grid_steps=grid_steps, metric=metric
            )

            # Track result
            combination_results.append(
                {"combination": subset_names, "num_models": num_models, "weights": weights, metric: metric_val}
            )

            # Update best if better
            if metric_val < best_overall_metric:
                best_overall_metric = metric_val
                best_overall_indices = list(indices)
                best_overall_weights = weights
                best_overall_names = subset_names
                print(f"  → NEW BEST! {metric}={metric_val:.6f}")

    print(f"\n{'='*60}")
    print(f"EXHAUSTIVE SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best combination: {', '.join(best_overall_names)}")
    print(f"Best {metric}: {best_overall_metric:.6f}")
    print(f"Best weights: {[f'{w:.3f}' for w in best_overall_weights]}")
    print(f"{'='*60}\n")

    return best_overall_indices, best_overall_weights, best_overall_metric, best_overall_names


def main() -> None:
    """Main ensemble optimization function.
    
    IMPORTANT: To prevent data leakage, this function:
    1. Uses VALIDATION predictions to optimize ensemble weights
    2. Applies optimized weights to TEST predictions for final evaluation
    """
    args = parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] output directory: {out_dir}")

    # Handle backward compatibility with legacy arguments
    lgbm_val = args.lgbm_val or args.lgbm
    cnn_val = args.cnn_val or args.cnn
    tft_val = args.tft_val or args.tft
    lgbm_test = args.lgbm_test or args.lgbm
    cnn_test = args.cnn_test or args.cnn
    tft_test = args.tft_test or args.tft
    
    # Warn if using legacy arguments (potential data leakage)
    if args.lgbm or args.cnn or args.tft:
        print("[WARNING] Using legacy arguments (--lgbm, --cnn, --tft).")
        print("[WARNING] This may cause DATA LEAKAGE if same file is used for optimization AND evaluation!")
        print("[WARNING] Use separate --*-val and --*-test arguments for proper evaluation.")

    # Load VALIDATION predictions (for weight optimization)
    print("\n[info] === Loading VALIDATION predictions (for weight optimization) ===")
    val_model_dfs = []
    val_model_names = []

    lgbm_val_df = load_predictions(lgbm_val, "LightGBM (val)")
    if lgbm_val_df is not None:
        val_model_dfs.append(lgbm_val_df)
        val_model_names.append("LightGBM")

    cnn_val_df = load_predictions(cnn_val, "CNN-BiLSTM (val)")
    if cnn_val_df is not None:
        val_model_dfs.append(cnn_val_df)
        val_model_names.append("CNN-BiLSTM")

    tft_val_df = load_predictions(tft_val, "TFT (val)")
    if tft_val_df is not None:
        val_model_dfs.append(tft_val_df)
        val_model_names.append("TFT")

    if len(val_model_dfs) < 2:
        raise ValueError(
            "at least 2 models required for ensemble. Provide 2-3 models using:\n"
            "  --lgbm-val PATH  --lgbm-test PATH\n"
            "  --cnn-val PATH   --cnn-test PATH\n"
            "  --tft-val PATH   --tft-test PATH"
        )

    print(f"[info] available models: {len(val_model_dfs)}")

    # Load TEST predictions (for final evaluation)
    print("\n[info] === Loading TEST predictions (for final evaluation) ===")
    test_model_dfs = []
    test_model_names = []

    lgbm_test_df = load_predictions(lgbm_test, "LightGBM (test)")
    if lgbm_test_df is not None:
        test_model_dfs.append(lgbm_test_df)
        test_model_names.append("LightGBM")

    cnn_test_df = load_predictions(cnn_test, "CNN-BiLSTM (test)")
    if cnn_test_df is not None:
        test_model_dfs.append(cnn_test_df)
        test_model_names.append("CNN-BiLSTM")

    tft_test_df = load_predictions(tft_test, "TFT (test)")
    if tft_test_df is not None:
        test_model_dfs.append(tft_test_df)
        test_model_names.append("TFT")

    # Verify same models for val and test
    if val_model_names != test_model_names:
        raise ValueError(
            f"Model mismatch between val ({val_model_names}) and test ({test_model_names}).\n"
            "Ensure same models are provided for both validation and test sets."
        )

    # Optimize ensemble weights on VALIDATION set
    print(f"\n[info] {'='*60}")
    print(f"[info] OPTIMIZING WEIGHTS ON VALIDATION SET (no data leakage)")
    print(f"[info] {'='*60}")

    if args.method == "exhaustive":
        # Exhaustive search: try ALL combinations of models
        best_indices, best_weights, best_metric, selected_names = exhaustive_search_ensemble(
            all_model_dfs=val_model_dfs,
            all_model_names=val_model_names,
            min_models=args.exhaustive_min_models,
            max_models=min(args.exhaustive_max_models, len(val_model_dfs)),
            metric=args.metric,
            grid_steps=args.grid_steps,
        )

        # Update model lists to only include selected models
        val_model_dfs = [val_model_dfs[i] for i in best_indices]
        test_model_dfs = [test_model_dfs[i] for i in best_indices]
        model_names = selected_names

        print(f"[info] selected {len(model_names)} models for final ensemble: {', '.join(model_names)}")

        # Align predictions for final selected models
        val_aligned = align_predictions(*val_model_dfs)
        test_aligned = align_predictions(*test_model_dfs)

    else:
        # Standard methods: use all provided models
        model_names = val_model_names
        print(f"[info] ensembling {len(val_model_dfs)} models: {', '.join(model_names)}")

        # Align predictions
        val_aligned = align_predictions(*val_model_dfs)
        test_aligned = align_predictions(*test_model_dfs)

        if args.method == "grid":
            best_weights, best_metric = grid_search_ensemble(
                val_aligned, num_models=len(val_model_dfs), grid_steps=args.grid_steps, metric=args.metric
            )
        elif args.method == "optuna":
            best_weights, best_metric = optuna_ensemble(
                val_aligned, num_models=len(val_model_dfs), n_trials=args.optuna_trials, metric=args.metric
            )
        else:
            raise ValueError(f"unknown optimization method: {args.method}")

    # Save ensemble weights
    weights_dict = {
        "model_names": model_names,
        "weights": [float(w) for w in best_weights],
        "optimization_method": args.method,
        "optimization_metric": args.metric,
        f"best_val_{args.metric}": float(best_metric),
        "note": "Weights optimized on VALIDATION set to prevent data leakage"
    }
    weights_path = out_dir / "ensemble_weights.json"
    weights_path.write_text(json.dumps(weights_dict, indent=2))
    print(f"[info] saved ensemble weights to {weights_path}")

    # Generate ensemble predictions on VALIDATION set
    val_aligned["y_pred_ensemble"] = ensemble_predict(val_aligned, best_weights, len(model_names))
    val_pred_path = out_dir / "predictions_val_ensemble.csv"
    val_aligned.to_csv(val_pred_path, index=False)
    print(f"[info] saved validation ensemble predictions to {val_pred_path}")

    # Apply weights to TEST set for final evaluation
    print(f"\n[info] {'='*60}")
    print(f"[info] APPLYING WEIGHTS TO TEST SET (final evaluation)")
    print(f"[info] {'='*60}")

    y_pred_ensemble_test = ensemble_predict(test_aligned, best_weights, len(model_names))
    test_aligned["y_pred_ensemble"] = y_pred_ensemble_test

    # Save test ensemble predictions
    test_pred_path = out_dir / "predictions_test_ensemble.csv"
    test_aligned.to_csv(test_pred_path, index=False)
    print(f"[info] saved test ensemble predictions to {test_pred_path}")

    # Compute final metrics on TEST set
    print("\n[info] === FINAL METRICS ON TEST SET ===")
    y_true_test = test_aligned["y_true"].values

    # Individual model metrics
    for i, name in enumerate(model_names):
        y_pred = test_aligned[f"pred_{i}"].values
        rmse_val = compute_rmse(y_true_test, y_pred)
        mae_val = compute_mae(y_true_test, y_pred)
        print(f"  {name:12s} - RMSE: {rmse_val:.6f}, MAE: {mae_val:.6f}")

    # Ensemble metrics
    rmse_ens = compute_rmse(y_true_test, y_pred_ensemble_test)
    mae_ens = compute_mae(y_true_test, y_pred_ensemble_test)
    print(f"  {'Ensemble':12s} - RMSE: {rmse_ens:.6f}, MAE: {mae_ens:.6f}")

    # Save metrics summary
    metrics_summary = {
        "validation_set": {
            f"ensemble_{args.metric}": float(best_metric),
        },
        "test_set": {
            "individual_models": {
                name: {
                    "rmse": float(compute_rmse(y_true_test, test_aligned[f"pred_{i}"].values)),
                    "mae": float(compute_mae(y_true_test, test_aligned[f"pred_{i}"].values)),
                }
                for i, name in enumerate(model_names)
            },
            "ensemble": {
                "rmse": float(rmse_ens),
                "mae": float(mae_ens),
            },
        },
        "note": "Weights optimized on validation set, metrics computed on test set (no data leakage)"
    }
    metrics_path = out_dir / "metrics_ensemble.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2))
    print(f"[info] saved metrics summary to {metrics_path}")

    print("\n[info] ensemble optimization completed successfully (no data leakage!)")


if __name__ == "__main__":
    main()
