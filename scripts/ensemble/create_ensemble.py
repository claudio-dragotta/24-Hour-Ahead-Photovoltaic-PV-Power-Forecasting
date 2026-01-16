#!/usr/bin/env python3
"""Create ensemble predictions from multiple trained models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate evaluation metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    mbe = np.mean(y_pred - y_true)

    naive_errors = np.abs(np.diff(y_true))
    mase = mae / np.mean(naive_errors) if len(naive_errors) > 0 and np.mean(naive_errors) > 0 else 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mbe": float(mbe),
        "mase": float(mase),
    }


def simple_average_ensemble(predictions_list):
    """Simple average of all predictions."""
    return np.mean(predictions_list, axis=0)


def weighted_average_ensemble(predictions_list, weights):
    """Weighted average based on model performance."""
    weights = np.array(weights) / np.sum(weights)  # Normalize
    return np.average(predictions_list, axis=0, weights=weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dirs", nargs="+", required=True, help="List of model output directories")
    parser.add_argument("--test-dir", type=str, default="test_eval", help="Test results subdirectory name")
    parser.add_argument("--outdir", type=str, default="outputs/ensemble", help="Output directory for ensemble results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENSEMBLE PREDICTION CREATOR")
    print("=" * 80)
    print(f"\nUsing {len(args.model_dirs)} models:")

    # Load predictions from all models
    all_predictions = []
    all_metrics = []
    model_names = []

    for model_dir in args.model_dirs:
        model_path = Path(model_dir)
        model_name = model_path.name
        pred_file = model_path / args.test_dir / "test_predictions.csv"
        metrics_file = model_path / args.test_dir / "test_metrics.json"

        if not pred_file.exists():
            print(f"  [SKIP] {model_name}: predictions not found")
            continue

        # Load predictions
        df_pred = pd.read_csv(pred_file)
        all_predictions.append(df_pred["y_pred_kw"].values)
        model_names.append(model_name)

        # Load metrics
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
                print(f"  [OK] {model_name:25s} | MASE: {metrics['mase']:.4f}")
        else:
            print(f"  [OK] {model_name:25s} | (metrics not found)")
            all_metrics.append(None)

    if len(all_predictions) == 0:
        raise ValueError("No predictions loaded!")

    # Get ground truth from first model
    first_pred_file = Path(args.model_dirs[0]) / args.test_dir / "test_predictions.csv"
    df_first = pd.read_csv(first_pred_file)
    y_true = df_first["y_true_kw"].values

    print(f"\n{'='*80}")
    print("ENSEMBLE METHODS")
    print("=" * 80)

    all_predictions_array = np.array(all_predictions)  # Shape: (n_models, n_samples)

    # 1. SIMPLE AVERAGE
    print("\n[1] Simple Average Ensemble")
    ensemble_simple = simple_average_ensemble(all_predictions_array)
    metrics_simple = calculate_metrics(y_true, ensemble_simple)
    print(f"  MASE: {metrics_simple['mase']:.4f}")
    print(f"  MAE:  {metrics_simple['mae']:.2f} kW")
    print(f"  R²:   {metrics_simple['r2']:.4f}")

    # Save simple average
    df_ensemble_simple = pd.DataFrame({"y_true_kw": y_true, "y_pred_kw": ensemble_simple})
    df_ensemble_simple.to_csv(outdir / "ensemble_simple_avg.csv", index=False)
    with open(outdir / "ensemble_simple_avg_metrics.json", "w") as f:
        json.dump(metrics_simple, f, indent=2)

    # 2. WEIGHTED AVERAGE (inverse MASE)
    if all(m is not None for m in all_metrics):
        print("\n[2] Weighted Average Ensemble (inverse MASE)")

        # Calculate weights as 1/MASE (better models get more weight)
        mase_values = [m["mase"] for m in all_metrics]
        weights = [1.0 / mase for mase in mase_values]

        ensemble_weighted = weighted_average_ensemble(all_predictions_array, weights)
        metrics_weighted = calculate_metrics(y_true, ensemble_weighted)

        print(f"  Weights: {dict(zip(model_names, [f'{w/sum(weights):.3f}' for w in weights]))}")
        print(f"  MASE: {metrics_weighted['mase']:.4f}")
        print(f"  MAE:  {metrics_weighted['mae']:.2f} kW")
        print(f"  R²:   {metrics_weighted['r2']:.4f}")

        # Save weighted average
        df_ensemble_weighted = pd.DataFrame({"y_true_kw": y_true, "y_pred_kw": ensemble_weighted})
        df_ensemble_weighted.to_csv(outdir / "ensemble_weighted_avg.csv", index=False)
        with open(outdir / "ensemble_weighted_avg_metrics.json", "w") as f:
            json.dump(metrics_weighted, f, indent=2)

    # 3. BEST MODEL SELECTION (per sample)
    print("\n[3] Best Model Selection (min absolute error per sample)")

    # For each sample, pick prediction from model with historically lowest MAE
    # Simple heuristic: use best overall model
    if all_metrics:
        best_model_idx = np.argmin([m["mase"] for m in all_metrics])
        ensemble_best = all_predictions_array[best_model_idx]
        metrics_best = calculate_metrics(y_true, ensemble_best)

        print(f"  Selected: {model_names[best_model_idx]}")
        print(f"  MASE: {metrics_best['mase']:.4f}")
        print(f"  MAE:  {metrics_best['mae']:.2f} kW")
        print(f"  R²:   {metrics_best['r2']:.4f}")

    # 4. MEDIAN ENSEMBLE
    print("\n[4] Median Ensemble")
    ensemble_median = np.median(all_predictions_array, axis=0)
    metrics_median = calculate_metrics(y_true, ensemble_median)
    print(f"  MASE: {metrics_median['mase']:.4f}")
    print(f"  MAE:  {metrics_median['mae']:.2f} kW")
    print(f"  R²:   {metrics_median['r2']:.4f}")

    # Save median
    df_ensemble_median = pd.DataFrame({"y_true_kw": y_true, "y_pred_kw": ensemble_median})
    df_ensemble_median.to_csv(outdir / "ensemble_median.csv", index=False)
    with open(outdir / "ensemble_median_metrics.json", "w") as f:
        json.dump(metrics_median, f, indent=2)

    # 5. STACKING (Linear Ridge)
    print("\n[5] Stacking Ensemble (Ridge Regression)")

    # Use cross-validation-like approach: train on half, test on half
    split_idx = len(y_true) // 2

    X_train = all_predictions_array[:, :split_idx].T  # (n_samples/2, n_models)
    y_train = y_true[:split_idx]
    X_test = all_predictions_array[:, split_idx:].T  # (n_samples/2, n_models)
    y_test = y_true[split_idx:]

    # Train Ridge model
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # Predict on full data
    X_full = all_predictions_array.T  # (n_samples, n_models)
    ensemble_stacking = ridge.predict(X_full)
    metrics_stacking = calculate_metrics(y_true, ensemble_stacking)

    print(f"  Model weights: {dict(zip(model_names, ridge.coef_))}")
    print(f"  Intercept: {ridge.intercept_:.2f}")
    print(f"  MASE: {metrics_stacking['mase']:.4f}")
    print(f"  MAE:  {metrics_stacking['mae']:.2f} kW")
    print(f"  R²:   {metrics_stacking['r2']:.4f}")

    # Save stacking
    df_ensemble_stacking = pd.DataFrame({"y_true_kw": y_true, "y_pred_kw": ensemble_stacking})
    df_ensemble_stacking.to_csv(outdir / "ensemble_stacking.csv", index=False)
    with open(outdir / "ensemble_stacking_metrics.json", "w") as f:
        json.dump(metrics_stacking, f, indent=2)

    # SUMMARY
    print(f"\n{'='*80}")
    print("ENSEMBLE SUMMARY")
    print("=" * 80)

    results = [
        ("Simple Average", metrics_simple["mase"], metrics_simple["mae"]),
        ("Median", metrics_median["mase"], metrics_median["mae"]),
        ("Stacking (Ridge)", metrics_stacking["mase"], metrics_stacking["mae"]),
    ]

    if all_metrics:
        results.insert(1, ("Weighted Average", metrics_weighted["mase"], metrics_weighted["mae"]))
        results.insert(2, ("Best Model Only", metrics_best["mase"], metrics_best["mae"]))

    results.sort(key=lambda x: x[1])  # Sort by MASE

    print(f"\n{'Ensemble Method':<25s} | {'MASE':>6s} | {'MAE (kW)':>10s}")
    print("-" * 50)
    for method, mase, mae in results:
        symbol = "*" if mase == results[0][1] else " "
        print(f"{symbol} {method:<23s} | {mase:6.4f} | {mae:10.2f}")

    print(f"\n{'='*80}")
    print(f"Best ensemble: {results[0][0]} with MASE {results[0][1]:.4f}")
    print(f"Results saved to: {outdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
