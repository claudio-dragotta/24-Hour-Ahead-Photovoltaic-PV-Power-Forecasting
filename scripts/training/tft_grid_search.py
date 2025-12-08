"""Grid Search for TFT hyperparameter optimization.

Automatically tests all combinations of hyperparameters to find the best TFT configuration.
Results are saved incrementally to avoid losing progress if interrupted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List

import pandas as pd

from scripts.training.train_tft import main as train_tft_main


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for grid search."""
    ap = argparse.ArgumentParser(description="Grid search for TFT hyperparameters")
    ap.add_argument(
        "--output-dir",
        type=str,
        default="outputs_grid_search",
        help="Base directory for grid search results",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum epochs per configuration (lower for faster search)",
    )
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Early stopping patience",
    )
    return ap.parse_args()


def get_param_grid() -> Dict[str, List]:
    """Define hyperparameter search space.
    
    Returns:
        Dictionary with parameter names and their possible values.
    """
    return {
        "hidden_size": [32, 64, 128],  # Model capacity
        "attention_heads": [2, 4, 8],  # Number of attention heads
        "dropout": [0.2, 0.3, 0.4],  # Regularization strength
        "learning_rate": [1e-4, 3e-4, 1e-3],  # Learning rate
        "hidden_continuous_size": [16, 32, 64],  # Continuous variable embedding size
    }


def run_single_config(
    config: Dict,
    run_id: int,
    base_output_dir: Path,
    max_epochs: int,
    early_stopping_patience: int,
) -> Dict:
    """Train TFT with a single hyperparameter configuration.
    
    Args:
        config: Dictionary with hyperparameter values
        run_id: Unique ID for this run
        base_output_dir: Base directory for all grid search outputs
        max_epochs: Maximum training epochs
        early_stopping_patience: Early stopping patience
        
    Returns:
        Dictionary with results (config + metrics)
    """
    import sys
    from io import StringIO
    
    # Create output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"run_{run_id:03d}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"RUN {run_id} - Testing configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  Output: {run_dir}")
    print(f"{'='*80}\n")
    
    # Prepare arguments for train_tft
    sys.argv = [
        "train_tft.py",
        "--outdir", str(run_dir),
        "--hidden-size", str(config["hidden_size"]),
        "--attention-heads", str(config["attention_heads"]),
        "--dropout", str(config["dropout"]),
        "--learning-rate", str(config["learning_rate"]),
        "--hidden-continuous-size", str(config["hidden_continuous_size"]),
        "--epochs", str(max_epochs),
        "--early-stopping-patience", str(early_stopping_patience),
        "--lr-patience", "5",  # Fixed for all runs
        "--dayweight-gamma", "1.5",  # Fixed: solar weighting
        "--dayweight-min", "0.1",
        "--metrics-zenith-max", "90.0",  # Filter nighttime from metrics
    ]
    
    # Run training
    try:
        train_tft_main()
        
        # Load results
        metrics_path = run_dir / "metrics_summary.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            result = {
                "run_id": run_id,
                "timestamp": timestamp,
                "config": config,
                "rmse": metrics.get("rmse_model_avg"),
                "mase": metrics.get("mase_model_avg"),
                "status": "success",
            }
        else:
            result = {
                "run_id": run_id,
                "timestamp": timestamp,
                "config": config,
                "rmse": None,
                "mase": None,
                "status": "failed_no_metrics",
            }
    except Exception as e:
        print(f"❌ Run {run_id} failed with error: {e}")
        result = {
            "run_id": run_id,
            "timestamp": timestamp,
            "config": config,
            "rmse": None,
            "mase": None,
            "status": f"failed: {str(e)[:100]}",
        }
    
    return result


def main():
    """Run grid search over TFT hyperparameters."""
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get parameter grid
    param_grid = get_param_grid()
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    print(f"\n{'='*80}")
    print(f"TFT GRID SEARCH")
    print(f"{'='*80}")
    print(f"Parameter grid:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")
    print(f"\nTotal combinations: {len(all_combinations)}")
    print(f"Estimated time: ~{len(all_combinations) * 20} minutes (assuming 20 min/run)")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*80}\n")
    
    # Results storage
    results_file = base_output_dir / "grid_search_results.json"
    results_csv = base_output_dir / "grid_search_results.csv"
    
    # Load existing results if resuming
    completed_runs = []
    if results_file.exists():
        with open(results_file) as f:
            completed_runs = json.load(f)
        print(f"📂 Resuming from previous run: {len(completed_runs)} configs already tested\n")
    
    # Run grid search
    all_results = completed_runs.copy()
    
    for run_id, combination in enumerate(all_combinations, start=len(completed_runs) + 1):
        # Create config dict
        config = {name: value for name, value in zip(param_names, combination)}
        
        # Check if already tested
        if any(r["config"] == config for r in completed_runs):
            print(f"⏭️  Skipping run {run_id} (already completed)")
            continue
        
        # Run single configuration
        result = run_single_config(
            config,
            run_id,
            base_output_dir,
            args.max_epochs,
            args.early_stopping_patience,
        )
        
        all_results.append(result)
        
        # Save results incrementally (don't lose progress)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Also save as CSV for easy viewing
        df_results = []
        for r in all_results:
            row = {
                "run_id": r["run_id"],
                "rmse": r["rmse"],
                "mase": r["mase"],
                "status": r["status"],
            }
            row.update(r["config"])
            df_results.append(row)
        
        pd.DataFrame(df_results).to_csv(results_csv, index=False)
        
        print(f"\n✅ Run {run_id}/{len(all_combinations)} completed")
        if result["rmse"] is not None:
            print(f"   RMSE: {result['rmse']:.4f}, MASE: {result['mase']:.4f}")
        print(f"   Results saved to: {results_file}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"Total runs: {len(all_results)}")
    
    # Find best configuration
    successful_runs = [r for r in all_results if r["rmse"] is not None]
    if successful_runs:
        best_run = min(successful_runs, key=lambda x: x["rmse"])
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   RMSE: {best_run['rmse']:.4f}")
        print(f"   MASE: {best_run['mase']:.4f}")
        print(f"   Config:")
        for key, value in best_run["config"].items():
            print(f"     {key}: {value}")
        print(f"\n   Output directory: {base_output_dir / f\"run_{best_run['run_id']:03d}*\"}")
        
        # Save best config separately
        best_config_file = base_output_dir / "best_config.json"
        with open(best_config_file, "w") as f:
            json.dump(
                {
                    "rmse": best_run["rmse"],
                    "mase": best_run["mase"],
                    "config": best_run["config"],
                    "run_id": best_run["run_id"],
                },
                f,
                indent=2,
            )
        print(f"\n   Best config saved to: {best_config_file}")
    else:
        print("\n❌ No successful runs found")
    
    print(f"\n📊 Full results: {results_csv}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
