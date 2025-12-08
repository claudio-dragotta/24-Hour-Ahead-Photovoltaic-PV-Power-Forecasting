"""Quick Grid Search for TFT - Tests only promising hyperparameter combinations.

Reduced search space for faster optimization (~2-3 hours instead of 80).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.tft_grid_search import main, run_single_config
import argparse
import json
from datetime import datetime
from itertools import product
import pandas as pd


def get_quick_param_grid():
    """Define reduced hyperparameter search space focused on best regions.
    
    Based on baseline results (hidden=32 works best), we test around that region.
    Total combinations: 3 * 2 * 2 * 2 = 24 configs (~8 hours)
    """
    return {
        "hidden_size": [32, 48, 64],  # Focus around baseline (32)
        "attention_heads": [2, 4],  # Baseline uses 2
        "dropout": [0.3, 0.4],  # Baseline uses 0.4 (higher is better)
        "learning_rate": [1e-4, 3e-4],  # Test around baseline 1e-4
        "hidden_continuous_size": [32],  # Fix this (less important)
    }


def main_quick():
    """Run quick grid search."""
    parser = argparse.ArgumentParser(description="Quick grid search for TFT hyperparameters")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_grid_search_quick",
        help="Base directory for grid search results",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum epochs per configuration",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Early stopping patience",
    )
    args = parser.parse_args()
    
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get reduced parameter grid
    param_grid = get_quick_param_grid()
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    print(f"\n{'='*80}")
    print(f"TFT QUICK GRID SEARCH (Reduced Space)")
    print(f"{'='*80}")
    print(f"Parameter grid:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")
    print(f"\nTotal combinations: {len(all_combinations)}")
    print(f"Estimated time: ~{len(all_combinations) * 20} minutes = {len(all_combinations) * 20 / 60:.1f} hours")
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
        print(f"📂 Resuming: {len(completed_runs)} configs already tested\n")
    
    # Run grid search
    all_results = completed_runs.copy()
    
    for run_id, combination in enumerate(all_combinations, start=len(completed_runs) + 1):
        # Create config dict
        config = {name: value for name, value in zip(param_names, combination)}
        
        # Check if already tested
        if any(r.get("config") == config for r in completed_runs):
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
        
        # Save results incrementally
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Save as CSV
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
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"QUICK GRID SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"Total runs: {len(all_results)}")
    
    # Find best configuration
    successful_runs = [r for r in all_results if r.get("rmse") is not None]
    if successful_runs:
        best_run = min(successful_runs, key=lambda x: x["rmse"])
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   RMSE: {best_run['rmse']:.4f}")
        print(f"   MASE: {best_run['mase']:.4f}")
        print(f"   Config:")
        for key, value in best_run["config"].items():
            print(f"     {key}: {value}")
        
        # Compare with baseline
        baseline_rmse = 3.7060
        improvement = ((baseline_rmse - best_run["rmse"]) / baseline_rmse) * 100
        if improvement > 0:
            print(f"\n   ✨ Improvement over baseline: {improvement:.2f}%")
        else:
            print(f"\n   ⚠️  Worse than baseline by {abs(improvement):.2f}%")
        
        # Save best config
        best_config_file = base_output_dir / "best_config.json"
        with open(best_config_file, "w") as f:
            json.dump(
                {
                    "rmse": best_run["rmse"],
                    "mase": best_run["mase"],
                    "config": best_run["config"],
                    "run_id": best_run["run_id"],
                    "baseline_rmse": baseline_rmse,
                    "improvement_pct": improvement,
                },
                f,
                indent=2,
            )
        print(f"\n   Best config saved to: {best_config_file}")
    
    print(f"\n📊 Full results: {results_csv}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main_quick()
