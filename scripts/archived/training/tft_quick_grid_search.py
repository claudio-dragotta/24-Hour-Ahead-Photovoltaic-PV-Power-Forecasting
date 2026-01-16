"""ARCHIVE: tft_quick_grid_search.py

Original: scripts/training/tft_quick_grid_search.py
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import json
import pandas as pd

from scripts.training.tft_grid_search import run_single_config


def get_quick_param_grid():
    return {
        "hidden_size": [32, 48, 64],
        "attention_heads": [2, 4],
        "dropout": [0.3, 0.4],
        "learning_rate": [1e-4, 3e-4],
        "hidden_continuous_size": [32],
    }


def main_quick():
    parser = argparse.ArgumentParser(description="Quick grid search for TFT hyperparameters")
    parser.add_argument("--output-dir", type=str, default="outputs_grid_search_quick")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    args = parser.parse_args()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    param_grid = get_quick_param_grid()
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    results_file = base_output_dir / "grid_search_results.json"
    completed_runs = []
    if results_file.exists():
        with open(results_file) as f:
            completed_runs = json.load(f)
    all_results = completed_runs.copy()
    for run_id, combination in enumerate(all_combinations, start=len(completed_runs) + 1):
        config = {name: value for name, value in zip(param_names, combination)}
        if any(r.get("config") == config for r in completed_runs):
            continue
        result = run_single_config(config, run_id, base_output_dir, args.max_epochs, args.early_stopping_patience)
        all_results.append(result)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        df_results = []
        for r in all_results:
            row = {"run_id": r["run_id"], "rmse": r["rmse"], "mase": r["mase"], "status": r["status"]}
            row.update(r["config"])
            df_results.append(row)
        pd.DataFrame(df_results).to_csv(base_output_dir / "grid_search_results.csv", index=False)


if __name__ == "__main__":
    main_quick()
