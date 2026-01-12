"""Grid Search for TFT hyperparameter optimization.

Runs combinations of hyperparameters, trains TFT for each config
and records results incrementally so the search can be resumed.
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
    ap = argparse.ArgumentParser(description="Grid search for TFT hyperparameters")
    ap.add_argument("--output-dir", type=str, default="outputs_grid_search")
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--early-stopping-patience", type=int, default=8)
    return ap.parse_args()


def get_param_grid() -> Dict[str, List]:
    return {
        "hidden_size": [32, 64, 128],
        "attention_heads": [2, 4, 8],
        "dropout": [0.2, 0.3, 0.4],
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "hidden_continuous_size": [16, 32, 64],
    }


def run_single_config(
    config: Dict,
    run_id: int,
    base_output_dir: Path,
    max_epochs: int,
    early_stopping_patience: int,
) -> Dict:
    import sys

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"run_{run_id:03d}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"RUN {run_id} - Testing configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  Output: {run_dir}")
    print(f"{'='*80}\n")

    sys.argv = [
        "train_tft.py",
        "--outdir",
        str(run_dir),
        "--hidden-size",
        str(config["hidden_size"]),
        "--attention-heads",
        str(config["attention_heads"]),
        "--dropout",
        str(config["dropout"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--hidden-continuous-size",
        str(config["hidden_continuous_size"]),
        "--epochs",
        str(max_epochs),
        "--early-stopping-patience",
        str(early_stopping_patience),
        "--lr-patience",
        "5",
    ]

    try:
        train_tft_main()

        metrics_path = run_dir / "metrics_summary.json"
        if metrics_path.exists():
            with open(metrics_path) as fh:
                metrics = json.load(fh)
            return {
                "run_id": run_id,
                "timestamp": timestamp,
                "config": config,
                "rmse": metrics.get("rmse_model_avg"),
                "mase": metrics.get("mase_model_avg"),
                "status": "success",
            }
        else:
            return {
                "run_id": run_id,
                "timestamp": timestamp,
                "config": config,
                "rmse": None,
                "mase": None,
                "status": "failed_no_metrics",
            }
    except Exception as e:
        print(f"❌ Run {run_id} failed with error: {e}")
        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "config": config,
            "rmse": None,
            "mase": None,
            "status": f"failed: {str(e)[:200]}",
        }


def main() -> None:
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    param_grid = get_param_grid()
    param_names = list(param_grid.keys())
    all_combinations = list(product(*param_grid.values()))

    print(f"\n{'='*80}")
    print("TFT GRID SEARCH")
    print(f"{'='*80}")
    print(f"Total combinations: {len(all_combinations)}")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*80}\n")

    results_file = base_output_dir / "grid_search_results.json"
    results_csv = base_output_dir / "grid_search_results.csv"

    completed_runs = []
    if results_file.exists():
        with open(results_file) as fh:
            completed_runs = json.load(fh)
        print(f"Resuming: {len(completed_runs)} configs already tested")

    all_results = completed_runs.copy()

    for run_id, combo in enumerate(all_combinations, start=len(completed_runs) + 1):
        config = {name: value for name, value in zip(param_names, combo)}
        if any(r.get("config") == config for r in completed_runs):
            print(f"Skipping run {run_id} (already done)")
            continue

        result = run_single_config(config, run_id, base_output_dir, args.max_epochs, args.early_stopping_patience)
        all_results.append(result)

        with open(results_file, "w") as fh:
            json.dump(all_results, fh, indent=2)

        rows = []
        for r in all_results:
            row = {"run_id": r.get("run_id"), "rmse": r.get("rmse"), "mase": r.get("mase"), "status": r.get("status")}
            if isinstance(r.get("config"), dict):
                row.update(r.get("config"))
            rows.append(row)
        pd.DataFrame(rows).to_csv(results_csv, index=False)

        print(f"Completed run {run_id}/{len(all_combinations)} - status: {result.get('status')}")

    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"Total runs: {len(all_results)}")

    successful = [r for r in all_results if r.get("rmse") is not None]
    if successful:
        best = min(successful, key=lambda x: x["rmse"])
        print(f"Best RMSE: {best['rmse']}, config: {best['config']}")
        with open(base_output_dir / "best_config.json", "w") as fh:
            json.dump(
                {"rmse": best["rmse"], "mase": best["mase"], "config": best["config"], "run_id": best["run_id"]},
                fh,
                indent=2,
            )
        print(f"Saved best config to: {base_output_dir / 'best_config.json'}")
    else:
        print("No successful runs found.")


if __name__ == "__main__":
    main()
Grid
