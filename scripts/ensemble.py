from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Blend TFT + LightGBM predictions with validation-derived weights")
    ap.add_argument("--tft-preds", type=str, default="outputs_tft/predictions_val_tft.csv")
    ap.add_argument("--lgbm-preds", type=str, default="outputs_lgbm/predictions_val_lgbm.csv")
    ap.add_argument("--processed-path", type=str, default="outputs/processed.parquet", help="Processed dataset for MASE denominator")
    ap.add_argument("--outdir", type=str, default="outputs_ensemble")
    ap.add_argument("--weight", type=float, default=None, help="Manually set TFT weight (0-1). If None, grid search.")
    ap.add_argument("--grid-step", type=float, default=0.05)
    return ap.parse_args()


def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"horizon_h", "y_pred", "time_idx"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {path}: {required - set(df.columns)}")
    return df


def mase_denominator(processed_path: Path) -> float | None:
    if not processed_path.exists():
        return None
    df = pd.read_parquet(processed_path)
    train_series = df["pv"].values[: int(len(df) * 0.7)]
    m = 24
    if len(train_series) <= m:
        return None
    return float(np.mean(np.abs(train_series[m:] - train_series[:-m])))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mase_denom: float | None) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mase = None
    if mase_denom and mase_denom > 0:
        mase = float(np.mean(np.abs(y_true - y_pred)) / mase_denom)
    return {"rmse": rmse, "mase": mase}


def find_best_weight(y_true: np.ndarray, tft: np.ndarray, lgbm: np.ndarray, grid_step: float) -> Tuple[float, dict]:
    best_w = 0.5
    best_rmse = np.inf
    best_metrics = {}
    for w in np.arange(0.0, 1.0 + 1e-6, grid_step):
        blended = w * tft + (1 - w) * lgbm
        rmse = float(np.sqrt(np.mean((y_true - blended) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = float(w)
            best_metrics = {"rmse": rmse}
    return best_w, best_metrics


def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tft = load_preds(Path(args.tft_preds))
    lgbm = load_preds(Path(args.lgbm_preds))

    merged = pd.merge(
        tft,
        lgbm,
        on=["time_idx", "horizon_h"],
        suffixes=("_tft", "_lgbm"),
        how="inner",
    )
    if "y_true_tft" in merged.columns:
        merged["y_true"] = merged["y_true_tft"]
    elif "y_true_lgbm" in merged.columns:
        merged["y_true"] = merged["y_true_lgbm"]

    y_true = merged["y_true"].values
    y_tft = merged["y_pred_tft"].values
    y_lgbm = merged["y_pred_lgbm"].values

    mase_denom = mase_denominator(Path(args.processed_path))

    weight = args.weight
    weight_metrics = {}
    if weight is None:
        weight, weight_metrics = find_best_weight(y_true, y_tft, y_lgbm, args.grid_step)

    blended = weight * y_tft + (1 - weight) * y_lgbm

    merged["y_pred_ensemble"] = blended
    merged["weight_tft"] = weight
    merged.to_csv(out_dir / "predictions_val_ensemble.csv", index=False)

    metrics_overall = compute_metrics(y_true, blended, mase_denom)
    metrics_h = []
    for h in sorted(merged["horizon_h"].unique()):
        sub = merged[merged["horizon_h"] == h]
        m = compute_metrics(sub["y_true"].values, sub["y_pred_ensemble"].values, mase_denom)
        m["horizon_h"] = int(h)
        metrics_h.append(m)

    result = {
        "weight_tft": weight,
        "metrics_overall": metrics_overall,
        "metrics_by_horizon": metrics_h,
        "weight_search": weight_metrics,
    }
    (out_dir / "ensemble_weights.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
