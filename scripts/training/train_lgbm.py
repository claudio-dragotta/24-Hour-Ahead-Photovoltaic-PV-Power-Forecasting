from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import dump
from tqdm import tqdm

from pv_forecasting.metrics import mase, rmse
from pv_forecasting.pipeline import load_and_engineer_features, persist_processed


def parse_int_list(text: str, default: Sequence[int]) -> List[int]:
    if not text:
        return list(default)
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except Exception:
        return list(default)


def add_supervised_targets(
    df: pd.DataFrame,
    horizon: int,
    use_future_meteo: bool = False,
    future_cols: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, List[str], Dict[int, pd.Series]]:
    """Create target columns per horizon and optional future meteo covariates."""
    future_cols = list(future_cols or [])
    data = df.copy()
    for h in range(1, horizon + 1):
        data[f"target_h{h}"] = data["pv"].shift(-h)
        if use_future_meteo:
            for c in future_cols:
                if c in data.columns:
                    data[f"{c}_fut_h{h}"] = data[c].shift(-h)

    # Select numeric feature columns (includes encoded weather_description)
    target_cols = [f"target_h{h}" for h in range(1, horizon + 1)]
    feature_cols = [
        c
        for c in data.columns
        if (
            not c.startswith("target_h")
            and c != "series_id"
            and np.issubdtype(data[c].dtype, np.number)
            and not data[c].isna().all()  # exclude columns that are entirely NaN
        )
    ]
    drop_subset = feature_cols + target_cols
    data = data.dropna(subset=drop_subset)
    targets = {h: data[f"target_h{h}"] for h in range(1, horizon + 1)}
    return data, feature_cols, targets


def chronological_masks(
    time_idx: pd.Series, train_ratio: float, val_ratio: float
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Split time_idx chronologically into train/val/test masks."""
    max_idx = time_idx.max()
    cutoff_train = int(max_idx * train_ratio)
    cutoff_val = int(max_idx * (train_ratio + val_ratio))
    train_mask = time_idx <= cutoff_train
    val_mask = (time_idx > cutoff_train) & (time_idx <= cutoff_val)
    test_mask = time_idx > cutoff_val
    return train_mask, val_mask, test_mask


def compute_solar_weights(zenith_deg: pd.Series, min_weight: float = 0.1) -> np.ndarray:
    """Compute sample weights based on solar zenith angle.

    Higher weights during daytime (low zenith) and lower weights at night (high zenith).
    Uses cosine of zenith angle as it represents solar irradiance intensity.

    Args:
        zenith_deg: Solar zenith angle in degrees (0° = sun overhead, 90° = horizon)
        min_weight: Minimum weight to assign (prevents zero weights at night)

    Returns:
        Array of sample weights normalized to mean=1.0
    """
    zenith_rad = np.deg2rad(zenith_deg.values)
    # cos(zenith) ranges from 1 (sun overhead) to 0 (horizon) to negative (night)
    cos_zenith = np.cos(zenith_rad)
    # Clip to [0, 1] and add minimum weight
    weights = np.maximum(cos_zenith, 0.0) + min_weight
    # Normalize to mean=1.0 to preserve overall scale
    weights = weights / weights.mean()
    return weights


def train_lightgbm_multi_horizon(
    data: pd.DataFrame,
    feature_cols: List[str],
    targets: Dict[int, pd.Series],
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    out_dir: Path,
    seed: int,
) -> Tuple[Dict[int, lgb.LGBMRegressor], pd.DataFrame, List[dict]]:
    """Train one LightGBM regressor per horizon with train/val/test split.

    - train: for model training
    - val: for early stopping
    - test: for final evaluation (never seen during training)
    """
    train_mask, val_mask, test_mask = chronological_masks(
        data["time_idx"], train_ratio=train_ratio, val_ratio=val_ratio
    )
    X_train = data.loc[train_mask, feature_cols]
    X_val = data.loc[val_mask, feature_cols]
    X_test = data.loc[test_mask, feature_cols]

    models: Dict[int, lgb.LGBMRegressor] = {}
    metrics: List[dict] = []
    pred_rows: List[dict] = []

    train_series = data.loc[train_mask, "pv"].values

    # Compute solar-based sample weights for training set
    if "sp_zenith" in data.columns:
        train_weights = compute_solar_weights(data.loc[train_mask, "sp_zenith"])
        print(f"\n{'='*60}")
        print(f"Training LightGBM with solar-weighted samples")
        print(f"Weight stats: min={train_weights.min():.3f}, max={train_weights.max():.3f}, mean={train_weights.mean():.3f}")
        print(f"{'='*60}")
    else:
        train_weights = None
        print(f"\n{'='*60}")
        print(f"Training LightGBM multi-horizon models (no solar weights)")
        print(f"{'='*60}")

    for h in tqdm(range(1, horizon + 1), desc="Training horizons", unit="model"):
        y_train = targets[h].loc[train_mask]
        y_val = targets[h].loc[val_mask]
        y_test = targets[h].loc[test_mask]

        model = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.02,
            num_leaves=96,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            objective="regression",
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,  # Apply solar-based sample weights
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        models[h] = model

        # Evaluate on TEST set (never seen during training!)
        y_pred_test = model.predict(X_test)
        naive_test = data["pv"].shift(24 - h).loc[test_mask]

        valid_mask = ~np.isnan(naive_test.values)
        y_true_valid = y_test.values[valid_mask]
        y_pred_valid = y_pred_test[valid_mask]
        naive_valid = naive_test.values[valid_mask]

        rmse_model = rmse(y_true_valid, y_pred_valid)
        rmse_naive = rmse(y_true_valid, naive_valid)
        mase_model = mase(y_true_valid, y_pred_valid, train_series=train_series, m=24)
        mase_naive = mase(y_true_valid, naive_valid, train_series=train_series, m=24)
        metrics.append(
            {
                "horizon_h": h,
                "rmse_model": rmse_model,
                "rmse_naive": rmse_naive,
                "mase_model": mase_model,
                "mase_naive": mase_naive,
            }
        )

        test_idx = data.index[test_mask]
        test_time_idx = data.loc[test_mask, "time_idx"].astype(int).values
        for i in range(len(y_test)):
            base_ts = test_idx[i]
            pred_rows.append(
                {
                    "time_idx": int(test_time_idx[i]),
                    "origin_timestamp_utc": base_ts.isoformat(),
                    "forecast_timestamp_utc": (base_ts + pd.Timedelta(hours=h)).isoformat(),
                    "horizon_h": h,
                    "y_true": float(y_test.iloc[i]),
                    "y_pred": float(y_pred_test[i]),
                    "y_naive": float(naive_test.iloc[i]) if not pd.isna(naive_test.iloc[i]) else np.nan,
                }
            )

        dump(model, out_dir / "models" / f"lgbm_h{h}.joblib")

    preds_df = pd.DataFrame(pred_rows)
    return models, preds_df, metrics


def run_walk_forward(
    data: pd.DataFrame,
    feature_cols: List[str],
    targets: Dict[int, pd.Series],
    horizon: int,
    folds: int,
    fold_val_frac: float,
    seed: int,
) -> List[dict]:
    """Simple expanding-window walk-forward evaluation (no model persistence)."""
    n = len(data)
    fold_metrics: List[dict] = []
    val_len = max(1, int(n * fold_val_frac))
    start_train = max(24, int(n * 0.5))

    print(f"\n{'='*60}")
    print(f"Running Walk-Forward Validation ({folds} folds)")
    print(f"{'='*60}")

    for fold in tqdm(range(folds), desc="Walk-forward folds", unit="fold"):
        train_end = start_train + fold * val_len
        val_end = train_end + val_len
        if val_end >= n:
            break
        train_mask = data.index[:train_end]
        val_mask = data.index[train_end:val_end]

        X_train = data.loc[train_mask, feature_cols]
        X_val = data.loc[val_mask, feature_cols]
        train_series = data.loc[train_mask, "pv"].values

        # Compute solar-based sample weights for this fold's training set
        if "sp_zenith" in data.columns:
            fold_weights = compute_solar_weights(data.loc[train_mask, "sp_zenith"])
        else:
            fold_weights = None

        for h in tqdm(range(1, horizon + 1), desc=f"  Fold {fold+1}/{folds}", unit="h", leave=False):
            y_train = targets[h].loc[train_mask]
            y_val = targets[h].loc[val_mask]
            model = lgb.LGBMRegressor(
                n_estimators=800,
                learning_rate=0.03,
                num_leaves=64,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed + fold,
                objective="regression",
                n_jobs=-1,
            )
            model.fit(
                X_train,
                y_train,
                sample_weight=fold_weights,  # Apply solar-based sample weights
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            y_pred_val = model.predict(X_val)
            naive_val = data["pv"].shift(24 - h).loc[val_mask]
            valid_mask = ~np.isnan(naive_val.values)
            y_true_valid = y_val.values[valid_mask]
            y_pred_valid = y_pred_val[valid_mask]
            naive_valid = naive_val.values[valid_mask]
            fold_metrics.append(
                {
                    "fold": fold + 1,
                    "horizon_h": h,
                    "rmse_model": rmse(y_true_valid, y_pred_valid),
                    "rmse_naive": rmse(y_true_valid, naive_valid),
                    "mase_model": mase(y_true_valid, y_pred_valid, train_series=train_series, m=24),
                    "mase_naive": mase(y_true_valid, naive_valid, train_series=train_series, m=24),
                }
            )
    return fold_metrics


def parse_args():
    ap = argparse.ArgumentParser(description="Train LightGBM multi-horizon baseline for 24h PV forecasting")
    ap.add_argument(
        "--processed-path",
        type=str,
        default="outputs/processed.parquet",
        help="path to pre-processed parquet file (preferred)",
    )
    ap.add_argument("--pv-path", type=str, default="data/raw/pv_dataset.xlsx")
    ap.add_argument("--wx-path", type=str, default="data/raw/wx_dataset.xlsx")
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--lag-hours", type=str, default="1,24,168")
    ap.add_argument("--rolling-hours", type=str, default="3,6")
    ap.add_argument("--outdir", type=str, default="outputs_lgbm")
    ap.add_argument("--train-ratio", type=float, default=0.6, help="Training set ratio (chronological split)")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio (chronological split)")
    ap.add_argument("--test-ratio", type=float, default=0.2, help="Test set ratio (chronological split)")
    ap.add_argument("--use-future-meteo", action="store_true", help="Add future meteo covariates if available")
    ap.add_argument("--walk-forward-folds", type=int, default=0, help="Run optional walk-forward evaluation")
    ap.add_argument("--fold-val-frac", type=float, default=0.1, help="Validation fraction per fold for walk-forward")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"LightGBM Multi-Horizon Training Pipeline")
    print(f"{'='*60}")
    print(f"Output directory: {args.outdir}")
    print(f"Horizon: {args.horizon} hours")
    print(f"Data split: Train {args.train_ratio:.1%} | Val {args.val_ratio:.1%} | Test {args.test_ratio:.1%}")
    print(f"Walk-forward folds: {args.walk_forward_folds if args.walk_forward_folds > 0 else 'disabled'}")
    print(f"Future meteo: {'enabled' if args.use_future_meteo else 'disabled'}")
    print(f"{'='*60}\n")

    out_dir = Path(args.outdir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    lag_hours = parse_int_list(args.lag_hours, default=(1, 24, 168))
    rolling_hours = parse_int_list(args.rolling_hours, default=(3, 6))

    print("Loading and engineering features...")

    # Check if processed data already exists
    processed_path = Path(args.processed_path)
    if processed_path.exists():
        print(f"Loading pre-processed data from {processed_path}...")
        df = pd.read_parquet(processed_path)
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features from cache\n")
    else:
        print("Processing raw data...")
        df = load_and_engineer_features(
            Path(args.pv_path),
            Path(args.wx_path),
            args.local_tz,
            lag_hours=lag_hours,
            rolling_hours=rolling_hours,
        )
        # Persist unified processed data for downstream scripts
        persist_processed(df, Path("outputs"))
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features\n")

    print("Creating supervised targets...")
    future_cols = ["ghi", "dni", "dhi", "temp", "humidity", "clouds", "wind_speed"]
    data, feature_cols, targets = add_supervised_targets(
        df, horizon=args.horizon, use_future_meteo=args.use_future_meteo, future_cols=future_cols
    )
    print(f"✓ Created {len(data)} training samples with {len(feature_cols)} features\n")

    # Train + validate + test with 3-way chronological split
    models, preds_df, metrics = train_lightgbm_multi_horizon(
        data=data,
        feature_cols=feature_cols,
        targets=targets,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        out_dir=out_dir,
        seed=args.seed,
    )

    preds_df.to_csv(out_dir / "predictions_test_lgbm.csv", index=False)
    (out_dir / "metrics_test_lgbm.json").write_text(json.dumps(metrics, indent=2))

    metric_summary = {
        "rmse_model_avg": float(np.mean([m["rmse_model"] for m in metrics])),
        "rmse_naive_avg": float(np.mean([m["rmse_naive"] for m in metrics])),
        "mase_model_avg": float(np.mean([m["mase_model"] for m in metrics])),
        "mase_naive_avg": float(np.mean([m["mase_naive"] for m in metrics])),
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metric_summary, indent=2))

    print(f"\n{'='*60}")
    print(f"Training Complete - Validation Results")
    print(f"{'='*60}")
    print(f"RMSE (model): {metric_summary['rmse_model_avg']:.4f}")
    print(f"RMSE (naive): {metric_summary['rmse_naive_avg']:.4f}")
    print(f"MASE (model): {metric_summary['mase_model_avg']:.4f}")
    print(f"MASE (naive): {metric_summary['mase_naive_avg']:.4f}")
    print(f"{'='*60}\n")

    # Feature importance analysis (averaged across all horizons)
    print(f"{'='*60}")
    print(f"FEATURE IMPORTANCE (Top 15)")
    print(f"{'='*60}")

    # Aggregate feature importances from all horizon models
    feat_importance_dict = {feat: [] for feat in feature_cols}
    for h, model in models.items():
        for feat, imp in zip(feature_cols, model.feature_importances_):
            feat_importance_dict[feat].append(imp)

    # Compute mean importance across all horizons
    feat_importance_avg = {feat: np.mean(imps) for feat, imps in feat_importance_dict.items()}
    feat_importance_sorted = sorted(feat_importance_avg.items(), key=lambda x: x[1], reverse=True)

    # Display top 15 features
    for i, (feat, imp) in enumerate(feat_importance_sorted[:15], 1):
        print(f"{i:2d}. {feat:25s} importance: {imp:10.1f}")

    # Save full feature importances to JSON
    (out_dir / "feature_importance_lgbm.json").write_text(
        json.dumps([{"feature": f, "importance": float(imp)} for f, imp in feat_importance_sorted], indent=2)
    )
    print(f"\n✓ Full feature importance saved to: feature_importance_lgbm.json\n")

    # Walk-forward evaluation (optional)
    wf_metrics: List[dict] = []
    if args.walk_forward_folds and args.walk_forward_folds > 0:
        wf_metrics = run_walk_forward(
            data=data,
            feature_cols=feature_cols,
            targets=targets,
            horizon=args.horizon,
            folds=args.walk_forward_folds,
            fold_val_frac=args.fold_val_frac,
            seed=args.seed,
        )
        (out_dir / "metrics_walk_forward.json").write_text(json.dumps(wf_metrics, indent=2))

    config = {
        "horizon": args.horizon,
        "lag_hours": lag_hours,
        "rolling_hours": rolling_hours,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "use_future_meteo": args.use_future_meteo,
        "walk_forward_folds": args.walk_forward_folds,
        "fold_val_frac": args.fold_val_frac,
        "seed": args.seed,
        "feature_cols": feature_cols,
        "future_cols": future_cols,
    }
    (out_dir / "config_lgbm.json").write_text(json.dumps(config, indent=2))

    print(f"\n{'='*60}")
    print(f"All Done! Results saved to: {out_dir}")
    print(f"{'='*60}")
    print(f"✓ Models: {out_dir / 'models'} (24 files)")
    print(f"✓ Predictions: {out_dir / 'predictions_test_lgbm.csv'}")
    print(f"✓ Metrics: {out_dir / 'metrics_test_lgbm.json'}")
    print(f"✓ Feature importance: {out_dir / 'feature_importance_lgbm.json'}")
    if args.walk_forward_folds > 0:
        print(f"✓ Walk-forward metrics: {out_dir / 'metrics_walk_forward.json'}")
    print(f"✓ Config: {out_dir / 'config_lgbm.json'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
