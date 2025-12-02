from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import dump

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

    data = data.dropna()
    feature_cols = [
        c
        for c in data.columns
        if not c.startswith("target_h") and c != "series_id" and np.issubdtype(data[c].dtype, np.number)
    ]
    targets = {h: data[f"target_h{h}"] for h in range(1, horizon + 1)}
    return data, feature_cols, targets


def chronological_masks(time_idx: pd.Series, train_ratio: float) -> Tuple[pd.Series, pd.Series]:
    cutoff = int(time_idx.max() * train_ratio)
    train_mask = time_idx <= cutoff
    val_mask = time_idx > cutoff
    return train_mask, val_mask


def train_lightgbm_multi_horizon(
    data: pd.DataFrame,
    feature_cols: List[str],
    targets: Dict[int, pd.Series],
    horizon: int,
    train_ratio: float,
    out_dir: Path,
    seed: int,
) -> Tuple[Dict[int, lgb.LGBMRegressor], pd.DataFrame, List[dict]]:
    """Train one LightGBM regressor per horizon; return models, preds and metrics."""
    train_mask, val_mask = chronological_masks(data["time_idx"], train_ratio=train_ratio)
    X_train = data.loc[train_mask, feature_cols]
    X_val = data.loc[val_mask, feature_cols]

    models: Dict[int, lgb.LGBMRegressor] = {}
    metrics: List[dict] = []
    pred_rows: List[dict] = []

    train_series = data.loc[train_mask, "pv"].values

    for h in range(1, horizon + 1):
        y_train = targets[h].loc[train_mask]
        y_val = targets[h].loc[val_mask]

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
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        models[h] = model

        y_pred_val = model.predict(X_val)
        naive_val = data["pv"].shift(24 - h).loc[val_mask]

        valid_mask = ~np.isnan(naive_val.values)
        y_true_valid = y_val.values[valid_mask]
        y_pred_valid = y_pred_val[valid_mask]
        naive_valid = naive_val.values[valid_mask]

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

        val_idx = data.index[val_mask]
        val_time_idx = data.loc[val_mask, "time_idx"].astype(int).values
        for i in range(len(y_val)):
            base_ts = val_idx[i]
            pred_rows.append(
                {
                    "time_idx": int(val_time_idx[i]),
                    "origin_timestamp_utc": base_ts.isoformat(),
                    "forecast_timestamp_utc": (base_ts + pd.Timedelta(hours=h)).isoformat(),
                    "horizon_h": h,
                    "y_true": float(y_val.iloc[i]),
                    "y_pred": float(y_pred_val[i]),
                    "y_naive": float(naive_val.iloc[i]) if not pd.isna(naive_val.iloc[i]) else np.nan,
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

    for fold in range(folds):
        train_end = start_train + fold * val_len
        val_end = train_end + val_len
        if val_end >= n:
            break
        train_mask = data.index[:train_end]
        val_mask = data.index[train_end:val_end]

        X_train = data.loc[train_mask, feature_cols]
        X_val = data.loc[val_mask, feature_cols]
        train_series = data.loc[train_mask, "pv"].values

        for h in range(1, horizon + 1):
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
    ap.add_argument("--pv-path", type=str, default="data/raw/pv_dataset.xlsx")
    ap.add_argument("--wx-path", type=str, default="data/raw/wx_dataset.xlsx")
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--lag-hours", type=str, default="1,24,168")
    ap.add_argument("--rolling-hours", type=str, default="3,6")
    ap.add_argument("--outdir", type=str, default="outputs_lgbm")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation share (chronological split)")
    ap.add_argument("--use-future-meteo", action="store_true", help="Add future meteo covariates if available")
    ap.add_argument("--walk-forward-folds", type=int, default=0, help="Run optional walk-forward evaluation")
    ap.add_argument("--fold-val-frac", type=float, default=0.1, help="Validation fraction per fold for walk-forward")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.outdir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    lag_hours = parse_int_list(args.lag_hours, default=(1, 24, 168))
    rolling_hours = parse_int_list(args.rolling_hours, default=(3, 6))

    df = load_and_engineer_features(
        Path(args.pv_path),
        Path(args.wx_path),
        args.local_tz,
        lag_hours=lag_hours,
        rolling_hours=rolling_hours,
    )

    # Persist unified processed data for downstream scripts
    persist_processed(df, Path("outputs"))

    future_cols = ["ghi", "dni", "dhi", "temp", "humidity", "clouds", "wind_speed"]
    data, feature_cols, targets = add_supervised_targets(
        df, horizon=args.horizon, use_future_meteo=args.use_future_meteo, future_cols=future_cols
    )

    # Train + validate on final split
    models, preds_df, metrics = train_lightgbm_multi_horizon(
        data=data,
        feature_cols=feature_cols,
        targets=targets,
        horizon=args.horizon,
        train_ratio=1 - args.val_ratio,
        out_dir=out_dir,
        seed=args.seed,
    )

    preds_df.to_csv(out_dir / "predictions_val_lgbm.csv", index=False)
    (out_dir / "metrics_val_lgbm.json").write_text(json.dumps(metrics, indent=2))

    metric_summary = {
        "rmse_model_avg": float(np.mean([m["rmse_model"] for m in metrics])),
        "rmse_naive_avg": float(np.mean([m["rmse_naive"] for m in metrics])),
        "mase_model_avg": float(np.mean([m["mase_model"] for m in metrics])),
        "mase_naive_avg": float(np.mean([m["mase_naive"] for m in metrics])),
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metric_summary, indent=2))

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
        "val_ratio": args.val_ratio,
        "use_future_meteo": args.use_future_meteo,
        "walk_forward_folds": args.walk_forward_folds,
        "fold_val_frac": args.fold_val_frac,
        "seed": args.seed,
        "feature_cols": feature_cols,
        "future_cols": future_cols,
    }
    (out_dir / "config_lgbm.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
