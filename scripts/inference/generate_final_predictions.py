"""ARCHIVED: generate_final_predictions

This script has been moved to:
    scripts/_archived/inference/generate_final_predictions.py

The archived copy contains the original implementation. This stub prevents
accidental execution from the active scripts/ tree.
"""

import sys

sys.exit(
        "This script was archived and moved to scripts/_archived/inference/generate_final_predictions.py. "
        "Use that file if you need the original implementation."
)


def compute_rmse(y_true, y_pred):
    """Calculate RMSE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mase(y_true, y_pred, seasonality=24):
    """Calculate MASE using naive seasonal forecast as baseline."""
    # For simplicity, compute MASE based on test set's own seasonal pattern
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # MAE of model
    mae_model = np.mean(np.abs(y_true_flat - y_pred_flat))

    # MAE of naive seasonal forecast (y[t] = y[t-24])
    if len(y_true_flat) > seasonality:
        naive_errors = np.abs(y_true_flat[seasonality:] - y_true_flat[:-seasonality])
        mae_naive = np.mean(naive_errors)
    else:
        mae_naive = np.mean(np.abs(y_true_flat))

    if mae_naive == 0:
        return np.nan

    return float(mae_model / mae_naive)


class PVForecastingDataset(torch.utils.data.Dataset):
    """Simple dataset for inference."""

    def __init__(
        self,
        df,
        pv_lag_features,
        weather_lag_features,
        forecast_features,
        target_col,
        seq_len,
        horizon,
        pv_scaler,
        weather_scaler,
        forecast_scaler,
        target_scaler,
    ):
        self.df = df.reset_index(drop=True)
        self.pv_lag_features = pv_lag_features
        self.weather_lag_features = weather_lag_features
        self.forecast_features = forecast_features
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        self.pv_scaler = pv_scaler
        self.weather_scaler = weather_scaler
        self.forecast_scaler = forecast_scaler
        self.target_scaler = target_scaler

        self.valid_indices = []
        for i in range(len(df) - seq_len - horizon + 1):
            if i + seq_len + horizon - 1 < len(df):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len
        target_start = end_idx
        target_end = target_start + self.horizon

        pv_history = self.df.loc[start_idx : end_idx - 1, self.pv_lag_features].values.astype(np.float32)
        weather_history = self.df.loc[start_idx : end_idx - 1, self.weather_lag_features].values.astype(np.float32)
        weather_forecast = self.df.loc[target_start : target_end - 1, self.forecast_features].values.astype(np.float32)
        targets = self.df.loc[target_start : target_end - 1, self.target_col].values.astype(np.float32)

        if self.pv_scaler is not None:
            pv_history = self.pv_scaler.transform(pv_history)
        if self.weather_scaler is not None:
            weather_history = self.weather_scaler.transform(weather_history)
        if self.forecast_scaler is not None:
            weather_forecast = self.forecast_scaler.transform(weather_forecast)
        if self.target_scaler is not None:
            targets = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()

        features = {
            "pv_history": torch.from_numpy(pv_history.astype(np.float32)),
            "weather_history": torch.from_numpy(weather_history.astype(np.float32)),
            "weather_forecast": torch.from_numpy(weather_forecast.astype(np.float32)),
        }
        targets = torch.from_numpy(targets.astype(np.float32))

        return features, targets


def generate_predictions():
    """Generate final test predictions with seed 2 model."""
    logger.info("=" * 60)
    logger.info("Generating FINAL predictions with Multi-Branch (seed 2)")
    logger.info("=" * 60)

    # Paths
    model_dir = Path("outputs/multi_branch/final_seed2")
    data_path = Path("outputs/processed.parquet")

    # Load model config
    with open(model_dir / "config_model.json") as f:
        config = json.load(f)

    # Load scalers
    with open(model_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Split (same as training: 60/20/20)
    n_samples = len(df)
    train_ratio = 0.6
    val_ratio = 0.2
    cutoff_train = int(n_samples * train_ratio)
    cutoff_val = int(n_samples * (train_ratio + val_ratio))

    test_df = df.iloc[cutoff_val:].copy()
    logger.info(f"Test set: {len(test_df)} samples")

    # Create dataset
    test_dataset = PVForecastingDataset(
        test_df,
        config["pv_lag_features"],
        config["weather_lag_features"],
        config["forecast_features"],
        "pv",
        config["seq_len"],
        config["horizon"],
        scalers["pv_scaler"],
        scalers["weather_scaler"],
        scalers["forecast_scaler"],
        scalers["target_scaler"],
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    logger.info(f"Test batches: {len(test_loader)}")

    # Load model
    checkpoint_path = model_dir / "multi-branch-best.ckpt"
    logger.info(f"Loading model from {checkpoint_path}")
    model = MultiBranchTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # Generate predictions
    all_preds = []
    all_targets = []

    logger.info("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            features, targets = batch
            features_device = {k: v.to(device) for k, v in features.items()}
            preds = model(features_device)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Denormalize
    logger.info("Denormalizing predictions")
    all_preds = scalers["target_scaler"].inverse_transform(all_preds)
    all_targets = scalers["target_scaler"].inverse_transform(all_targets)

    # Calculate metrics
    logger.info("\n" + "=" * 60)
    logger.info("FINAL METRICS (Test Set)")
    logger.info("=" * 60)

    rmse = compute_rmse(all_targets.flatten(), all_preds.flatten())
    mase = compute_mase(all_targets, all_preds, seasonality=24)

    logger.info(f"RMSE: {rmse:.4f} kW")
    logger.info(f"MASE: {mase:.4f}")

    # Per-horizon metrics
    logger.info("\nPer-Horizon Metrics:")
    logger.info("-" * 40)
    horizon_metrics = []
    for h in range(24):
        h_rmse = compute_rmse(all_targets[:, h], all_preds[:, h])
        h_mase = compute_mase(all_targets[:, h : h + 1], all_preds[:, h : h + 1], seasonality=24)
        horizon_metrics.append({"horizon": h + 1, "rmse": h_rmse, "mase": h_mase})
        logger.info(f"  h={h+1:2d}: RMSE={h_rmse:.3f}, MASE={h_mase:.3f}")

    # Save predictions - Wide format (one row per sample)
    logger.info("\nSaving predictions (wide format)...")
    pred_cols = {f"pred_h{h+1}": all_preds[:, h] for h in range(24)}
    target_cols = {f"actual_h{h+1}": all_targets[:, h] for h in range(24)}

    pred_df_wide = pd.DataFrame({**pred_cols, **target_cols})
    output_path_wide = model_dir / "predictions_test_wide.csv"
    pred_df_wide.to_csv(output_path_wide, index=False)
    logger.info(f"Saved: {output_path_wide}")

    # Save predictions - Long format (for plotting)
    logger.info("Saving predictions (long format)...")
    predictions_list = []
    for i in range(len(all_preds)):
        for h in range(24):
            predictions_list.append(
                {"sample_idx": i, "horizon": h + 1, "prediction": all_preds[i, h], "actual": all_targets[i, h]}
            )

    pred_df_long = pd.DataFrame(predictions_list)
    output_path_long = model_dir / "predictions_test_long.csv"
    pred_df_long.to_csv(output_path_long, index=False)
    logger.info(f"Saved: {output_path_long}")

    # Save metrics
    metrics_summary = {
        "model": "Multi-Branch Transformer",
        "seed": 2,
        "rmse": float(rmse),
        "mase": float(mase),
        "n_samples": len(all_preds),
        "horizon_metrics": horizon_metrics,
    }

    with open(model_dir / "metrics_final.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved: {model_dir / 'metrics_final.json'}")

    logger.info("\n" + "=" * 60)
    logger.info(f"✅ DONE! Generated {len(all_preds)} predictions × 24 horizons")
    logger.info(f"   RMSE: {rmse:.4f} kW | MASE: {mase:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_predictions()
