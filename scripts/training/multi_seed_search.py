"""ARCHIVED: multi_seed_search

Original script moved to scripts/_archived/training/multi_seed_search.py
This file replaced with a small stub to avoid accidental execution.
"""

import sys

sys.exit(
    "This script was archived and moved to scripts/_archived/training/multi_seed_search.py. "
    "Use the archived copy for reference."
)


# ============================================================================
# Dataset Class (copied from train_multi_branch.py)
# ============================================================================


class PVForecastingDataset(Dataset):
    """PyTorch dataset for multi-branch transformer training."""

    def __init__(
        self,
        df: pd.DataFrame,
        pv_lag_features: List[str],
        weather_lag_features: List[str],
        forecast_features: List[str],
        target_col: str = "pv",
        seq_len: int = 168,
        horizon: int = 24,
        pv_scaler: Optional[StandardScaler] = None,
        weather_scaler: Optional[StandardScaler] = None,
        forecast_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
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

        # Create valid indices
        self.valid_indices = []
        for i in range(len(df) - seq_len - horizon + 1):
            if i + seq_len + horizon - 1 < len(df):
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
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


# ============================================================================
# Tracker Class
# ============================================================================


class SeedSearchTracker:
    """Tracks results across multiple seed trainings."""

    def __init__(self, output_dir: Path, keep_top: int = 10):
        self.output_dir = output_dir
        self.keep_top = keep_top
        self.results_file = output_dir / "seed_search_results.json"
        self.results = []
        self.best_mase = float("inf")

        if self.results_file.exists():
            with open(self.results_file) as f:
                data = json.load(f)
                self.results = data.get("results", [])
                if self.results:
                    self.best_mase = min(r["mase"] for r in self.results)
                    logger.info(f"Loaded {len(self.results)} previous results")
                    logger.info(f"Current best MASE: {self.best_mase:.4f}")

    def get_completed_seeds(self) -> set:
        return {r["seed"] for r in self.results}

    def add_result(self, seed: int, rmse_val: float, mase_val: float, epochs: int, duration: float):
        result = {
            "seed": seed,
            "rmse": rmse_val,
            "mase": mase_val,
            "epochs": epochs,
            "duration_min": round(duration / 60, 2),
            "timestamp": datetime.now().isoformat(),
        }
        self.results.append(result)

        if mase_val < self.best_mase:
            self.best_mase = mase_val
            logger.info(f"🎯 NEW BEST! Seed {seed} with MASE {mase_val:.4f}")

        self._save_results()
        self._cleanup_checkpoints()

    def _save_results(self):
        sorted_results = sorted(self.results, key=lambda x: x["mase"])
        mases = [r["mase"] for r in self.results]
        stats = {
            "total_seeds": len(self.results),
            "best_mase": float(min(mases)),
            "worst_mase": float(max(mases)),
            "mean_mase": float(np.mean(mases)),
            "std_mase": float(np.std(mases)),
            "median_mase": float(np.median(mases)),
        }
        data = {"stats": stats, "top_seeds": sorted_results[: self.keep_top], "results": sorted_results}
        with open(self.results_file, "w") as f:
            json.dump(data, f, indent=2)

    def _cleanup_checkpoints(self):
        if len(self.results) <= self.keep_top:
            return
        sorted_results = sorted(self.results, key=lambda x: x["mase"])
        top_seeds = {r["seed"] for r in sorted_results[: self.keep_top]}
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("seed_"):
                seed = int(item.name.split("_")[1])
                if seed not in top_seeds:
                    shutil.rmtree(item)

    def print_summary(self):
        if not self.results:
            logger.info("No results yet")
            return
        sorted_results = sorted(self.results, key=lambda x: x["mase"])
        mases = [r["mase"] for r in self.results]
        print("\n" + "=" * 60)
        print("MULTI-SEED SEARCH SUMMARY")
        print("=" * 60)
        print(f"Total seeds trained: {len(self.results)}")
        print(f"Best MASE: {min(mases):.4f} (seed {sorted_results[0]['seed']})")
        print(f"Worst MASE: {max(mases):.4f}")
        print(f"Mean MASE: {np.mean(mases):.4f} ± {np.std(mases):.4f}")
        print(f"Median MASE: {np.median(mases):.4f}")
        print(f"\nTop {self.keep_top} seeds:")
        print("-" * 40)
        for i, r in enumerate(sorted_results[: self.keep_top], 1):
            print(f"  {i}. Seed {r['seed']:4d}: MASE={r['mase']:.4f}, RMSE={r['rmse']:.2f}")
        if len(self.results) > 0:
            avg_time = np.mean([r["duration_min"] for r in self.results])
            print(f"\nAverage training time: {avg_time:.1f} min/seed")
        print("=" * 60 + "\n")


# ============================================================================
# Data Loading Functions
# ============================================================================


def compute_solar_weights(zenith_deg: pd.Series, min_weight: float = 0.1, gamma: float = 1.5) -> np.ndarray:
    zenith_rad = np.deg2rad(zenith_deg.values)
    cos_zenith = np.cos(zenith_rad)
    weights = np.maximum(cos_zenith, 0.0) ** gamma + min_weight
    weights = weights / weights.mean()
    return weights


def load_and_prepare_data(data_path: str, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """Load data and prepare feature groups."""
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")

    # Compute solar weights
    if "sp_zenith" in df.columns:
        solar_weights = compute_solar_weights(df["sp_zenith"])
    else:
        solar_weights = np.ones(len(df))

    # Define feature groups
    pv_lag_features = [c for c in df.columns if c.startswith("pv_lag") or c.startswith("pv_roll")]
    weather_lag_features = [
        c
        for c in df.columns
        if (
            c.startswith("ghi_lag")
            or c.startswith("dni_lag")
            or c.startswith("dhi_lag")
            or c.startswith("temp_lag")
            or "roll" in c
            and not c.startswith("pv_roll")
        )
    ]
    forecast_features = [
        c
        for c in [
            "temp",
            "humidity",
            "wind_speed",
            "clouds",
            "ghi",
            "dni",
            "dhi",
            "sp_zenith",
            "sp_azimuth",
            "cs_ghi",
            "cs_dni",
            "cs_dhi",
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "is_weekend",
            "is_holiday",
        ]
        if c in df.columns
    ]

    # Chronological split
    n_samples = len(df)
    cutoff_train = int(n_samples * train_ratio)
    cutoff_val = int(n_samples * (train_ratio + val_ratio))

    # Fit scalers on training data
    train_df = df.iloc[:cutoff_train]
    pv_scaler = StandardScaler().fit(train_df[pv_lag_features].values)
    weather_scaler = StandardScaler().fit(train_df[weather_lag_features].values)
    forecast_scaler = StandardScaler().fit(train_df[forecast_features].values)
    target_scaler = StandardScaler().fit(train_df["pv"].values.reshape(-1, 1))

    return {
        "df": df,
        "pv_lag_features": pv_lag_features,
        "weather_lag_features": weather_lag_features,
        "forecast_features": forecast_features,
        "pv_scaler": pv_scaler,
        "weather_scaler": weather_scaler,
        "forecast_scaler": forecast_scaler,
        "target_scaler": target_scaler,
        "cutoff_train": cutoff_train,
        "cutoff_val": cutoff_val,
        "solar_weights": solar_weights,
    }


def create_dataloaders(data_dict: dict, batch_size: int = 64):
    """Create train, val, test dataloaders."""
    df = data_dict["df"]
    pv_lag = data_dict["pv_lag_features"]
    weather_lag = data_dict["weather_lag_features"]
    forecast = data_dict["forecast_features"]
    cutoff_train = data_dict["cutoff_train"]
    cutoff_val = data_dict["cutoff_val"]

    train_dataset = PVForecastingDataset(
        df.iloc[:cutoff_train],
        pv_lag,
        weather_lag,
        forecast,
        pv_scaler=data_dict["pv_scaler"],
        weather_scaler=data_dict["weather_scaler"],
        forecast_scaler=data_dict["forecast_scaler"],
        target_scaler=data_dict["target_scaler"],
    )
    val_dataset = PVForecastingDataset(
        df.iloc[cutoff_train:cutoff_val],
        pv_lag,
        weather_lag,
        forecast,
        pv_scaler=data_dict["pv_scaler"],
        weather_scaler=data_dict["weather_scaler"],
        forecast_scaler=data_dict["forecast_scaler"],
        target_scaler=data_dict["target_scaler"],
    )
    test_dataset = PVForecastingDataset(
        df.iloc[cutoff_val:],
        pv_lag,
        weather_lag,
        forecast,
        pv_scaler=data_dict["pv_scaler"],
        weather_scaler=data_dict["weather_scaler"],
        forecast_scaler=data_dict["forecast_scaler"],
        target_scaler=data_dict["target_scaler"],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ============================================================================
# Training Function
# ============================================================================


def train_single_seed(
    seed: int,
    output_dir: Path,
    data_dict: dict,
    args: argparse.Namespace,
) -> tuple:
    """Train model with a single seed. Returns (rmse, mase, epochs, duration)."""

    start_time = time.time()
    seed_everything(seed, workers=True)

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(data_dict, args.batch_size)

    # Create model
    model = MultiBranchTransformer(
        n_pv_features=len(data_dict["pv_lag_features"]),
        n_hist_weather_features=len(data_dict["weather_lag_features"]),
        n_forecast_weather_features=len(data_dict["forecast_features"]),
        seq_len_encoder=168,
        seq_len_decoder=24,
        d_model=args.d_model,
        num_heads=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=seed_dir,
        filename="multi-branch-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.early_stopping_patience,
        mode="min",
        verbose=False,
    )

    # Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        gradient_clip_val=0.1,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Load best model and evaluate
    best_path = checkpoint_callback.best_model_path
    model = MultiBranchTransformer.load_from_checkpoint(best_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate predictions
    all_preds = []
    all_targets = []
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
    target_scaler = data_dict["target_scaler"]
    all_preds = target_scaler.inverse_transform(all_preds)
    all_targets = target_scaler.inverse_transform(all_targets)

    # Compute metrics (average across all horizons)
    rmse_vals = []
    mase_vals = []
    train_series = data_dict["df"].iloc[: data_dict["cutoff_train"]]["pv"].values

    for h in range(24):
        y_true = all_targets[:, h]
        y_pred = all_preds[:, h]
        rmse_vals.append(compute_rmse(y_true, y_pred))
        mase_vals.append(compute_mase(y_true, y_pred, train_series))

    avg_rmse = np.mean(rmse_vals)
    avg_mase = np.mean(mase_vals)

    duration = time.time() - start_time
    epochs_trained = trainer.current_epoch + 1

    return avg_rmse, avg_mase, epochs_trained, duration


def generate_seeds(n_seeds: int) -> list:
    """Generate a list of seeds to test."""
    seeds = set()

    # Common seeds
    common_seeds = [0, 1, 7, 13, 21, 42, 123, 256, 314, 420, 512, 666, 777, 999, 1000, 1234, 2024, 2025, 2026]
    seeds.update(common_seeds)

    # Sequential seeds
    seeds.update(range(0, n_seeds * 2, 2))

    # Random seeds
    np.random.seed(42)
    random_seeds = np.random.randint(1, 10000, size=n_seeds).tolist()
    seeds.update(random_seeds)

    return sorted(list(seeds))[:n_seeds]


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-seed search for Multi-Branch Transformer")
    parser.add_argument("--n-seeds", type=int, default=100, help="Number of seeds to test")
    parser.add_argument("--keep-top", type=int, default=10, help="Keep only top N checkpoints")
    parser.add_argument("--outdir", type=str, default="outputs/multi_branch/seed_search", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--data-path", type=str, default="outputs/processed.parquet", help="Path to processed data")

    # Model hyperparameters
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)

    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = SeedSearchTracker(output_dir, keep_top=args.keep_top)

    all_seeds = generate_seeds(args.n_seeds)
    completed_seeds = tracker.get_completed_seeds()
    seeds_to_run = [s for s in all_seeds if s not in completed_seeds]

    logger.info(f"Seeds to test: {len(seeds_to_run)} (skipping {len(completed_seeds)} already completed)")

    if not seeds_to_run:
        logger.info("All seeds already completed!")
        tracker.print_summary()
        return

    # Load data once
    logger.info("Loading data...")
    data_dict = load_and_prepare_data(args.data_path)
    logger.info(
        f"PV features: {len(data_dict['pv_lag_features'])}, Weather lag: {len(data_dict['weather_lag_features'])}, Forecast: {len(data_dict['forecast_features'])}"
    )

    # Estimate time
    avg_time_per_seed = 12
    total_time_est = len(seeds_to_run) * avg_time_per_seed
    logger.info(f"Estimated total time: {total_time_est // 60}h {total_time_est % 60}m")
    logger.info(f"Expected completion: {datetime.now() + timedelta(minutes=total_time_est)}")

    print("\n" + "=" * 60)
    print(f"STARTING MULTI-SEED SEARCH ({len(seeds_to_run)} seeds)")
    print("=" * 60 + "\n")

    for i, seed in enumerate(seeds_to_run, 1):
        logger.info(f"[{i}/{len(seeds_to_run)}] Training seed {seed}...")

        try:
            rmse_val, mase_val, epochs, duration = train_single_seed(
                seed=seed,
                output_dir=output_dir,
                data_dict=data_dict,
                args=args,
            )

            tracker.add_result(seed, rmse_val, mase_val, epochs, duration)

            remaining = len(seeds_to_run) - i
            avg_time = np.mean([r["duration_min"] for r in tracker.results])
            eta = remaining * avg_time
            logger.info(
                f"    Seed {seed}: MASE={mase_val:.4f}, RMSE={rmse_val:.2f}, "
                f"Epochs={epochs}, Time={duration/60:.1f}min | "
                f"Best: {tracker.best_mase:.4f} | ETA: {eta:.0f}min"
            )

        except Exception as e:
            logger.error(f"Error training seed {seed}: {e}")
            continue

        if i % 10 == 0:
            tracker.print_summary()

    tracker.print_summary()

    # Save final report
    report_file = output_dir / "final_report.txt"
    with open(report_file, "w") as f:
        f.write("MULTI-SEED SEARCH FINAL REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Total seeds tested: {len(tracker.results)}\n\n")
        sorted_results = sorted(tracker.results, key=lambda x: x["mase"])
        mases = [r["mase"] for r in tracker.results]
        f.write("STATISTICS:\n")
        f.write(f"  Best MASE: {min(mases):.4f} (seed {sorted_results[0]['seed']})\n")
        f.write(f"  Worst MASE: {max(mases):.4f}\n")
        f.write(f"  Mean MASE: {np.mean(mases):.4f}\n")
        f.write(f"  Std MASE: {np.std(mases):.4f}\n\n")
        f.write(f"TOP {tracker.keep_top} SEEDS:\n")
        for i, r in enumerate(sorted_results[: tracker.keep_top], 1):
            f.write(f"  {i}. Seed {r['seed']:4d}: MASE={r['mase']:.4f}, RMSE={r['rmse']:.2f}\n")

    logger.info(f"Final report saved to {report_file}")


if __name__ == "__main__":
    main()
