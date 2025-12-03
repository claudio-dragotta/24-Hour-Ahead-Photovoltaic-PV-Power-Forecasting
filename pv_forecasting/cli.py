"""Command-line interface for PV power forecasting.

Provides easy-to-use commands for training models and making predictions:
- pv-train: Train a forecasting model (CNN-BiLSTM, TFT, or LightGBM)
- pv-predict: Generate predictions using a trained model

Example usage:
    $ pv-train cnn --config configs/cnn_bilstm.yaml
    $ pv-train lgbm --config configs/lgbm.yaml --epochs 100
    $ pv-predict --model outputs/cnn_bilstm/model_best.keras --data data/processed/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import load_config
from .logger import setup_logger


def train():
    """Entry point for 'pv-train' command."""
    parser = argparse.ArgumentParser(
        description="Train PV power forecasting models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection (positional argument)
    parser.add_argument(
        "model",
        choices=["cnn", "tft", "lgbm"],
        help="Model type to train (cnn=CNN-BiLSTM, tft=Temporal Fusion Transformer, lgbm=LightGBM)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file (overrides defaults)",
    )

    # Data paths (override config)
    parser.add_argument("--pv-path", type=Path, help="Path to PV dataset (Excel file)")
    parser.add_argument("--wx-path", type=Path, help="Path to weather dataset (Excel file)")
    parser.add_argument("--local-tz", type=str, help="Local timezone (e.g., 'Australia/Sydney')")

    # Model hyperparameters (override config)
    parser.add_argument("--horizon", type=int, help="Forecast horizon in hours")
    parser.add_argument("--seq-len", type=int, help="Input sequence length (hours) for sequence models")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")

    # Output
    parser.add_argument("--output-dir", type=Path, help="Output directory for model and results")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(level=log_level)

    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        # Load default config based on model type
        default_configs = {
            "cnn": Path("configs/cnn_bilstm.yaml"),
            "tft": Path("configs/tft.yaml"),
            "lgbm": Path("configs/lgbm.yaml"),
        }
        default_config_path = default_configs.get(args.model)
        if default_config_path and default_config_path.exists():
            logger.info(f"Loading default config for {args.model} from {default_config_path}")
            config = load_config(default_config_path)
        else:
            logger.warning(f"No config file provided. Using hardcoded defaults.")
            config = load_config()

    # Override config with command-line arguments
    if args.pv_path:
        config.data.pv_path = args.pv_path
    if args.wx_path:
        config.data.wx_path = args.wx_path
    if args.local_tz:
        config.data.local_tz = args.local_tz
    if args.horizon:
        config.model.horizon = args.horizon
    if args.seq_len:
        config.model.seq_len = args.seq_len
    if args.epochs:
        config.model.epochs = args.epochs
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.learning_rate:
        config.model.learning_rate = args.learning_rate
    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.seed:
        config.model.seed = args.seed

    # Set model type based on CLI argument
    config.model.model_type = {"cnn": "cnn_bilstm", "tft": "tft", "lgbm": "lgbm"}[args.model]

    logger.info(f"Training {config.model.model_type} model with horizon={config.model.horizon}h")

    # Import and run appropriate training function
    try:
        if args.model == "cnn":
            from .training.train_cnn import train_cnn_bilstm

            train_cnn_bilstm(config)
        elif args.model == "tft":
            from .training.train_tft import train_tft

            train_tft(config)
        elif args.model == "lgbm":
            from .training.train_lgbm import train_lightgbm

            train_lightgbm(config)
    except ImportError as e:
        logger.error(f"Failed to import training module: {e}")
        logger.error("Make sure training modules are properly set up.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Training completed successfully!")


def predict():
    """Entry point for 'pv-predict' command."""
    parser = argparse.ArgumentParser(
        description="Generate predictions with trained PV forecasting model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model and data paths
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.keras, .pth, or directory with LightGBM models)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to processed data file (.parquet) or directory with raw data",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file used during training",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Output file for predictions (CSV format)",
    )

    # Model type (auto-detect from file extension if not provided)
    parser.add_argument(
        "--model-type",
        choices=["cnn", "tft", "lgbm"],
        help="Model type (auto-detected if not specified)",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(level=log_level)

    # Auto-detect model type if not provided
    model_type = args.model_type
    if not model_type:
        if args.model.suffix in [".keras", ".h5"]:
            model_type = "cnn"
        elif args.model.suffix in [".pth", ".pt", ".ckpt"]:
            model_type = "tft"
        elif args.model.is_dir():
            model_type = "lgbm"
        else:
            logger.error("Could not auto-detect model type. Please specify --model-type")
            sys.exit(1)

    logger.info(f"Generating predictions with {model_type} model")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")

    # Load configuration if provided
    config = load_config(args.config) if args.config else load_config()

    # Import and run prediction
    try:
        from .inference.predict import predict as run_prediction

        run_prediction(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output,
            model_type=model_type,
            config=config,
        )
    except ImportError as e:
        logger.error(f"Failed to import prediction module: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    # For testing: python -m pv_forecasting.cli train --help
    train()
