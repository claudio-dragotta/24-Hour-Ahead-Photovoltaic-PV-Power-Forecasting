"""Configuration management for PV forecasting models.

This module provides centralized configuration loading from YAML files,
with defaults and validation. Supports multiple model types (CNN-BiLSTM,
TFT, LightGBM) and different experiment configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    pv_path: Path = Path("data/raw/pv_dataset.xlsx")
    wx_path: Path = Path("data/raw/wx_dataset.xlsx")
    local_tz: str = "Australia/Sydney"
    processed_dir: Path = Path("data/processed")
    lag_hours: List[int] = field(default_factory=lambda: [1, 24, 168])
    rolling_hours: List[int] = field(default_factory=lambda: [3, 6])
    include_solar: bool = True
    include_clearsky: bool = True
    dropna: bool = True


@dataclass
class ModelConfig:
    """Model architecture and training configuration."""

    model_type: str = "cnn_bilstm"  # "cnn_bilstm", "tft", "lgbm"
    horizon: int = 24
    seq_len: int = 168  # For sequence models (CNN-BiLSTM, TFT)
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    seed: int = 42
    early_stopping_patience: int = 8


@dataclass
class LightGBMConfig:
    """LightGBM-specific hyperparameters."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 7
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    use_future_meteo: bool = False
    future_cols: List[str] = field(default_factory=lambda: ["ghi", "dni", "dhi", "temp"])


@dataclass
class OutputConfig:
    """Output paths and logging configuration."""

    output_dir: Path = Path("outputs")
    save_predictions: bool = True
    save_model: bool = True
    save_history: bool = True
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> Config:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Config object with values from YAML file.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If config file doesn't exist.
            ValueError: If YAML is invalid.

        Example:
            >>> config = Config.from_yaml(Path("configs/cnn_bilstm.yaml"))
            >>> print(config.model.horizon)
            24
        """
        if not HAS_YAML:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

        # Parse nested config sections
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        lgbm_config = LightGBMConfig(**config_dict.get("lightgbm", {}))
        output_config = OutputConfig(**config_dict.get("output", {}))

        # Convert string paths to Path objects
        data_config.pv_path = Path(data_config.pv_path)
        data_config.wx_path = Path(data_config.wx_path)
        data_config.processed_dir = Path(data_config.processed_dir)
        output_config.output_dir = Path(output_config.output_dir)

        return cls(
            data=data_config,
            model=model_config,
            lightgbm=lgbm_config,
            output=output_config,
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Config:
        """Load configuration from dictionary (e.g., from argparse).

        Args:
            config_dict: Dictionary with configuration values.

        Returns:
            Config object.

        Example:
            >>> config = Config.from_dict({"model": {"horizon": 48}})
            >>> config.model.horizon
            48
        """
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        lgbm_config = LightGBMConfig(**config_dict.get("lightgbm", {}))
        output_config = OutputConfig(**config_dict.get("output", {}))

        return cls(
            data=data_config,
            model=model_config,
            lightgbm=lgbm_config,
            output=output_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of configuration.

        Example:
            >>> config = Config()
            >>> config_dict = config.to_dict()
            >>> config_dict["model"]["horizon"]
            24
        """
        return {
            "data": {
                "pv_path": str(self.data.pv_path),
                "wx_path": str(self.data.wx_path),
                "local_tz": self.data.local_tz,
                "processed_dir": str(self.data.processed_dir),
                "lag_hours": self.data.lag_hours,
                "rolling_hours": self.data.rolling_hours,
                "include_solar": self.data.include_solar,
                "include_clearsky": self.data.include_clearsky,
                "dropna": self.data.dropna,
            },
            "model": {
                "model_type": self.model.model_type,
                "horizon": self.model.horizon,
                "seq_len": self.model.seq_len,
                "epochs": self.model.epochs,
                "batch_size": self.model.batch_size,
                "learning_rate": self.model.learning_rate,
                "train_ratio": self.model.train_ratio,
                "val_ratio": self.model.val_ratio,
                "seed": self.model.seed,
                "early_stopping_patience": self.model.early_stopping_patience,
            },
            "lightgbm": {
                "n_estimators": self.lightgbm.n_estimators,
                "learning_rate": self.lightgbm.learning_rate,
                "max_depth": self.lightgbm.max_depth,
                "num_leaves": self.lightgbm.num_leaves,
                "min_child_samples": self.lightgbm.min_child_samples,
                "subsample": self.lightgbm.subsample,
                "colsample_bytree": self.lightgbm.colsample_bytree,
                "reg_alpha": self.lightgbm.reg_alpha,
                "reg_lambda": self.lightgbm.reg_lambda,
                "use_future_meteo": self.lightgbm.use_future_meteo,
                "future_cols": self.lightgbm.future_cols,
            },
            "output": {
                "output_dir": str(self.output.output_dir),
                "save_predictions": self.output.save_predictions,
                "save_model": self.output.save_model,
                "save_history": self.output.save_history,
                "log_level": self.output.log_level,
            },
        }

    def save_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            output_path: Path where to save the YAML file.

        Raises:
            ImportError: If PyYAML is not installed.

        Example:
            >>> config = Config()
            >>> config.save_yaml(Path("outputs/config_used.yaml"))
        """
        if not HAS_YAML:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, indent=2, sort_keys=False)

        logger.info(f"Configuration saved to {output_path}")


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file or return defaults.

    Convenience function that handles missing config files gracefully.

    Args:
        config_path: Path to YAML config file. If None, returns default config.

    Returns:
        Config object with values from YAML or defaults.

    Example:
        >>> # Load from file
        >>> config = load_config(Path("configs/default.yaml"))
        >>> # Use defaults
        >>> config = load_config()
    """
    if config_path is None or not config_path.exists():
        if config_path is not None:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
        return Config()

    try:
        return Config.from_yaml(config_path)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        logger.warning("Using default configuration.")
        return Config()
