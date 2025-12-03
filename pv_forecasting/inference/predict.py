"""Inference module for generating predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import Config
from ..logger import get_logger

logger = get_logger(__name__)


def predict(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    model_type: str,
    config: Optional[Config] = None,
) -> None:
    """Generate predictions using trained model.

    Args:
        model_path: Path to trained model file.
        data_path: Path to input data.
        output_path: Where to save predictions.
        model_type: Type of model ("cnn", "tft", "lgbm").
        config: Optional configuration.

    Note:
        This is a temporary stub. Full integration pending.
        For now, use: python predict.py
    """
    logger.error("Prediction via CLI not yet fully implemented.")
    logger.info("Please use: python predict.py --help")
    raise NotImplementedError("Use predict.py script directly for now")
