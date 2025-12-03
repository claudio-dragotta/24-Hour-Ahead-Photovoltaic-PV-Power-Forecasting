"""TFT training module - wrapper around tft_train.py."""

from __future__ import annotations

from ..config import Config
from ..logger import get_logger

logger = get_logger(__name__)


def train_tft(config: Config) -> None:
    """Train Temporal Fusion Transformer model.

    Args:
        config: Configuration object.

    Note:
        This is a temporary wrapper. Full integration pending.
        For now, use: python tft_train.py
    """
    logger.error("TFT training via CLI not yet fully implemented.")
    logger.info("Please use: python tft_train.py --help")
    raise NotImplementedError("Use tft_train.py script directly for now")
