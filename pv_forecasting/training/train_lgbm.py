"""LightGBM training module - wrapper around lgbm_train.py."""

from __future__ import annotations

from ..config import Config
from ..logger import get_logger

logger = get_logger(__name__)


def train_lightgbm(config: Config) -> None:
    """Train LightGBM multi-horizon model.

    Args:
        config: Configuration object.

    Note:
        This is a temporary wrapper. Full integration pending.
        For now, use: python lgbm_train.py
    """
    logger.error("LightGBM training via CLI not yet fully implemented.")
    logger.info("Please use: python lgbm_train.py --help")
    raise NotImplementedError("Use lgbm_train.py script directly for now")
