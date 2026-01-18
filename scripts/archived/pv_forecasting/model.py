"""CNN-BiLSTM model architecture for multi-horizon PV power forecasting.

DEPRECATED: This module is kept for backward compatibility.
New code should import from pv_forecasting.models.cnn_bilstm instead.

This module defines the deep learning model used for 24-hour ahead photovoltaic
power forecasting. The architecture combines:
- 1D Convolutional layers for feature extraction from time series
- Bidirectional LSTM for capturing temporal dependencies in both directions
- Dense output layer for multi-horizon forecasting

The model is designed to forecast multiple future timesteps simultaneously
(direct multi-horizon strategy) rather than iterative one-step-ahead prediction.
"""

from __future__ import annotations

import warnings
from typing import Tuple

from tensorflow import keras

# Import from new location for backward compatibility
from .models.cnn_bilstm import build_cnn_bilstm as _build_cnn_bilstm


def build_cnn_bilstm(input_shape: Tuple[int, int], horizon: int) -> keras.Model:
    """Build CNN-BiLSTM model for multi-horizon time series forecasting.

    DEPRECATED: This function is kept for backward compatibility.
    New code should import from pv_forecasting.models.cnn_bilstm instead.

    See pv_forecasting.models.cnn_bilstm.build_cnn_bilstm for full documentation.

    Args:
        input_shape: Shape of input sequences as (seq_len, n_features).
        horizon: Number of future timesteps to predict.

    Returns:
        Compiled Keras model.
    """
    warnings.warn(
        "Importing build_cnn_bilstm from pv_forecasting.model is deprecated. "
        "Please use 'from pv_forecasting.models import build_cnn_bilstm' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _build_cnn_bilstm(input_shape, horizon)
