"""Model architectures for PV power forecasting.

This module provides different model architectures:
- CNN-BiLSTM: Convolutional + Bidirectional LSTM
- TFT: Temporal Fusion Transformer (via pytorch-forecasting)
- LightGBM: Gradient boosting (wrapper)
"""

from .cnn_bilstm import build_cnn_bilstm

__all__ = ["build_cnn_bilstm"]
