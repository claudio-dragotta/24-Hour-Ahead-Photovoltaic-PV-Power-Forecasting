"""Model architectures for PV power forecasting.

This module provides different model architectures:
- CNN-BiLSTM: Convolutional + Bidirectional LSTM (baseline)
- CNN-BiLSTM-Fusion-Attention: Multi-Level Fusion with Attention
- TFT: Temporal Fusion Transformer (via pytorch-forecasting)
- LightGBM: Gradient boosting (wrapper)
"""

from .cnn_bilstm import build_cnn_bilstm, build_cnn_bilstm_fusion_attention

__all__ = ["build_cnn_bilstm", "build_cnn_bilstm_fusion_attention"]
