"""Model architectures for PV power forecasting.

This module provides different model architectures:
- CNN-BiLSTM: Convolutional + Bidirectional LSTM (baseline)
- CNN-BiLSTM-Fusion-Attention: Multi-Level Fusion with Attention
- MultiBranchTransformer: Multi-branch architecture with hierarchical attention fusion
- TFT: Temporal Fusion Transformer (via pytorch-forecasting)
- LightGBM: Gradient boosting (wrapper)
"""

from .cnn_bilstm import build_cnn_bilstm, build_cnn_bilstm_fusion_attention
from .multi_branch_tft import MultiBranchTransformer

__all__ = [
    "build_cnn_bilstm",
    "build_cnn_bilstm_fusion_attention",
    "MultiBranchTransformer"
]
