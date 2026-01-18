"""Model architectures for PV power forecasting.

This module provides different model architectures:
- MultiBranchTransformer: Multi-branch architecture with hierarchical attention fusion
- TFT: Temporal Fusion Transformer (via pytorch-forecasting)
- LightGBM: Gradient boosting (wrapper)
"""

from .multi_branch_tft import MultiBranchTransformer

__all__ = ["MultiBranchTransformer"]
