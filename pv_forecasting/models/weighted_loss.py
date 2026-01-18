"""Weighted loss functions for PV forecasting."""

import torch
import torch.nn as nn


class PeakWeightedMSELoss(nn.Module):
    """MSE Loss with higher weight for peak production values.

    Penalizes errors more when PV > threshold (e.g., 0.6 normalized).
    This helps the model learn high production patterns better.
    """

    def __init__(self, peak_threshold: float = 0.6, peak_weight: float = 3.0):
        """
        Args:
            peak_threshold: Normalized PV value above which to apply peak weight
            peak_weight: Multiplier for errors above threshold (e.g., 3.0 = 3x penalty)
        """
        super().__init__()
        self.peak_threshold = peak_threshold
        self.peak_weight = peak_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted PV values (normalized [0,1])
            target: True PV values (normalized [0,1])

        Returns:
            Weighted MSE loss
        """
        # Standard MSE
        squared_errors = (pred - target) ** 2

        # Create weight mask: higher weight for peak production
        # Weight = 1.0 for low production, peak_weight for high production
        # Linear interpolation in between
        weights = torch.ones_like(target)

        # Above threshold: apply peak weight
        peak_mask = target > self.peak_threshold
        weights[peak_mask] = self.peak_weight

        # Optional: smooth transition around threshold
        # transition_mask = (target > self.peak_threshold - 0.1) & (target <= self.peak_threshold)
        # if transition_mask.any():
        #     transition_vals = target[transition_mask]
        #     # Linear interpolation from 1.0 to peak_weight
        #     interp_weights = 1.0 + (self.peak_weight - 1.0) * (
        #         (transition_vals - (self.peak_threshold - 0.1)) / 0.1
        #     )
        #     weights[transition_mask] = interp_weights

        # Apply weights
        weighted_loss = (squared_errors * weights).mean()

        return weighted_loss


class AdaptiveWeightedMSELoss(nn.Module):
    """MSE Loss with adaptive weighting based on target value.

    Weight increases smoothly with target value to emphasize
    learning the full range of production.
    """

    def __init__(self, min_weight: float = 1.0, max_weight: float = 5.0):
        """
        Args:
            min_weight: Weight for zero production
            max_weight: Weight for maximum production (normalized 1.0)
        """
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted PV values (normalized [0,1])
            target: True PV values (normalized [0,1])

        Returns:
            Adaptively weighted MSE loss
        """
        # Standard MSE
        squared_errors = (pred - target) ** 2

        # Adaptive weight: linearly increases from min_weight to max_weight
        # as target goes from 0 to 1
        weights = self.min_weight + (self.max_weight - self.min_weight) * target

        # Apply weights
        weighted_loss = (squared_errors * weights).mean()

        return weighted_loss


class QuantileLoss(nn.Module):
    """Quantile loss for asymmetric penalization.

    Useful when you want to penalize over-prediction differently from under-prediction.
    For PV: under-prediction is often worse (missing high production).
    """

    def __init__(self, quantile: float = 0.7):
        """
        Args:
            quantile: Quantile level (0.5 = median, >0.5 penalizes under-prediction more)
        """
        super().__init__()
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted values
            target: True values

        Returns:
            Quantile loss
        """
        errors = target - pred

        # Asymmetric loss
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)

        return loss.mean()
