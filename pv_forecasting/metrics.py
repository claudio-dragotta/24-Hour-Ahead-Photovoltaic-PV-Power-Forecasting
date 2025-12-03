"""Evaluation metrics for PV power forecasting.

This module provides standard and domain-specific metrics for evaluating
solar power forecast accuracy. All metrics operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    RMSE measures the standard deviation of prediction errors. It penalizes
    large errors more heavily than MAE due to squaring.

    Args:
        y_true: Ground truth values (actual observations).
        y_pred: Predicted values.

    Returns:
        RMSE value (same units as input). Always non-negative.

    Example:
        >>> y_true = np.array([0.5, 0.6, 0.7])
        >>> y_pred = np.array([0.52, 0.58, 0.72])
        >>> rmse(y_true, y_pred)
        0.024

    Note:
        - RMSE = 0 indicates perfect predictions
        - RMSE ≥ MAE (equality holds only when all errors are equal)
        - Units are same as target variable (e.g., kW for power)
        - More sensitive to outliers than MAE
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE).

    MAE measures the average magnitude of errors without considering direction.
    It treats all errors equally (no squaring).

    Args:
        y_true: Ground truth values (actual observations).
        y_pred: Predicted values.

    Returns:
        MAE value (same units as input). Always non-negative.

    Example:
        >>> y_true = np.array([0.5, 0.6, 0.7])
        >>> y_pred = np.array([0.52, 0.58, 0.72])
        >>> mae(y_true, y_pred)
        0.02

    Note:
        - MAE = 0 indicates perfect predictions
        - More robust to outliers than RMSE
        - Easier to interpret (average error magnitude)
        - Units are same as target variable
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mase(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, train_series: np.ndarray, m: int = 24) -> float:
    """Calculate Mean Absolute Scaled Error (MASE).

    MASE scales forecast errors relative to a naive seasonal baseline.
    A value < 1 indicates better performance than the naive forecast.

    The naive forecast is: y(t) = y(t-m), where m is the seasonal period.
    For hourly solar data, m=24 represents daily seasonality.

    Args:
        y_true_flat: Ground truth values for test period (1D array).
        y_pred_flat: Predicted values for test period (1D array).
        train_series: Training data used to compute naive baseline error.
            Should be the full training series, not just recent values.
        m: Seasonal period. Default: 24 (daily seasonality for hourly data).
            Common values: 24 (daily), 168 (weekly), 8760 (yearly).

    Returns:
        MASE value (dimensionless). Lower is better.
            - MASE < 1: Better than naive seasonal forecast
            - MASE = 1: Equal to naive forecast
            - MASE > 1: Worse than naive forecast
            - MASE = NaN: If train series has zero variance

    Raises:
        ValueError: If training series is shorter than seasonal period m.

    Example:
        >>> train = np.array([0.4, 0.5, 0.6, 0.7, 0.8] * 30)  # 150 samples
        >>> y_true = np.array([0.85, 0.90, 0.95])
        >>> y_pred = np.array([0.83, 0.91, 0.93])
        >>> mase(y_true, y_pred, train, m=24)
        0.67  # Better than naive forecast

    Note:
        - Scale-independent: can compare across different time series
        - Denominator is computed on training data (prevents overfitting)
        - Handles zero-valued series better than MAPE
        - Recommended metric for time series forecasting competitions
        - Original paper: Hyndman & Koehler (2006)

    References:
        Hyndman, R. J., & Koehler, A. B. (2006). "Another look at measures
        of forecast accuracy." International Journal of Forecasting, 22(4), 679-688.
    """
    train_series = np.asarray(train_series).astype(float)
    if len(train_series) <= m:
        raise ValueError(
            f"Training series too short for MASE computation. "
            f"Need at least {m + 1} samples, got {len(train_series)}."
        )

    # Compute naive forecast error on training data
    # Mean absolute difference between y(t) and y(t-m)
    denom = np.mean(np.abs(train_series[m:] - train_series[:-m]))

    if denom == 0:
        # Training series has no seasonal variation (constant or all zeros)
        return np.nan

    # Compute model's mean absolute error on test data
    err = np.mean(np.abs(y_true_flat - y_pred_flat))

    return float(err / denom)
