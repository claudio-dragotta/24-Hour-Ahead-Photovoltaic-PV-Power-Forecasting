from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, train_series: np.ndarray, m: int = 24) -> float:
    """Mean Absolute Scaled Error for seasonal period m.

    - Denominator computed on the training series (recommended): mean |Y_t - Y_{t-m}|.
    - y_true_flat and y_pred_flat are 1D arrays over the evaluation horizon(s).
    """
    train_series = np.asarray(train_series).astype(float)
    if len(train_series) <= m:
        raise ValueError("Training series too short for MASE denominator.")
    denom = np.mean(np.abs(train_series[m:] - train_series[:-m]))
    if denom == 0:
        return np.nan
    err = np.mean(np.abs(y_true_flat - y_pred_flat))
    return float(err / denom)

