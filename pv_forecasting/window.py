"""Sliding window utilities for sequence-to-sequence forecasting models.

This module provides functions to convert time series data into sliding window
format suitable for training sequence models (RNN, LSTM, CNN-BiLSTM, TFT, etc.).
Also includes time-aware train/validation/test splitting that preserves temporal order.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    target_col: str,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Create sliding windows from time series data for sequence-to-sequence models.

    Converts a DataFrame with time series features into supervised learning format
    by creating overlapping sliding windows. Each window contains:
    - Input sequence: seq_len timesteps of all features
    - Target sequence: horizon timesteps of target variable (future values)

    This format is required for training sequence models like LSTM, CNN-BiLSTM,
    and Temporal Fusion Transformer (TFT).

    Args:
        df: Input dataframe with time series data.
            - Index: DatetimeIndex (timezone-aware, sorted chronologically)
            - Columns: Features including the target variable
        target_col: Name of target column to predict (e.g., "pv").
            This column will be extracted for Y (target sequences).
        seq_len: Length of input sequence (lookback window) in timesteps.
            Example: 48 for 48-hour lookback with hourly data.
        horizon: Length of target sequence (forecast horizon) in timesteps.
            Example: 24 for 24-hour ahead forecasting with hourly data.

    Returns:
        Tuple containing:
            - X: Input sequences, shape (N, seq_len, n_features)
                N = number of windows
                seq_len = input sequence length
                n_features = number of feature columns in df
            - Y: Target sequences, shape (N, horizon)
                N = number of windows
                horizon = forecast horizon length
            - origins: List of timestamps marking the end of each input window.
                Length N. Used for aligning predictions with original time series.

    Example:
        >>> # Hourly data with 100 timesteps and 5 features
        >>> df = pd.DataFrame(
        ...     np.random.rand(100, 5),
        ...     columns=['pv', 'ghi', 'temp', 'hour_sin', 'hour_cos'],
        ...     index=pd.date_range('2020-01-01', periods=100, freq='H', tz='UTC')
        ... )
        >>> X, Y, origins = make_windows(df, target_col='pv', seq_len=48, horizon=24)
        >>> X.shape
        (29, 48, 5)  # 29 windows, 48-hour input, 5 features
        >>> Y.shape
        (29, 24)  # 29 windows, 24-hour target
        >>> len(origins)
        29
        >>> origins[0]
        Timestamp('2020-01-03 00:00:00+0000', tz='UTC')

    Note:
        - Windows are created with stride=1 (maximum overlap for training data)
        - Number of windows = len(df) - seq_len - horizon + 1
        - Origins list helps map predictions back to original timestamps
        - All features in df are included in X (multivariate input)
        - Only target_col is included in Y (univariate or single-target output)
        - For 24-hour ahead forecasting with hourly data: horizon=24
        - For 48-hour lookback with hourly data: seq_len=48
        - Returns float32 arrays for memory efficiency

    Warning:
        - Input dataframe must be sorted chronologically
        - Missing values (NaN) should be handled before calling this function
        - Ensure sufficient data: len(df) >= seq_len + horizon
        - Large seq_len or many features can create very large arrays

    See Also:
        - chronological_split: Split windows into train/val/test sets
        - pv_forecasting.model: CNN-BiLSTM model that uses these windows
    """
    values = df.values
    y_idx = df.columns.get_loc(target_col)
    n = len(df)
    n_features = values.shape[1]
    samples = n - seq_len - horizon + 1
    X = np.zeros((samples, seq_len, n_features), dtype=np.float32)
    Y = np.zeros((samples, horizon), dtype=np.float32)
    origins: list[pd.Timestamp] = []
    for i in range(samples):
        X[i] = values[i : i + seq_len]
        Y[i] = values[i + seq_len : i + seq_len + horizon, y_idx]
        origins.append(df.index[i + seq_len - 1])
    return X, Y, origins


def chronological_split(
    X: np.ndarray, Y: np.ndarray, origins: list[pd.Timestamp], train_ratio=0.7, val_ratio=0.1
) -> Tuple[tuple, tuple, tuple]:
    """Split windowed data into train/validation/test sets preserving temporal order.

    For time series forecasting, it's critical to split data chronologically
    (not randomly) to prevent information leakage from future to past. This
    function splits windowed sequences in temporal order:
    - Train set: earliest windows (first train_ratio of data)
    - Validation set: middle windows (next val_ratio of data)
    - Test set: latest windows (remaining data)

    This mimics real-world deployment where models train on historical data
    and predict future unseen data.

    Args:
        X: Input sequences from make_windows, shape (N, seq_len, n_features).
            N = total number of windows
        Y: Target sequences from make_windows, shape (N, horizon).
        origins: List of timestamps from make_windows, length N.
            Each timestamp marks the end of the input window (start of forecast).
        train_ratio: Fraction of data for training. Default: 0.7 (70%).
        val_ratio: Fraction of data for validation. Default: 0.1 (10%).
            Test ratio is implicitly 1 - train_ratio - val_ratio (typically 20%).

    Returns:
        Tuple of three tuples, each containing (X_subset, Y_subset, origins_subset):
            - train: (X_train, Y_train, origins_train) - earliest windows
            - val: (X_val, Y_val, origins_val) - middle windows
            - test: (X_test, Y_test, origins_test) - latest windows

    Example:
        >>> # Create windows from 1000 hours of data
        >>> X, Y, origins = make_windows(df, 'pv', seq_len=48, horizon=24)
        >>> X.shape
        (929, 48, 10)
        >>> # Split: 70% train, 10% val, 20% test
        >>> train, val, test = chronological_split(X, Y, origins, train_ratio=0.7, val_ratio=0.1)
        >>> X_train, Y_train, origins_train = train
        >>> X_val, Y_val, origins_val = val
        >>> X_test, Y_test, origins_test = test
        >>> X_train.shape, X_val.shape, X_test.shape
        ((650, 48, 10), (92, 48, 10), (187, 48, 10))
        >>> # Verify temporal order
        >>> origins_train[-1] < origins_val[0] < origins_test[0]
        True

    Note:
        - Split boundaries use floor division (int conversion)
        - With default ratios (0.7, 0.1): train=70%, val=10%, test=20%
        - Common alternative ratios: (0.8, 0.1, 0.1) or (0.6, 0.2, 0.2)
        - Origins list helps verify temporal order and align predictions
        - Test set represents the most recent (future) data
        - NO shuffling or random sampling (maintains temporal order)
        - Use validation set for hyperparameter tuning and early stopping
        - Use test set ONLY for final evaluation (never for training/tuning)

    Warning:
        - NEVER use random splitting for time series (causes data leakage)
        - Ensure ratios sum to ≤ 1.0 (typically exactly 1.0)
        - Small datasets may have very few validation/test samples
        - For walk-forward validation, use a different approach (see lgbm_train.py)

    See Also:
        - make_windows: Creates X, Y, origins inputs for this function
        - train.py: Example usage for CNN-BiLSTM training
        - lgbm_train.py: Alternative walk-forward validation approach

    References:
        Time series cross-validation:
        - Tashman, L. J. (2000). "Out-of-sample tests of forecasting accuracy:
          an analysis and review." International Journal of Forecasting.
        - Hyndman & Athanasopoulos (2021). "Forecasting: Principles and Practice"
          Chapter 5: Time series cross-validation.
    """
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)
    return (
        (X[idx_train], Y[idx_train], origins[idx_train]),
        (X[idx_val], Y[idx_val], origins[idx_val]),
        (X[idx_test], Y[idx_test], origins[idx_test]),
    )

