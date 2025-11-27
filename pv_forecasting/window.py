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
    """Create sliding windows from a time-indexed feature DataFrame.

    Returns X, Y and list of origin timestamps (the last timestamp of the input window).
    X shape: (N, seq_len, n_features)
    Y shape: (N, horizon)
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

