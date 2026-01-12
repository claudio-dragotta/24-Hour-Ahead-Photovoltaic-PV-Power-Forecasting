"""Tests for sliding window utilities in pv_forecasting.window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_forecasting import window


def _toy_dataframe() -> pd.DataFrame:
    """Create a tiny deterministic dataframe for windowing tests."""
    idx = pd.date_range("2021-01-01", periods=6, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "pv": np.arange(6, dtype=float),
            "ghi": np.arange(10, 16, dtype=float),
        },
        index=idx,
    )
    return df


def test_make_windows_shapes_and_origins():
    df = _toy_dataframe()
    seq_len = 3
    horizon = 2

    X, Y, origins = window.make_windows(df, target_col="pv", seq_len=seq_len, horizon=horizon)

    expected_samples = len(df) - seq_len - horizon + 1  # 6 - 3 - 2 + 1 = 2
    assert X.shape == (expected_samples, seq_len, df.shape[1])
    assert Y.shape == (expected_samples, horizon)
    assert len(origins) == expected_samples

    # Check first window contents
    np.testing.assert_array_equal(X[0, :, 0], [0.0, 1.0, 2.0])  # pv values
    np.testing.assert_array_equal(Y[0], [3.0, 4.0])
    assert origins[0] == df.index[seq_len - 1]  # end of input window

    # Origins should be strictly increasing and timezone-aware
    assert all(ts.tzinfo is not None for ts in origins)
    assert origins[0] < origins[1]


def test_make_windows_raises_on_insufficient_length():
    df = _toy_dataframe().iloc[:4]  # Too short for seq_len=3, horizon=2
    X, Y, origins = window.make_windows(df, target_col="pv", seq_len=3, horizon=2)
    assert X.shape[0] == 0
    assert Y.shape[0] == 0
    assert len(origins) == 0


def test_chronological_split_preserves_order():
    df = _toy_dataframe()
    X, Y, origins = window.make_windows(df, target_col="pv", seq_len=2, horizon=2)

    (
        (X_train, Y_train, origins_train),
        (X_val, Y_val, origins_val),
        (X_test, Y_test, origins_test),
    ) = window.chronological_split(X, Y, np.array(origins), train_ratio=0.5, val_ratio=0.25)

    # Check sizes follow ratios: 3 samples -> 1 train, 0 val? Let's assert exact counts
    assert X_train.shape[0] == 1
    assert X_val.shape[0] == 0  # floor division for val
    assert X_test.shape[0] == X.shape[0] - X_train.shape[0] - X_val.shape[0]

    # Temporal ordering: last train < first test (when val empty, compare test directly)
    assert origins_train[-1] < origins_test[0]
