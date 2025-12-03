"""Shared pytest fixtures for pv_forecasting tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_pv_data() -> pd.DataFrame:
    """Create sample PV production data with UTC timezone."""
    dates = pd.date_range("2010-01-01", periods=100, freq="H", tz="UTC")
    pv_values = np.random.rand(100) * 0.8  # Normalized PV values
    return pd.DataFrame({"pv": pv_values}, index=dates)


@pytest.fixture
def sample_weather_data() -> pd.DataFrame:
    """Create sample weather data with UTC timezone."""
    dates = pd.date_range("2010-01-01", periods=100, freq="H", tz="UTC")
    return pd.DataFrame(
        {
            "ghi": np.random.rand(100) * 1000,
            "dni": np.random.rand(100) * 800,
            "dhi": np.random.rand(100) * 300,
            "temp": np.random.rand(100) * 30 + 10,
            "humidity": np.random.rand(100) * 100,
            "wind_speed": np.random.rand(100) * 10,
            "clouds": np.random.rand(100) * 100,
        },
        index=dates,
    )


@pytest.fixture
def sample_merged_data(sample_pv_data, sample_weather_data) -> pd.DataFrame:
    """Create merged PV + weather data."""
    return pd.concat([sample_pv_data, sample_weather_data], axis=1)


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    """Create sample data with engineered features for training."""
    dates = pd.date_range("2010-01-01", periods=200, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "pv": np.random.rand(200) * 0.8,
            "ghi": np.random.rand(200) * 1000,
            "dni": np.random.rand(200) * 800,
            "dhi": np.random.rand(200) * 300,
            "temp": np.random.rand(200) * 30 + 10,
            "hour_sin": np.sin(2 * np.pi * dates.hour / 24),
            "hour_cos": np.cos(2 * np.pi * dates.hour / 24),
            "doy_sin": np.sin(2 * np.pi * dates.dayofyear / 365),
            "doy_cos": np.cos(2 * np.pi * dates.dayofyear / 365),
            "time_idx": np.arange(200),
            "series_id": "pv_site_1",
        },
        index=dates,
    )
    return df


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Create sample predictions and ground truth for metrics tests."""
    np.random.seed(42)
    y_true = np.random.rand(100) * 0.8
    y_pred = y_true + np.random.randn(100) * 0.1
    return y_true, y_pred


@pytest.fixture
def sample_time_series() -> np.ndarray:
    """Create sample time series for baseline computation."""
    np.random.seed(42)
    return np.random.rand(200) * 0.8
