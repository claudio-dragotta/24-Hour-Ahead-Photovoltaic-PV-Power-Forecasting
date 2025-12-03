"""Complete feature engineering pipeline for PV forecasting.

This module provides end-to-end functions to load raw data and generate
all features needed for training forecasting models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from .data import align_hourly, load_pv_xlsx, load_wx_xlsx, save_processed
from .features import (
    add_clearsky,
    add_kc,
    add_lags,
    add_rollings_h,
    add_solar_position,
    add_time_cyclical,
    standardize_feature_columns,
)


DEFAULT_LAG_HOURS: Tuple[int, ...] = (1, 24, 168)
"""Default lag periods: 1h (recent), 24h (daily), 168h (weekly)."""

DEFAULT_ROLLING_HOURS: Tuple[int, ...] = (3, 6)
"""Default rolling windows: 3h and 6h moving averages."""


def load_and_engineer_features(
    pv_path: Path,
    wx_path: Path,
    local_tz: str,
    lag_hours: Sequence[int] = DEFAULT_LAG_HOURS,
    rolling_hours: Sequence[int] = DEFAULT_ROLLING_HOURS,
    include_solar: bool = True,
    include_clearsky: bool = True,
    dropna: bool = True,
    keep_wx_future: bool = False,
) -> pd.DataFrame:
    """Load raw data and generate complete feature set for PV forecasting.

    This is the main entry point for data preparation. It loads PV and weather
    data, aligns them on UTC timestamps, and applies all feature engineering steps:
    time features, physics-based features (solar position, clear-sky), lags, and
    rolling statistics.

    Args:
        pv_path: Path to PV production Excel file.
        wx_path: Path to weather data Excel file.
        local_tz: Local timezone for PV site (IANA name, e.g., "Australia/Sydney").
        lag_hours: Lag periods in hours. Default: (1, 24, 168) for 1h, 1day, 1week.
        rolling_hours: Rolling window sizes in hours. Default: (3, 6).
        include_solar: If True, adds solar position angles (zenith, azimuth).
            Requires pvlib. Default: True.
        include_clearsky: If True, adds clear-sky irradiance and clearness index.
            Requires pvlib. Default: True.
        dropna: If True, drops rows with NaN values (from lags/rolling windows).
            Set to False if you want to handle NaNs differently. Default: True.
        keep_wx_future: If True, keeps rows with weather but no PV data
            (for inference with forecasts). Default: False.

    Returns:
        DataFrame with:
            - Index: UTC timezone-aware DatetimeIndex (sorted)
            - 'pv': Target variable (normalized power output)
            - Time features: hour_sin/cos, doy_sin/cos
            - Physics features: sp_zenith, sp_azimuth, cs_ghi/dni/dhi, kc
            - Weather: ghi, dni, dhi, temp, humidity, clouds, wind_speed
            - Lag features: pv_lag1/24/168, ghi_lag1/24/168, etc.
            - Rolling features: pv_roll3h/6h, ghi_roll3h/6h, etc.
            - 'time_idx': Sequential integer index (0, 1, 2, ...)
            - 'series_id': Constant identifier "pv_site_1"

    Raises:
        ImportError: If pvlib not installed when include_solar or include_clearsky=True.
        ValueError: If data files not found or have invalid format.

    Example:
        >>> df = load_and_engineer_features(
        ...     pv_path=Path("data/raw/pv_dataset.xlsx"),
        ...     wx_path=Path("data/raw/wx_dataset.xlsx"),
        ...     local_tz="Australia/Sydney",
        ... )
        >>> df.shape
        (17374, 45)  # ~17k samples with ~45 features
        >>> df.columns.tolist()
        ['pv', 'ghi', 'dni', 'hour_sin', 'hour_cos', 'pv_lag1', ...]

    Note:
        - All weather column names are standardized to lowercase
        - Index is sorted chronologically
        - By default, drops NaN rows (caused by lags at start of series)
        - Sets time_idx for TFT/sequential models
        - Sets series_id for multi-series forecasting frameworks
        - Feature count depends on what's available in weather data

    See Also:
        - add_time_cyclical: For time-based cyclical features
        - add_solar_position: For solar angle features
        - add_clearsky: For clear-sky irradiance
        - add_lags: For temporal lag features
        - add_rollings_h: For moving averages
    """
    pv = load_pv_xlsx(Path(pv_path), local_tz)
    wx = load_wx_xlsx(Path(wx_path))
    df = align_hourly(pv, wx, keep_wx_future=keep_wx_future)

    df = standardize_feature_columns(df)
    df = add_time_cyclical(df)

    if include_solar:
        df = add_solar_position(df)
    if include_clearsky:
        df = add_clearsky(df)
    if include_clearsky and "ghi" in df.columns:
        df = add_kc(df, ghi_col="ghi")

    lag_cols = [c for c in ["pv", "ghi", "dni", "dhi"] if c in df.columns]
    if lag_hours and lag_cols:
        df = add_lags(df, lag_cols, list(lag_hours))

    roll_cols = [c for c in ["pv", "ghi", "dni"] if c in df.columns]
    if rolling_hours and roll_cols:
        df = add_rollings_h(df, roll_cols, list(rolling_hours))

    df = df.sort_index()
    if dropna:
        # Drop rows with NaN only in critical numeric columns (lag features, target, irradiance)
        # Keep text columns like 'weather_description' even if NaN
        critical_cols = [c for c in df.columns if any(x in c for x in ['pv', 'ghi', 'dni', 'dhi', 'lag', 'roll'])]
        df = df.dropna(subset=critical_cols)

    df = df.copy()
    df["time_idx"] = np.arange(len(df))
    df["series_id"] = "pv_site_1"
    return df


def persist_processed(df: pd.DataFrame, out_dir: Path) -> Path:
    """Save processed feature-engineered data to Parquet format.

    Convenience wrapper around save_processed that saves to
    'out_dir/processed.parquet'.

    Args:
        df: Feature-engineered dataframe to save.
        out_dir: Output directory (will be created if needed).

    Returns:
        Path to saved file.

    Example:
        >>> df = load_and_engineer_features(pv_path, wx_path, local_tz)
        >>> save_path = persist_processed(df, Path("outputs"))
        >>> print(save_path)
        outputs/processed.parquet

    Note:
        - Creates directory if it doesn't exist
        - Overwrites existing file
        - Parquet format preserves all data types and timezone info
    """
    return save_processed(df, out_dir)
