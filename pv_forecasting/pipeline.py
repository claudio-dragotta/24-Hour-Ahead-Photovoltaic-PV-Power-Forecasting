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
DEFAULT_ROLLING_HOURS: Tuple[int, ...] = (3, 6)


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
    """Load raw PV/weather data, align on UTC index, and build engineered features.

    Returns a frame with UTC DatetimeIndex, standardized lowercase weather names,
    cyclic time features, solar/clearsky physics features, and lag/rolling stats.
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
        df = df.dropna()

    df = df.copy()
    df["time_idx"] = np.arange(len(df))
    df["series_id"] = "pv_site_1"
    return df


def persist_processed(df: pd.DataFrame, out_dir: Path) -> Path:
    """Save the processed frame to Parquet in `out_dir/processed.parquet`."""
    return save_processed(df, out_dir)
