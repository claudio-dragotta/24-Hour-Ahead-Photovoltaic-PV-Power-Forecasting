"""Data loading and processing utilities for PV forecasting.

This module handles loading PV production and weather data from Excel files,
timezone alignment, and dataset merging with proper handling of DST transitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from .timeutils import localize_pv_index, parse_wx_index_with_tz, to_utc


def load_pv_xlsx(path: Path, local_tz: str) -> pd.DataFrame:
    """Load PV production data from multi-sheet Excel file.

    Reads all sheets, converts local naive timestamps to UTC timezone-aware,
    and concatenates into a single time series. Handles millisecond precision
    timestamps and removes duplicates.

    Args:
        path: Path to Excel file containing PV data.
            Expected format: First row is metadata (e.g., "Max kWp 82.41"),
            second row onwards has columns [timestamp, pv_power].
        local_tz: Local timezone for the PV site (IANA timezone name).
            Example: "Australia/Sydney", "Europe/Rome", "US/Pacific".

    Returns:
        DataFrame with:
            - Index: UTC timezone-aware DatetimeIndex
            - Column 'pv': Normalized PV power output [0-1]

    Example:
        >>> df = load_pv_xlsx(Path("data/raw/pv_dataset.xlsx"), "Australia/Sydney")
        >>> df.index.tz
        <UTC>
        >>> df['pv'].max()
        0.987

    Note:
        - Handles DST transitions automatically (ambiguous=False, nonexistent='shift_forward')
        - Removes duplicate timestamps (keeps first occurrence)
        - Converts all values to numeric (coerces errors to NaN)
        - Multiple sheets are concatenated chronologically
        - Original timestamps often have millisecond precision (.985, .980, etc.)
    """
    # Read all sheets in the Excel file
    xl = pd.ExcelFile(path)
    dfs = []

    for sheet_name in xl.sheet_names:
        # Skip first row (header like "Max kWp  82.41"), use second row as header
        df = pd.read_excel(path, sheet_name=sheet_name, header=None, skiprows=1)
        # First column is timestamp, second is PV power
        df.columns = ["timestamp", "pv"]

        idx_local = localize_pv_index(df["timestamp"], local_tz)
        idx_utc = to_utc(idx_local)
        # Reset index name to avoid conflicts
        idx_utc.name = None
        pv_vals = pd.to_numeric(df["pv"], errors="coerce").values
        sheet_df = pd.DataFrame({"pv": pv_vals}, index=idx_utc)
        dfs.append(sheet_df)

    # Concatenate all sheets
    out = pd.concat(dfs, axis=0)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def load_wx_xlsx(path: Path) -> pd.DataFrame:
    """Load weather data from multi-sheet Excel file.

    Reads all sheets, parses timezone-aware timestamps, converts to UTC,
    and concatenates weather variables into a single time series.

    Args:
        path: Path to Excel file containing weather data.
            Expected format: First row is header with column names.
            One column must be timestamp (dt_iso, timestamp, or first column).
            Other columns are weather variables (GHI, DNI, DHI, temp, etc.).

    Returns:
        DataFrame with:
            - Index: UTC timezone-aware DatetimeIndex
            - Columns: Weather variables (GHI, DNI, DHI, temp, humidity, etc.)

    Example:
        >>> df = load_wx_xlsx(Path("data/raw/wx_dataset.xlsx"))
        >>> df.index.tz
        <UTC>
        >>> df.columns.tolist()
        ['GHI', 'DNI', 'DHI', 'T', 'RH', 'WS', 'rain_1h', 'clouds_all', ...]

    Note:
        - Automatically detects timestamp column (dt_iso, timestamp, or first column)
        - Converts all columns to numeric where possible (non-numeric become NaN)
        - Handles timezone-aware timestamps with explicit UTC offset
        - Removes duplicate timestamps (keeps first occurrence)
        - Multiple sheets are concatenated chronologically
        - Non-numeric columns are excluded from final output
    """
    # Read all sheets in the Excel file
    xl = pd.ExcelFile(path)
    dfs = []

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        cols = {c.lower(): c for c in df.columns}
        ts_col = cols.get("dt_iso") or cols.get("timestamp") or list(df.columns)[0]

        idx_parsed = parse_wx_index_with_tz(df[ts_col])
        idx_utc = to_utc(idx_parsed)

        drop_cols: Iterable[str] = [ts_col]
        feat_df = df.drop(columns=list(drop_cols), errors="ignore")
        # Cast numeric where possible (coerce non-numeric to NaN)
        # EXCEPT for weather_description which is categorical and will be encoded later
        for c in feat_df.columns:
            if 'weather' in c.lower() and 'description' in c.lower():
                continue  # Keep weather_description as string for encoding later
            try:
                feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
            except (ValueError, TypeError):
                # Keep original values if conversion fails entirely
                pass
        sheet_df = feat_df.copy()
        sheet_df.index = idx_utc
        dfs.append(sheet_df)

    # Concatenate all sheets
    out = pd.concat(dfs, axis=0)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def align_hourly(pv: pd.DataFrame, wx: pd.DataFrame, keep_wx_future: bool = False) -> pd.DataFrame:
    """Align PV and weather data on UTC timestamps with forward-fill.

    Merges PV production and weather data, handling different temporal resolutions.
    Weather data is forward-filled to match PV timestamps (meteorological variables
    change slowly, so last observation carried forward is appropriate).

    Args:
        pv: PV production dataframe with UTC DatetimeIndex and 'pv' column.
        wx: Weather dataframe with UTC DatetimeIndex and weather columns.
        keep_wx_future: If True, keeps rows with weather data but no PV data
            (useful for inference when future weather forecasts are available).
            If False, only keeps rows with PV observations. Default: False.

    Returns:
        DataFrame with:
            - Index: UTC timezone-aware DatetimeIndex (sorted)
            - Column 'pv': PV power output
            - Other columns: Weather variables (forward-filled)

    Example:
        >>> pv_df = load_pv_xlsx(Path("pv_dataset.xlsx"), "Australia/Sydney")
        >>> wx_df = load_wx_xlsx(Path("wx_dataset.xlsx"))
        >>> merged = align_hourly(pv_df, wx_df)
        >>> merged.columns.tolist()
        ['pv', 'GHI', 'DNI', 'DHI', 'T', 'RH', ...]
        >>> merged.index.is_monotonic_increasing
        True

    Note:
        - Uses outer join to preserve all timestamps
        - Weather data forward-filled (last observation carried forward)
        - PV data is NOT filled (missing values remain NaN)
        - Removes duplicate timestamps (keeps first)
        - Only numeric weather columns are retained
        - By default, drops rows without PV data (historical training mode)
        - Set keep_wx_future=True for inference with future weather forecasts

    Warning:
        Forward-filling weather data assumes it changes slowly. For variables
        like precipitation, this may not be accurate. Consider using
        interpolation or forecasts for those variables.
    """
    # Join datasets with forward-fill interpolation for weather data
    pv_h = pv.copy()
    wx_h = wx.copy()

    # Keep numeric columns + weather_description (will be encoded later)
    numeric_cols = wx_h.select_dtypes(include=[np.number]).columns.tolist()
    weather_desc_cols = [c for c in wx_h.columns if 'weather' in c.lower() and 'description' in c.lower()]
    keep_cols = numeric_cols + weather_desc_cols
    wx_h = wx_h[keep_cols]

    # Convert to UTC without rounding
    pv_h.index = pv_h.index.tz_convert("UTC")
    wx_h.index = wx_h.index.tz_convert("UTC")

    # Outer join to get all timestamps
    df = pv_h.join(wx_h, how="outer")

    # Sort by index
    df = df.sort_index()

    # Forward-fill weather data to match PV timestamps (weather changes slowly)
    weather_cols = [c for c in df.columns if c != "pv"]
    df[weather_cols] = df[weather_cols].ffill()

    # Keep only rows with PV data unless explicitly keeping future weather-only rows
    if not keep_wx_future:
        df = df[df["pv"].notna()]

    # Remove duplicates if any (keep first occurrence)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    return df


def save_processed(df: pd.DataFrame, out_dir: Path) -> Path:
    """Save processed dataframe to Parquet format.

    Saves the fully processed and feature-engineered dataframe for fast loading
    in subsequent training runs. Parquet format preserves timezone information
    and is much faster than CSV for large datasets.

    Args:
        df: Processed dataframe with features (timezone-aware index).
        out_dir: Output directory where file will be saved.
            Directory is created if it doesn't exist.

    Returns:
        Path to saved file (out_dir / "processed.parquet").

    Example:
        >>> df_processed = align_hourly(pv, wx)
        >>> df_processed = add_time_cyclical(df_processed)
        >>> save_path = save_processed(df_processed, Path("outputs"))
        >>> print(save_path)
        outputs/processed.parquet

    Note:
        - Creates output directory if needed (parents=True, exist_ok=True)
        - Overwrites existing file without warning
        - Parquet preserves:
            - Timezone information
            - Data types (int, float, datetime)
            - Index structure
        - Much faster than CSV for large datasets
        - Smaller file size due to compression
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "processed.parquet"
    df.to_parquet(out_path)
    return out_path


def save_history(history: dict, out_dir: Path) -> Path:
    """Save training history to JSON file.

    Saves model training metrics (loss, validation loss, etc.) for later
    analysis and visualization.

    Args:
        history: Dictionary containing training history.
            Typically from model.fit() or manual logging.
            Example: {'loss': [0.5, 0.3, 0.2], 'val_loss': [0.6, 0.4, 0.25]}
        out_dir: Output directory where file will be saved.
            Directory is created if it doesn't exist.

    Returns:
        Path to saved file (out_dir / "history.json").

    Example:
        >>> history = {'loss': [0.5, 0.3], 'val_loss': [0.6, 0.4]}
        >>> save_path = save_history(history, Path("outputs"))
        >>> print(save_path)
        outputs/history.json

    Note:
        - Creates output directory if needed
        - Overwrites existing file without warning
        - JSON format is human-readable and easy to parse
        - Indented with 2 spaces for readability
        - UTF-8 encoding
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "history.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return p
