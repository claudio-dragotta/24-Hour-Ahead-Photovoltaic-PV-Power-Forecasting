"""Timezone handling utilities for PV forecasting.

This module provides robust timezone conversion functions that handle
Daylight Saving Time (DST) transitions correctly. Essential for aligning
PV production data (often in naive local time) with weather data (often UTC).
"""

from __future__ import annotations

import pandas as pd


def localize_pv_index(
    s: pd.Series | pd.DatetimeIndex,
    local_tz: str,
) -> pd.DatetimeIndex:
    """Convert naive local timestamps to timezone-aware with robust DST handling.

    PV production data is often recorded in naive local time (no timezone info).
    This function safely converts to timezone-aware timestamps, handling DST
    transitions that occur in spring (missing hour) and fall (duplicated hour).

    Args:
        s: Naive timestamps to localize (Series or DatetimeIndex).
        local_tz: Target timezone (IANA name like "Australia/Sydney", "Europe/Rome").

    Returns:
        Timezone-aware DatetimeIndex in the specified local timezone.

    Raises:
        ValueError: If any timestamps cannot be parsed.
        RuntimeError: If DST localization fails (with helpful error message).

    Example:
        >>> # Naive timestamps
        >>> naive = pd.DatetimeIndex(['2020-01-01 12:00', '2020-06-15 15:30'])
        >>> localized = localize_pv_index(naive, 'Australia/Sydney')
        >>> localized.tz
        <DstTzInfo 'Australia/Sydney' ...>

    DST Handling:
        Spring Forward (missing hour, e.g., 2:30 AM doesn't exist):
            - Uses nonexistent='shift_forward' to move to next valid time (3:30 AM)
            - Example: 2020-10-04 02:30 → 2020-10-04 03:30 in Australia/Sydney

        Fall Back (duplicated hour, e.g., 2:30 AM occurs twice):
            - Uses ambiguous=False to assume standard time (second occurrence)
            - Avoids the duplicated hour from DST→Standard transition

    Note:
        - Input must be naive (no timezone info)
        - All timestamps must be parsable (errors='coerce' then checked for NaN)
        - DST rules depend on timezone (Southern hemisphere is opposite to Northern)
        - Sydney DST: October to April (opposite to Europe/US)
    """
    if isinstance(s, pd.Series):
        ts = pd.DatetimeIndex(pd.to_datetime(s, utc=False, errors="coerce"))
    else:
        ts = pd.DatetimeIndex(pd.to_datetime(s, utc=False, errors="coerce"))
    if ts.isna().any():
        bad = ts[ts.isna()]
        raise ValueError(f"Found non-parsable PV timestamps: {bad[:5]}")
    try:
        # Use False for ambiguous times (standard time), shift forward for nonexistent
        loc = ts.tz_localize(local_tz, ambiguous=False, nonexistent="shift_forward")
    except Exception as e:
        # Fallback to strict localize to surface the issue clearly
        raise RuntimeError(
            "Failed to localize PV timestamps. Try adjusting `local_tz` or "
            "manually inspecting the DST transition period."
        ) from e
    return loc


def parse_wx_index_with_tz(
    s: pd.Series | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Parse timezone-aware weather timestamps (with explicit UTC offset).

    Weather data from APIs often includes explicit timezone offsets
    (e.g., "2020-01-01T12:00:00+10:00"). This function parses these
    strings directly to UTC-aware timestamps.

    Args:
        s: Timestamps with explicit timezone info (strings or DatetimeIndex).
            Expected format: ISO 8601 with offset (e.g., "2020-01-01T12:00:00+10:00").

    Returns:
        Timezone-aware DatetimeIndex in UTC.

    Raises:
        ValueError: If any timestamps cannot be parsed.

    Example:
        >>> timestamps = pd.Series([
        ...     '2020-01-01T12:00:00+10:00',
        ...     '2020-01-01T15:30:00+10:00',
        ... ])
        >>> utc_index = parse_wx_index_with_tz(timestamps)
        >>> utc_index.tz
        <UTC>
        >>> utc_index[0]
        Timestamp('2020-01-01 02:00:00+0000', tz='UTC')  # Converted from +10

    Note:
        - Input must have explicit timezone info (offset like +10:00)
        - Automatically converts to UTC (utc=True)
        - Handles various ISO 8601 formats
        - errors='coerce' then checked for NaN (raises if any unparsable)
    """
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts.isna().any():
        # If strings contain explicit offsets like '+10:00', pandas with utc=True handles it
        bad = ts[ts.isna()]
        raise ValueError(f"Found non-parsable weather timestamps: {bad[:5]}")
    return ts


def to_utc(idx: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    """Convert timezone-aware index to UTC.

    Convenience function to standardize all timestamps to UTC timezone.
    Use this after localization to ensure all data is in the same timezone
    before merging.

    Args:
        idx: Timezone-aware DatetimeIndex or Series to convert.

    Returns:
        DatetimeIndex in UTC timezone.

    Raises:
        ValueError: If index is not timezone-aware (naive timestamps).

    Example:
        >>> # Local timezone-aware index
        >>> local = pd.DatetimeIndex(['2020-01-01 12:00'], tz='Australia/Sydney')
        >>> utc = to_utc(local)
        >>> utc.tz
        <UTC>
        >>> utc[0]
        Timestamp('2020-01-01 02:00:00+0000', tz='UTC')  # Sydney is UTC+10

    Note:
        - Input MUST be timezone-aware (will raise error if naive)
        - Preserves the actual moment in time (only changes representation)
        - All data should be in UTC before merging datasets
        - UTC has no DST transitions (simplifies downstream processing)
    """
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        raise ValueError("Index must be timezone-aware to convert to UTC")
    return idx.tz_convert("UTC")
