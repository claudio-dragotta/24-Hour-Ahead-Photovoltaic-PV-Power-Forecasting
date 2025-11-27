from __future__ import annotations

import pandas as pd


def localize_pv_index(
    s: pd.Series | pd.DatetimeIndex,
    local_tz: str,
) -> pd.DatetimeIndex:
    """Localize naive timestamps (PV) to `local_tz` with robust DST handling.

    - ambiguous=False will assume standard time for duplicated clock hours in fall.
    - nonexistent='shift_forward' moves spring-forward missing times to the next valid time.
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
    """Parse timezone-aware weather timestamps and return as aware UTC index."""
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts.isna().any():
        # If strings contain explicit offsets like '+10:00', pandas with utc=True handles it
        bad = ts[ts.isna()]
        raise ValueError(f"Found non-parsable weather timestamps: {bad[:5]}")
    return ts


def to_utc(idx: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        raise ValueError("Index must be timezone-aware to convert to UTC")
    return idx.tz_convert("UTC")

