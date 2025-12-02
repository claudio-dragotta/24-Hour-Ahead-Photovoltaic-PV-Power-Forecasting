from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

try:
    import pvlib
    _HAS_PVLIB = True
except Exception:
    _HAS_PVLIB = False


def add_time_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index.tz_convert("UTC")
    hour = idx.hour.values
    day_of_year = idx.dayofyear.values
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    return df


def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for l in lags:
            out[f"{c}_lag{l}"] = out[c].shift(l)
    return out


def add_rollings(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for w in windows:
            out[f"{c}_roll{w}m"] = out[c].rolling(window=w, min_periods=max(1, w // 2)).mean()
    return out


def add_rollings_h(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for w in windows:
            out[f"{c}_roll{w}h"] = out[c].rolling(window=w, min_periods=max(1, w // 2)).mean()
    return out


def standardize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with unified, lowercase weather/irradiance column names.

    Keeps original columns intact in input; returns a new frame with normalized names.
    """
    m = {c.lower(): c for c in df.columns}
    out = df.copy()
    # Map common irradiance/weather synonyms to lowercase canonical names
    mapping: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"ghi", "dhi", "dni", "temp", "temperature", "humidity", "wind_speed", "wind", "wind_deg", "clouds_all", "clouds", "rain_1h", "lat", "lon"}:
            mapping[c] = cl if cl not in {"temperature"} else "temp"
        elif cl == "ghi" or cl == "ghi(w/m2)":
            mapping[c] = "ghi"
        elif cl == "dni":
            mapping[c] = "dni"
        elif cl == "dhi":
            mapping[c] = "dhi"
        elif cl == "cloud_cover":
            mapping[c] = "clouds"
        else:
            # leave others as-is
            pass
    out = out.rename(columns=mapping)
    # Also normalize 'clouds_all' -> 'clouds'
    if "clouds_all" in out.columns:
        out = out.rename(columns={"clouds_all": "clouds"})
    return out


def add_solar_position(
    df: pd.DataFrame,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    altitude: float = 0.0,
) -> pd.DataFrame:
    """Add solar zenith/azimuth using pvlib. Index must be tz-aware UTC.

    If lat/lon not provided, tries to infer from constant columns 'lat'/'lon'.
    """
    if not _HAS_PVLIB:
        raise ImportError("pvlib not installed. Please add it to requirements and install.")
    if df.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware (UTC).")
    out = df.copy()
    lat0 = lat
    lon0 = lon
    if lat0 is None and "lat" in out.columns:
        try:
            lat0 = float(pd.to_numeric(out["lat"], errors="coerce").dropna().iloc[0])
        except Exception:
            pass
    if lon0 is None and "lon" in out.columns:
        try:
            lon0 = float(pd.to_numeric(out["lon"], errors="coerce").dropna().iloc[0])
        except Exception:
            pass
    if lat0 is None or lon0 is None:
        raise ValueError("Latitude/Longitude required: pass lat/lon or include 'lat'/'lon' columns.")
    loc = pvlib.location.Location(latitude=lat0, longitude=lon0, altitude=altitude, tz="UTC")
    sp = loc.get_solarposition(out.index)
    out["sp_zenith"] = sp["zenith"].values.astype(float)
    out["sp_azimuth"] = sp["azimuth"].values.astype(float)
    return out


def add_clearsky(
    df: pd.DataFrame,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    altitude: float = 0.0,
    model: str = "ineichen",
) -> pd.DataFrame:
    """Add clear-sky irradiance columns (ghi/dni/dhi) using pvlib.

    Index must be tz-aware UTC. Requires lat/lon.
    """
    if not _HAS_PVLIB:
        raise ImportError("pvlib not installed. Please add it to requirements and install.")
    if df.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware (UTC).")
    out = df.copy()
    lat0 = lat
    lon0 = lon
    if lat0 is None and "lat" in out.columns:
        lat0 = float(pd.to_numeric(out["lat"], errors="coerce").dropna().iloc[0])
    if lon0 is None and "lon" in out.columns:
        lon0 = float(pd.to_numeric(out["lon"], errors="coerce").dropna().iloc[0])
    if lat0 is None or lon0 is None:
        raise ValueError("Latitude/Longitude required: pass lat/lon or include 'lat'/'lon' columns.")
    loc = pvlib.location.Location(latitude=lat0, longitude=lon0, altitude=altitude, tz="UTC")
    cs = loc.get_clearsky(out.index, model=model)
    # Columns: 'ghi','dni','dhi'
    out["cs_ghi"] = cs["ghi"].values.astype(float)
    out["cs_dni"] = cs["dni"].values.astype(float)
    out["cs_dhi"] = cs["dhi"].values.astype(float)
    return out


def add_kc(df: pd.DataFrame, ghi_col: str = "ghi") -> pd.DataFrame:
    """Add clear-sky index Kc = GHI / cs_ghi when both available."""
    out = df.copy()
    if ghi_col in out.columns and "cs_ghi" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            kc = np.where(out["cs_ghi"].values > 0, out[ghi_col].values / out["cs_ghi"].values, np.nan)
        out["kc"] = np.clip(kc, 0.0, 2.0)
    return out
