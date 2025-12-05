"""Feature engineering functions for PV power forecasting.

This module provides functions to create time-based, lag-based, and physics-informed
features for solar power forecasting. All functions operate on pandas DataFrames with
timezone-aware DatetimeIndex.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import pvlib

    _HAS_PVLIB = True
except Exception:
    _HAS_PVLIB = False


def add_time_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features (hour and day-of-year) to dataframe.

    Converts hour-of-day and day-of-year into sine/cosine pairs to preserve
    cyclical nature (e.g., 23:00 and 00:00 are close, Dec 31 and Jan 1 are close).

    Args:
        df: Input dataframe with timezone-aware DatetimeIndex.

    Returns:
        DataFrame with four additional columns:
            - hour_sin: Sine of hour angle (0-23 mapped to 0-2π)
            - hour_cos: Cosine of hour angle
            - doy_sin: Sine of day-of-year angle (1-365 mapped to 0-2π)
            - doy_cos: Cosine of day-of-year angle

    Example:
        >>> df = pd.DataFrame(index=pd.date_range('2020-01-01', periods=24, freq='H', tz='UTC'))
        >>> df_with_time = add_time_cyclical(df)
        >>> 'hour_sin' in df_with_time.columns
        True

    Note:
        - Index is automatically converted to UTC before feature extraction
        - Sine and cosine values are always in range [-1, 1]
        - Leap years use 365 for day-of-year normalization (slight approximation)
    """
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
    """Add lagged features for specified columns.

    Creates new columns with time-shifted values, useful for capturing
    temporal dependencies in time series forecasting.

    Args:
        df: Input dataframe with time series data.
        cols: List of column names to create lag features for.
        lags: List of lag periods (positive integers).
            Example: [1, 24, 168] creates 1-hour, 1-day, and 1-week lags.

    Returns:
        DataFrame with original columns plus new lag columns named as
        "{column}_lag{period}". Example: "pv_lag24" for 24-hour lag of 'pv'.

    Example:
        >>> df = pd.DataFrame({'pv': [0.5, 0.6, 0.7, 0.8]})
        >>> df_with_lags = add_lags(df, cols=['pv'], lags=[1, 2])
        >>> df_with_lags['pv_lag1'].iloc[2]  # Value from index 1
        0.6

    Note:
        - First N rows will have NaN values where N equals the maximum lag
        - NaN rows should typically be dropped before model training
        - Does not validate if columns exist in dataframe
    """
    out = df.copy()
    for c in cols:
        for l in lags:
            out[f"{c}_lag{l}"] = out[c].shift(l)
    return out


def add_rollings(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    """Add rolling mean features with generic time units (minutes assumed).

    Computes moving averages over specified window sizes. Useful for
    capturing short-term trends and smoothing noisy signals.

    Args:
        df: Input dataframe with time series data.
        cols: List of column names to create rolling features for.
        windows: List of window sizes (in generic time units, typically minutes).
            Example: [60, 180] for 1-hour and 3-hour windows.

    Returns:
        DataFrame with original columns plus new rolling mean columns named as
        "{column}_roll{window}m". Example: "pv_roll60m" for 60-unit rolling mean.

    Example:
        >>> df = pd.DataFrame({'pv': [0.1, 0.2, 0.3, 0.4, 0.5]})
        >>> df_with_rolling = add_rollings(df, cols=['pv'], windows=[3])
        >>> # First 2 values use partial windows (min_periods=2)

    Note:
        - Uses min_periods=max(1, window//2) to handle edge cases
        - Partial windows at the start use available data points
        - This function is for minute-level data; use add_rollings_h for hourly
    """
    out = df.copy()
    for c in cols:
        for w in windows:
            out[f"{c}_roll{w}m"] = out[c].rolling(window=w, min_periods=max(1, w // 2)).mean()
    return out


def add_rollings_h(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    """Add rolling mean features for hourly data.

    Computes moving averages over specified window sizes in hours.
    Similar to add_rollings but specifically designed for hourly time series.

    Args:
        df: Input dataframe with hourly time series data.
        cols: List of column names to create rolling features for.
        windows: List of window sizes in hours.
            Example: [3, 6, 24] for 3-hour, 6-hour, and 1-day windows.

    Returns:
        DataFrame with original columns plus new rolling mean columns named as
        "{column}_roll{window}h". Example: "pv_roll3h" for 3-hour rolling mean.

    Example:
        >>> # Hourly PV data
        >>> df = pd.DataFrame({'pv': np.random.rand(100)})
        >>> df_with_rolling = add_rollings_h(df, cols=['pv'], windows=[3, 6])
        >>> 'pv_roll3h' in df_with_rolling.columns
        True

    Note:
        - Assumes data is sampled at hourly intervals
        - Uses min_periods=max(1, window//2) for partial windows
        - Recommended windows for solar data: [3, 6, 24, 168] (3h, 6h, 1d, 1w)
    """
    out = df.copy()
    for c in cols:
        for w in windows:
            out[f"{c}_roll{w}h"] = out[c].rolling(window=w, min_periods=max(1, w // 2)).mean()
    return out


def standardize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize weather and irradiance column names to lowercase.

    Normalizes column names from various weather data sources to a consistent
    naming convention. Handles common synonyms and variations.

    Args:
        df: Input dataframe with weather/irradiance columns.

    Returns:
        DataFrame with standardized column names:
            - GHI, Ghi, ghi(W/m2) → ghi
            - DNI, Dni → dni
            - DHI, Dhi → dhi
            - Temperature, TEMP → temp
            - cloud_cover, clouds_all → clouds
            - Other columns remain unchanged

    Example:
        >>> df = pd.DataFrame({'GHI': [100, 200], 'Temperature': [20, 25]})
        >>> result = standardize_feature_columns(df)
        >>> list(result.columns)
        ['ghi', 'temp']

    Note:
        - Original dataframe is not modified (returns a copy)
        - Case-insensitive matching for known weather variables
        - Unknown columns are preserved with original names
        - Useful for integrating data from multiple weather APIs
    """
    m = {c.lower(): c for c in df.columns}
    out = df.copy()
    # Map common irradiance/weather synonyms to lowercase canonical names
    mapping: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {
            "ghi",
            "dhi",
            "dni",
            "temp",
            "temperature",
            "humidity",
            "wind_speed",
            "wind",
            "wind_deg",
            "clouds_all",
            "clouds",
            "rain_1h",
            "lat",
            "lon",
        }:
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
    """Add solar position angles (zenith and azimuth) using pvlib.

    Computes the sun's position in the sky for each timestamp. Essential
    for physics-informed solar forecasting models.

    Args:
        df: Input dataframe with timezone-aware DatetimeIndex (UTC).
        lat: Site latitude in decimal degrees. If None, attempts to read
            from 'lat' column in dataframe.
        lon: Site longitude in decimal degrees. If None, attempts to read
            from 'lon' column in dataframe.
        altitude: Site altitude in meters above sea level. Default: 0.0.

    Returns:
        DataFrame with two additional columns:
            - sp_zenith: Solar zenith angle in degrees [0-180]
                (0° = sun directly overhead, 90° = horizon)
            - sp_azimuth: Solar azimuth angle in degrees [0-360]
                (0° = North, 90° = East, 180° = South, 270° = West)

    Raises:
        ImportError: If pvlib is not installed.
        ValueError: If index is not timezone-aware or if lat/lon cannot be determined.

    Example:
        >>> df = pd.DataFrame(index=pd.date_range('2020-06-21 12:00', periods=24, freq='H', tz='UTC'))
        >>> df_with_sun = add_solar_position(df, lat=-33.86, lon=151.21)
        >>> df_with_sun['sp_zenith'].iloc[0]  # Low zenith at noon
        30.5

    Note:
        - Requires pvlib library: `pip install pvlib`
        - Index must be timezone-aware (automatically converted to UTC)
        - For Sydney, Australia: lat=-33.86, lon=151.21
        - Solar elevation = 90° - zenith
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
    """Add theoretical clear-sky irradiance using pvlib.

    Computes what solar irradiance would be under perfectly clear sky conditions.
    Useful as a baseline and for calculating clearness index (actual/clear-sky ratio).

    Args:
        df: Input dataframe with timezone-aware DatetimeIndex (UTC).
        lat: Site latitude in decimal degrees. If None, attempts to read
            from 'lat' column in dataframe.
        lon: Site longitude in decimal degrees. If None, attempts to read
            from 'lon' column in dataframe.
        altitude: Site altitude in meters above sea level. Default: 0.0.
        model: Clear-sky model to use. Default: "ineichen" (recommended).
            Other options: "simplified_solis", "haurwitz".

    Returns:
        DataFrame with three additional columns (all in W/m²):
            - cs_ghi: Clear-sky Global Horizontal Irradiance
            - cs_dni: Clear-sky Direct Normal Irradiance
            - cs_dhi: Clear-sky Diffuse Horizontal Irradiance

    Raises:
        ImportError: If pvlib is not installed.
        ValueError: If index is not timezone-aware or if lat/lon cannot be determined.

    Example:
        >>> df = pd.DataFrame(index=pd.date_range('2020-06-21 12:00', periods=24, freq='H', tz='UTC'))
        >>> df_with_cs = add_clearsky(df, lat=-33.86, lon=151.21)
        >>> df_with_cs['cs_ghi'].max()  # Peak clear-sky irradiance
        1050.3

    Note:
        - Requires pvlib library: `pip install pvlib`
        - Ineichen model is most accurate but requires additional atmospheric data
        - Clear-sky values are theoretical maximums (no clouds/aerosols)
        - Nighttime values are automatically set to 0
        - Use with add_kc() to compute clearness index
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
    """Add clearness index (Kc) = measured GHI / clear-sky GHI.

    The clearness index indicates how much of the theoretical clear-sky irradiance
    is actually reaching the ground. Values close to 1 indicate clear sky,
    lower values indicate clouds or atmospheric attenuation.

    Args:
        df: Input dataframe containing measured GHI and clear-sky GHI columns.
        ghi_col: Name of measured GHI column. Default: "ghi".

    Returns:
        DataFrame with one additional column:
            - kc: Clearness index, clipped to range [0, 2]
                - kc ≈ 1.0: Clear sky
                - kc < 0.8: Cloudy
                - kc > 1.0: Enhanced irradiance (reflections from clouds)
                - kc = NaN: Nighttime or missing data

    Example:
        >>> # Assumes df already has 'ghi' and 'cs_ghi' columns
        >>> df_with_kc = add_kc(df, ghi_col='ghi')
        >>> df_with_kc['kc'].mean()  # Typical value: 0.5-0.7
        0.65

    Note:
        - Requires 'cs_ghi' column (add with add_clearsky first)
        - Division by zero (nighttime) is handled: kc set to NaN
        - Values clipped to [0, 2] to handle measurement errors
        - Kc > 1 can occur due to cloud edge enhancement effects
        - Important feature for cloud detection and forecasting
    """
    out = df.copy()
    if ghi_col in out.columns and "cs_ghi" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            kc = np.where(out["cs_ghi"].values > 0, out[ghi_col].values / out["cs_ghi"].values, np.nan)
        out["kc"] = np.clip(kc, 0.0, 2.0)
    return out


def encode_weather_description(df: pd.DataFrame, col: str = "weather_description") -> pd.DataFrame:
    """Encode weather description text to numerical ordinal values based on PV production impact.

    Converts categorical weather descriptions to numerical values ranked by expected
    impact on PV power generation. Higher values = better conditions for PV production.

    Encoding scheme (based on typical PV production):
        10.0 = clear sky / sunny → Maximum production
        8.0  = few clouds / partly cloudy → High production (80-90%)
        6.0  = scattered clouds / mostly cloudy → Medium production (60-70%)
        4.0  = broken clouds / overcast → Low production (40-50%)
        2.0  = rain / drizzle / light rain → Very low production (20-30%)
        1.0  = heavy rain / thunderstorm / snow → Minimal production (10-20%)
        0.0  = fog / mist / haze → Very low production (0-10%)
        5.0  = unknown / other → Default middle value

    Args:
        df: Input dataframe containing weather description column.
        col: Name of weather description column. Default: "weather_description".

    Returns:
        DataFrame with weather_description replaced by numerical encoding.
        Missing values are filled with 5.0 (neutral/unknown).

    Example:
        >>> df = pd.DataFrame({'weather_description': ['clear sky', 'rain', 'cloudy']})
        >>> df_encoded = encode_weather_description(df)
        >>> df_encoded['weather_description'].tolist()
        [10.0, 2.0, 4.0]

    Note:
        - Case-insensitive matching (converts to lowercase)
        - Uses substring matching (e.g., "light rain shower" → matches "rain")
        - Priority order: checks most specific conditions first
        - Missing/unknown values filled with 5.0 (middle value)
        - This is an ordinal encoding reflecting physical PV production impact
    """
    out = df.copy()

    if col not in out.columns:
        return out

    # Convert to lowercase string for matching
    weather = out[col].astype(str).str.lower()

    # Initialize with default value (unknown)
    encoded = pd.Series(5.0, index=out.index)

    # Priority encoding: most specific first
    # Clear sky conditions → Maximum PV production
    clear_mask = weather.str.contains("clear|sunny", case=False, na=False)
    encoded[clear_mask] = 10.0

    # Few clouds → High PV production
    few_clouds_mask = weather.str.contains("few cloud|partly", case=False, na=False)
    encoded[few_clouds_mask] = 8.0

    # Scattered clouds → Medium-high PV production
    scattered_mask = weather.str.contains("scatter|mostly cloud", case=False, na=False)
    encoded[scattered_mask] = 6.0

    # Overcast / broken clouds → Medium-low PV production
    overcast_mask = weather.str.contains("overcast|broken|cloud", case=False, na=False) & ~(
        clear_mask | few_clouds_mask | scattered_mask
    )
    encoded[overcast_mask] = 4.0

    # Heavy rain / storm → Minimal PV production
    heavy_rain_mask = weather.str.contains("heavy|thunderstorm|storm|snow", case=False, na=False)
    encoded[heavy_rain_mask] = 1.0

    # Light rain / drizzle → Very low PV production
    rain_mask = weather.str.contains("rain|drizzle", case=False, na=False) & ~heavy_rain_mask
    encoded[rain_mask] = 2.0

    # Fog / mist → Very low PV production (blocks sun)
    fog_mask = weather.str.contains("fog|mist|haze", case=False, na=False)
    encoded[fog_mask] = 0.0

    # Handle NaN / missing values
    nan_mask = out[col].isna()
    encoded[nan_mask] = 5.0  # Neutral/unknown

    out[col] = encoded
    return out
