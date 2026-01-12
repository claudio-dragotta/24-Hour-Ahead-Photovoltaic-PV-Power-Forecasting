"""Data preprocessing script using unified CSV file.

This script loads the combined CSV file (pv_wx_combined.csv),
applies all feature engineering, and saves to outputs/processed.parquet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pv_forecasting.features import (
    add_calendar_flags,
    add_clearsky,
    add_kc,
    add_lags,
    add_rolling_vars_h,
    add_rollings_h,
    add_solar_position,
    add_time_cyclical,
    encode_weather_onehot,
    standardize_feature_columns,
)
from pv_forecasting.timeutils import to_utc


def load_unified_csv(path: Path) -> pd.DataFrame:
    """Load unified PV + weather data from combined CSV file."""
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])

    # Set UTC timestamp as index
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Rename pv_power to pv
    df = df.rename(columns={"pv_power": "pv"})

    # Drop timestamp_local (we use UTC)
    if "timestamp_local" in df.columns:
        df = df.drop(columns=["timestamp_local"])

    # Keep weather_description as string, convert others to numeric
    for c in df.columns:
        if c == "weather_description":
            continue
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except:
            pass

    return df


def main():
    print("\n" + "=" * 70)
    print("PV FORECASTING - DATA PREPROCESSING (Unified CSV)")
    print("=" * 70)

    # Configuration
    input_path = Path("data/raw/pv_wx_combined.csv")
    output_dir = Path("outputs")

    # Default feature engineering parameters
    lag_hours = (1, 24, 48, 72, 96, 168)
    rolling_hours = (3, 6, 12, 24)

    print(f"\nInput file: {input_path}")

    # Load data
    print("\n1. Loading unified data...")
    df = load_unified_csv(input_path)
    print(f"   Samples: {len(df)}")
    print(f"   Range: {df.index.min()} to {df.index.max()}")
    print(f"   Weather descriptions: {df['weather_description'].nunique()} unique")

    # Standardize columns
    print("\n2. Standardizing feature columns...")
    df = standardize_feature_columns(df)

    # Encode weather description (one-hot)
    print("\n3. Encoding weather description (one-hot)...")
    print(f"   Categories before encoding: {df['weather_description'].nunique()}")
    df = encode_weather_onehot(df, col="weather_description")
    wx_cols = [c for c in df.columns if c.startswith("wx_")]
    print(f"   Created {len(wx_cols)} binary columns: {wx_cols}")

    # Add time features
    print("\n4. Adding time features...")
    df = add_time_cyclical(df)
    df = add_calendar_flags(df)

    # Add solar position
    print("\n5. Adding solar position features...")
    df = add_solar_position(df)

    # Add clear-sky irradiance
    print("\n6. Adding clear-sky irradiance...")
    df = add_clearsky(df)
    if "ghi" in df.columns:
        df = add_kc(df, ghi_col="ghi")
        if "kc" in df.columns:
            df["kc"] = df["kc"].fillna(0.0)

    # Add lag features
    print("\n7. Adding lag features...")
    lag_cols = [c for c in ["pv", "ghi", "dni", "dhi"] if c in df.columns]
    if lag_hours and lag_cols:
        df = add_lags(df, lag_cols, list(lag_hours))
        print(f"   Lag columns: {lag_cols}")
        print(f"   Lag hours: {lag_hours}")

    # Add rolling features
    print("\n8. Adding rolling features...")
    roll_cols = [c for c in ["pv", "ghi", "dni"] if c in df.columns]
    if rolling_hours and roll_cols:
        df = add_rollings_h(df, roll_cols, list(rolling_hours))
        df = add_rolling_vars_h(df, roll_cols, list(rolling_hours))
        print(f"   Rolling columns: {roll_cols}")
        print(f"   Rolling windows: {rolling_hours}")

    # Sort and clean
    print("\n9. Cleaning data...")
    df = df.sort_index()

    # Fill NaN in rain_1h with 0 (no rain)
    if "rain_1h" in df.columns:
        nan_rain = df["rain_1h"].isna().sum()
        df["rain_1h"] = df["rain_1h"].fillna(0.0)
        print(f"    Filled {nan_rain} NaN in rain_1h with 0")

    # Drop NaN in critical columns
    critical_cols = [c for c in df.columns if any(x in c for x in ["pv", "ghi", "dni", "dhi", "lag", "roll", "var"])]
    initial_len = len(df)
    df = df.dropna(subset=critical_cols)
    print(f"    Dropped {initial_len - len(df)} rows with NaN")

    # Add time index and series ID
    df = df.copy()
    df["time_idx"] = np.arange(len(df))
    df["series_id"] = "pv_site_1"

    # Save
    print("\n10. Saving processed data...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "processed.parquet"
    df.to_parquet(output_path)
    print(f"    Saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nDataset summary:")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Output: {output_path}")

    print(f"\nFeature columns ({len(df.columns)}):")
    for i, col in enumerate(sorted(df.columns)):
        print(f"  {i+1:2d}. {col}")

    # Weather one-hot distribution
    wx_cols = [c for c in df.columns if c.startswith("wx_")]
    if wx_cols:
        print(f"\nWeather one-hot distribution:")
        for col in sorted(wx_cols):
            count = int(df[col].sum())
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
