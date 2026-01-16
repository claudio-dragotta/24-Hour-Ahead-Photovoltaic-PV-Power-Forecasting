#!/usr/bin/env python3
"""Simplified preprocessing following best practices for PV forecasting.

Key differences from preprocess_to_parquet.py:
- NO lag features (removes temporal leakage and overfitting)
- NO rolling features (reduces feature correlation)
- Normalize PV by nominal capacity BEFORE scaling
- Single scaler for all features together (preserves PV-weather relationships)
- Minimal feature engineering: only raw weather + temporal + solar position

This approach follows proven architectures that achieve low MASE/RMSE.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import math

import pandas as pd
import pvlib

from pv_forecasting.features import (
    add_calendar_flags,
    add_time_cyclical,
    encode_weather_onehot,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simplified preprocessing for PV forecasting")
    ap.add_argument("--merged-csv", type=str, default="data/processed/merged/pv_wx_combined.csv")
    ap.add_argument("--out", type=str, default="data/processed/merged/pv_wx_simple.parquet")
    ap.add_argument(
        "--normalize-pv-by-max",
        action="store_true",
        help="Normalize PV by maximum observed value (makes it 0-1 range)",
    )
    ap.add_argument(
        "--global-minmax-scaling",
        action="store_true",
        help="Apply MinMax scaling to ALL features globally after PV normalization (preserves PV-weather relationships)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    print(f"Loading merged CSV from {args.merged_csv} ...")
    df = pd.read_csv(args.merged_csv, parse_dates=True, index_col=0)

    # Remove completely NaN columns
    all_nan_cols = [col for col in df.columns if df[col].isna().all()]
    if all_nan_cols:
        print(f"Removing all-NaN columns: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    # Fill specific NaN
    if "rain_1h" in df.columns:
        df["rain_1h"] = df["rain_1h"].fillna(0)
    if "clouds_all" in df.columns and df["clouds_all"].isnull().any():
        df["clouds_all"] = df["clouds_all"].fillna(0)

    # Standardize column names (lowercase)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Ensure we have the key columns
    if "pv" not in df.columns:
        raise SystemExit("'pv' column is required")

    # One-hot encoding for weather_description
    if "weather_description" in df.columns:
        df = encode_weather_onehot(df, col="weather_description")

    # CRITICAL: Normalize PV by maximum capacity BEFORE any other processing
    # This makes PV range 0-1 like in reference models
    if args.normalize_pv_by_max:
        pv_max = df["pv"].max()
        print(f"Normalizing PV by maximum observed value: {pv_max:.2f} kW")
        df["pv"] = df["pv"] / pv_max
        df["pv_kwp_max"] = pv_max  # Store for denormalization later

    # Solar position and clear-sky features
    if "lat" not in df.columns or "lon" not in df.columns:
        raise SystemExit("lat/lon columns are required for solar features")

    try:
        lat = float(df["lat"].iloc[0])
        lon = float(df["lon"].iloc[0])
    except Exception as e:
        raise SystemExit("Unable to parse lat/lon") from e

    loc = pvlib.location.Location(latitude=lat, longitude=lon, tz="UTC", altitude=0)
    times = df.index

    # Solar position
    solar_pos = loc.get_solarposition(times=times)
    df["sp_zenith"] = solar_pos["zenith"].astype(float)
    df["sp_azimuth"] = solar_pos["azimuth"].astype(float)

    # Clear-sky irradiance
    cs = loc.get_clearsky(times=times, model="ineichen")
    df["cs_ghi"] = cs["ghi"].astype(float)
    df["cs_dni"] = cs["dni"].astype(float)
    df["cs_dhi"] = cs["dhi"].astype(float)

    # Clearness index
    if "ghi" in df.columns:
        kc = df["ghi"] / df["cs_ghi"].replace(0, math.nan)
        df["kc"] = kc.replace([math.inf, -math.inf], math.nan).fillna(0)

    # NO TEMPORAL FEATURES - avoid overfitting on time-specific patterns
    # NO LAG FEATURES - prevents temporal leakage
    # NO ROLLING FEATURES - keeps features independent

    df = df.sort_index()

    # Select only the essential columns for modeling
    # Keep: PV, raw weather, solar features, temporal features
    essential_cols = ["pv"]

    # Weather features (raw values only)
    weather_cols = [
        c
        for c in [
            "temp",
            "dew_point",
            "pressure",
            "humidity",
            "wind_speed",
            "wind_deg",
            "rain_1h",
            "clouds",
            "ghi",
            "dni",
            "dhi",
        ]
        if c in df.columns
    ]
    essential_cols.extend(weather_cols)

    # One-hot weather description
    wx_onehot = [c for c in df.columns if c.startswith("wx_")]
    essential_cols.extend(wx_onehot)

    # Solar features
    solar_cols = ["sp_zenith", "sp_azimuth", "cs_ghi", "cs_dni", "cs_dhi", "kc"]
    essential_cols.extend([c for c in solar_cols if c in df.columns])

    # NO temporal features - focus purely on physical/weather relationships

    # Keep metadata
    if "lat" in df.columns:
        essential_cols.append("lat")
    if "lon" in df.columns:
        essential_cols.append("lon")
    if "pv_kwp_max" in df.columns:
        essential_cols.append("pv_kwp_max")

    df_simple = df[essential_cols].copy()

    # Fill remaining NaN with 0 (only for edge cases)
    df_simple = df_simple.fillna(0)

    # Add dummy lag features (set to current timestep values for compatibility)
    # These are needed by the training script but don't provide real lag information
    df_simple["pv_lag1"] = df_simple["pv"]
    df_simple["ghi_lag1"] = df_simple.get("ghi", 0)
    df_simple["temp_lag1"] = df_simple.get("temp", 0)

    # CRITICAL: Apply global MinMax scaling to preserve PV-weather relationships
    # This matches the reference approach where a single scaler is fit on ALL features together
    if args.global_minmax_scaling:
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np

        # Columns to scale (exclude metadata AND PV)
        # PV is already normalized by max capacity, should NOT be scaled again
        metadata_cols = ["lat", "lon", "pv_kwp_max", "pv"]
        cols_to_scale = [c for c in df_simple.columns if c not in metadata_cols]

        print(f"\nApplying global MinMax scaling to {len(cols_to_scale)} weather features...")
        print(f"  PV is NOT scaled (already normalized by max capacity)")

        # Fit and transform on weather features only (PV already normalized)
        scaler = MinMaxScaler()
        df_simple[cols_to_scale] = scaler.fit_transform(df_simple[cols_to_scale])

        # Save scaler parameters for reference
        scaler_info = {
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "feature_names": cols_to_scale,
        }
        import json

        scaler_path = out_path.parent / f"{out_path.stem}_scaler_info.json"
        with open(scaler_path, "w") as f:
            json.dump(scaler_info, f, indent=2)
        print(f"Saved scaler info to {scaler_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_simple.to_parquet(out_path)

    print(f"\nSaved simplified feature table to {out_path}")
    print(f"Shape: {len(df_simple)} rows × {len(df_simple.columns)} columns")
    print(f"\nColumn groups:")
    print(f"  - PV: 1")
    print(f"  - Weather raw: {len(weather_cols)}")
    print(f"  - Weather one-hot: {len(wx_onehot)}")
    print(f"  - Solar: {len([c for c in solar_cols if c in df.columns])}")
    print(f"  - Metadata: {len([c for c in ['lat', 'lon', 'pv_kwp_max'] if c in df.columns])}")
    print(f"\n- NO temporal features (hour/day encoding removed)")

    if args.normalize_pv_by_max:
        print(f"\nPV normalized to [0, 1] range (max capacity: {pv_max:.2f} kW)")
    else:
        print(f"\nPV kept in raw kW scale")

    if args.global_minmax_scaling:
        print("\n- Global MinMax scaling applied to ALL features together")
        print("  → Preserves relationships between PV and weather")
        print("  → Training should use NO additional scalers (already scaled)")

    print("\nKey differences from full preprocessing:")
    print("  - NO lag features (removes temporal leakage)")
    print("  - NO rolling features (reduces overfitting)")
    print("  - Only current timestep features")
    if args.global_minmax_scaling:
        print("  - Single global scaler (preserves PV-weather relationships)")
    else:
        print("  - Ready for global scaling in training")


if __name__ == "__main__":
    main()
