#!/usr/bin/env python3
"""Preprocess test data WITH temporal features (matching WITH_TEMPORAL training)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from sklearn.preprocessing import MinMaxScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pv", type=str, default="data/test/pv_test.xlsx")
    parser.add_argument("--test-wx", type=str, default="data/test/wx_test.xlsx")
    parser.add_argument(
        "--train-scaler-info", type=str, default="data/processed/merged/pv_wx_simple_scaled_FIXED_scaler_info.json"
    )
    parser.add_argument("--out", type=str, default="data/test/processed/pv_wx_test_WITH_TEMPORAL.parquet")
    args = parser.parse_args()

    print("=" * 80)
    print("TEST PREPROCESSING WITH TEMPORAL FEATURES")
    print("=" * 80)

    # Load training scaler info
    print(f"\nLoading training scaler from {args.train_scaler_info}...")
    with open(args.train_scaler_info, "r") as f:
        scaler_info = json.load(f)

    # Reconstruct training scaler
    train_scaler = MinMaxScaler()
    train_scaler.data_min_ = np.array(scaler_info["data_min"])
    train_scaler.data_max_ = np.array(scaler_info["data_max"])
    train_scaler.data_range_ = train_scaler.data_max_ - train_scaler.data_min_

    # Handle zero-range features
    train_scaler.scale_ = np.where(train_scaler.data_range_ == 0, 1.0, 1.0 / train_scaler.data_range_)
    train_scaler.min_ = np.where(train_scaler.data_range_ == 0, 0.0, -train_scaler.data_min_ * train_scaler.scale_)

    train_scaler.n_features_in_ = len(scaler_info["feature_names"])
    train_scaler.feature_names_in_ = np.array(scaler_info["feature_names"])

    weather_feature_names = scaler_info["feature_names"]
    print(f"Loaded scaler for {len(weather_feature_names)} WEATHER features")
    print("  (PV is NOT in this scaler - correct!)")
    print("  (NO temporal features - matching training!)")

    # Load test PV data
    print(f"\nLoading test PV data from {args.test_pv}...")
    df_pv = pd.read_excel(args.test_pv, header=None, nrows=8761)

    # Extract max kWp
    test_pv_max_kw = float(df_pv.iloc[0, 1])
    print(f"Test PV max capacity: {test_pv_max_kw:.2f} kW")

    # PV data starts from row 1
    pv_values = df_pv.iloc[1:, 1].values.astype(float)
    print(f"Test PV records: {len(pv_values)}")
    print(f"Test PV range (raw): [{pv_values.min():.2f}, {pv_values.max():.2f}] kW")

    # CRITICAL: Normalize PV using TRAINING max capacity (NOT test capacity!)
    # Model was trained on PV normalized by training capacity
    # Test PV must use SAME normalization for consistency
    TRAINING_PV_MAX_KW = 68.92  # From training data
    pv_normalized = pv_values / TRAINING_PV_MAX_KW
    print(f"Test PV normalized: [{pv_normalized.min():.4f}, {pv_normalized.max():.4f}]")
    print(f"  Note: Normalized by TRAINING capacity ({TRAINING_PV_MAX_KW:.2f} kW) for consistency!")
    print(f"  (Test capacity is {test_pv_max_kw:.2f} kW, but we use training capacity)")
    if pv_normalized.max() > 1.0:
        print(
            f"  [WARNING]  Max normalized PV = {pv_normalized.max():.4f} > 1.0 (test system has higher capacity than training)"
        )

    # Load test weather data
    print(f"\nLoading test weather data from {args.test_wx}...")
    df_wx = pd.read_excel(args.test_wx)
    print(f"Weather data shape: {df_wx.shape}")

    # CRITICAL FIX: Test data has capitalized column names (Ghi, Dni, Dhi)
    # Rename them to lowercase to match training data
    rename_map = {}
    for col in df_wx.columns:
        if col.lower() != col:
            if col.lower() not in df_wx.columns:  # Only rename if lowercase version doesn't exist
                rename_map[col] = col.lower()

    if rename_map:
        print(f"\nRenaming capitalized columns to lowercase:")
        for old, new in rename_map.items():
            print(f"  {old} → {new}")
        df_wx = df_wx.rename(columns=rename_map)

    # Create combined test dataframe
    df_test = df_wx.copy()
    df_test["pv"] = pv_normalized

    # Add metadata
    df_test["lat"] = 32.889
    df_test["lon"] = 151.194
    # Store TRAINING capacity for denormalization (not test capacity!)
    df_test["pv_kwp_max"] = TRAINING_PV_MAX_KW  # Use training capacity!
    df_test["test_pv_kwp_max"] = test_pv_max_kw  # Store test capacity for reference

    # CALCULATE SOLAR FEATURES (critical for model performance!)
    print("\nCalculating solar features...")
    lat = 32.889
    lon = 151.194
    loc = pvlib.location.Location(latitude=lat, longitude=lon, tz="UTC", altitude=0)

    # Create DatetimeIndex from dt_iso
    if "dt_iso" in df_test.columns:
        times = pd.DatetimeIndex(pd.to_datetime(df_test["dt_iso"], utc=True))
    else:
        # If no dt_iso, create hourly index (fallback)
        times = pd.date_range(start="2012-07-01", periods=len(df_test), freq="h", tz="UTC")

    # Solar position
    solar_pos = loc.get_solarposition(times=times)
    df_test["sp_zenith"] = solar_pos["zenith"].values.astype(float)
    df_test["sp_azimuth"] = solar_pos["azimuth"].values.astype(float)

    # Clear-sky irradiance
    cs = loc.get_clearsky(times=times, model="ineichen")
    df_test["cs_ghi"] = cs["ghi"].values.astype(float)
    df_test["cs_dni"] = cs["dni"].values.astype(float)
    df_test["cs_dhi"] = cs["dhi"].values.astype(float)

    # Clearness index (kc) - ghi is still RAW at this point (not scaled)
    if "ghi" in df_test.columns:
        kc = df_test["ghi"] / df_test["cs_ghi"].replace(0, math.nan)
        df_test["kc"] = kc.replace([math.inf, -math.inf], math.nan).fillna(0).astype(float)
    else:
        df_test["kc"] = 0.0

    print(f"  Solar features calculated: sp_zenith, sp_azimuth, cs_ghi, cs_dni, cs_dhi, kc")

    # ADD TEMPORAL FEATURES (critical for WITH_TEMPORAL model!)
    print("\nCalculating temporal features...")

    # Cyclical encoding for hour of day
    df_test["hour_sin"] = np.sin(2 * np.pi * times.hour / 24.0)
    df_test["hour_cos"] = np.cos(2 * np.pi * times.hour / 24.0)

    # Cyclical encoding for day of year
    day_of_year = times.dayofyear
    df_test["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.0)
    df_test["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.0)

    # Weekend indicator
    df_test["is_weekend"] = times.dayofweek.isin([5, 6]).astype(float)

    # Holiday indicator (set to 0 for test data - no holidays in the period)
    df_test["is_holiday"] = 0.0

    print(f"  Temporal features calculated: hour_sin, hour_cos, doy_sin, doy_cos, is_weekend, is_holiday")

    # Ensure all required weather features exist
    print("\nChecking weather features...")
    missing_features = [f for f in weather_feature_names if f not in df_test.columns]
    if missing_features:
        print(f"WARNING: Missing {len(missing_features)} features in test data")
        print(f"  First 10: {missing_features[:10]}")
        print("  Setting missing features to 0")
        for feat in missing_features:
            df_test[feat] = 0

    # Extract weather values for scaling
    test_weather_values = df_test[weather_feature_names].values

    # Check for NaN
    if np.isnan(test_weather_values).any():
        print(f"\nWARNING: Found {np.isnan(test_weather_values).sum()} NaN values")
        print("Filling NaN with 0...")
        test_weather_values = np.nan_to_num(test_weather_values, 0)

    # Apply training scaler to WEATHER features only
    print(f"\nApplying training scaler to weather features...")
    test_weather_scaled = train_scaler.transform(test_weather_values)

    # Check statistics AFTER scaling (before clipping)
    print("\nScaled weather statistics (before clipping):")
    ood_count = 0
    for feat in ["rain_1h", "wind_speed", "temp", "humidity"]:
        if feat in weather_feature_names:
            idx = weather_feature_names.index(feat)
            vals = test_weather_scaled[:, idx]
            print(f"  {feat}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")
            if vals.max() > 1.0 or vals.min() < 0.0:
                print(f"    [WARNING]  OOD detected → will be clipped")
                ood_count += 1

    # CLIP weather features to [0, 1]
    print("\n" + "=" * 80)
    print("APPLYING CLIPPING TO WEATHER FEATURES ONLY")
    print("=" * 80)

    below_zero = (test_weather_scaled < 0).sum()
    above_one = (test_weather_scaled > 1).sum()

    print(f"Weather values < 0: {below_zero} ({below_zero/test_weather_scaled.size*100:.2f}%)")
    print(f"Weather values > 1: {above_one} ({above_one/test_weather_scaled.size*100:.2f}%)")

    test_weather_clipped = np.clip(test_weather_scaled, 0, 1)

    # Update dataframe with clipped weather values
    df_test[weather_feature_names] = test_weather_clipped

    # Check statistics AFTER clipping
    print("\nWeather statistics (AFTER clipping):")
    for feat in ["rain_1h", "wind_speed", "temp", "humidity"]:
        if feat in weather_feature_names:
            vals = df_test[feat].values
            print(f"  {feat}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

    # Add dummy lag features for compatibility
    df_test["pv_lag1"] = df_test["pv"]
    df_test["ghi_lag1"] = df_test.get("ghi", 0)
    df_test["temp_lag1"] = df_test.get("temp", 0)

    # Save
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_test.to_parquet(output_path)

    print(f"\n{'='*80}")
    print(f"Test data saved to {output_path}")
    print(f"Shape: {df_test.shape}")
    print(f"\nPV statistics:")
    print(f"  Range: [{df_test['pv'].min():.4f}, {df_test['pv'].max():.4f}]")
    print(f"  Mean: {df_test['pv'].mean():.4f}")
    print(f"\nKey features of this preprocessing:")
    print(f"  - WITH temporal features (hour_sin/cos, doy_sin/cos, is_weekend, is_holiday)")
    print(f"  - WITH solar features (sp_zenith, sp_azimuth, cs_ghi, cs_dni, cs_dhi, kc)")
    print(f"  - PV normalized by TRAINING max ({TRAINING_PV_MAX_KW:.2f} kW) - NOT test capacity!")
    print(f"  - Test system capacity: {test_pv_max_kw:.2f} kW (stored but not used for normalization)")
    print(f"  - PV NOT scaled with weather features")
    print(f"  - Weather features scaled with training scaler")
    print(f"  - Weather features clipped to [0, 1] (handles OOD)")
    print(f"  - PV can be > 1.0 if test system has higher capacity than training")
    print(f"\nExpected impact:")
    print(f"  - Consistent normalization between train and test")
    print(f"  - Temporal patterns can help with diurnal/seasonal variations")
    print(f"  - Proper handling of extreme weather (clipped)")
    print(f"  - Model will recognize high PV output correctly")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
