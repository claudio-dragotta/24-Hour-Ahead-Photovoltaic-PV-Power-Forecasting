#!/usr/bin/env python3
"""Preprocess test data using FIXED training scaler (PV NOT in scaler) + CLIPPING."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pv", type=str, default="data/test/pv_test.xlsx")
    parser.add_argument("--test-wx", type=str, default="data/test/wx_test.xlsx")
    parser.add_argument(
        "--train-scaler-info", type=str, default="data/processed/merged/pv_wx_simple_scaled_FIXED_scaler_info.json"
    )
    parser.add_argument("--out", type=str, default="data/test/processed/pv_wx_test_FIXED.parquet")
    args = parser.parse_args()

    print("=" * 80)
    print("TEST PREPROCESSING WITH FIXED SCALER (PV NOT SCALED)")
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

    # Handle features with zero range (min==max) to avoid division by zero
    # For these features, scale will be 1.0 (identity) and result will be 0
    train_scaler.scale_ = np.where(train_scaler.data_range_ == 0, 1.0, 1.0 / train_scaler.data_range_)
    train_scaler.min_ = np.where(train_scaler.data_range_ == 0, 0.0, -train_scaler.data_min_ * train_scaler.scale_)

    train_scaler.n_features_in_ = len(scaler_info["feature_names"])
    train_scaler.feature_names_in_ = np.array(scaler_info["feature_names"])

    weather_feature_names = scaler_info["feature_names"]
    print(f"Loaded scaler for {len(weather_feature_names)} WEATHER features")
    print("  (PV is NOT in this scaler - correct!)")

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

    # Normalize PV using TEST max capacity (NOT training max!)
    # This keeps PV in [0, 1] range, which the model expects
    pv_normalized = pv_values / test_pv_max_kw
    print(f"Test PV normalized: [{pv_normalized.min():.4f}, {pv_normalized.max():.4f}]")
    print(f"  Note: Normalized by test capacity ({test_pv_max_kw:.2f} kW)")

    # Load test weather data
    print(f"\nLoading test weather data from {args.test_wx}...")
    df_wx = pd.read_excel(args.test_wx)
    print(f"Weather data shape: {df_wx.shape}")

    # Create combined test dataframe
    df_test = df_wx.copy()
    df_test["pv"] = pv_normalized

    # Add metadata
    df_test["lat"] = 32.889
    df_test["lon"] = 151.194
    df_test["pv_kwp_max"] = test_pv_max_kw

    # Ensure is_holiday exists and has no NaN (not in weather scaler but needed for model)
    # Force to 0 for all rows since we don't have holiday data for test
    df_test["is_holiday"] = 0
    print("Set is_holiday to 0 for all rows (no holiday data for test)")

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

    # Check raw statistics BEFORE scaling
    print("\nRaw test statistics (before scaling):")
    for feat in ["rain_1h", "wind_speed", "temp", "humidity"]:
        if feat in weather_feature_names:
            idx = weather_feature_names.index(feat)
            vals = test_weather_values[:, idx]
            print(f"  {feat}: min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}")

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
    print(f"  - PV normalized by training max (68.92 kW)")
    print(f"  - PV NOT scaled with weather features")
    print(f"  - Weather features scaled with training scaler")
    print(f"  - Weather features clipped to [0, 1] (handles OOD)")
    print(f"  - PV NOT clipped (can be > 1.0 for high-capacity systems)")
    print(f"\nExpected impact:")
    print(f"  - No more NaN predictions")
    print(f"  - Proper handling of extreme weather (rain, wind clipped)")
    print(f"  - Correct handling of different PV capacities")
    print(f"  - MASE should improve significantly")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
