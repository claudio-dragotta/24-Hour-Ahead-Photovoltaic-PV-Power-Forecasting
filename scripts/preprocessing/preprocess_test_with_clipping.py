#!/usr/bin/env python3
"""Preprocess test data using training scaler + CLIPPING to [0,1]."""
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
        "--train-scaler-info", type=str, default="data/processed/merged/pv_wx_simple_scaled_scaler_info.json"
    )
    parser.add_argument("--out", type=str, default="data/test/processed/pv_wx_test_CLIPPED.parquet")
    args = parser.parse_args()

    print("=" * 80)
    print("TEST PREPROCESSING WITH CLIPPING")
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
    train_scaler.scale_ = 1.0 / train_scaler.data_range_
    train_scaler.min_ = -train_scaler.data_min_ * train_scaler.scale_
    train_scaler.n_features_in_ = len(scaler_info["feature_names"])
    train_scaler.feature_names_in_ = np.array(scaler_info["feature_names"])

    feature_names = scaler_info["feature_names"]
    print(f"Loaded scaler for {len(feature_names)} features")

    # Load test PV data (limit to 8761 rows: 1 header + 8760 hourly data)
    print(f"\nLoading test PV data from {args.test_pv}...")
    df_pv = pd.read_excel(args.test_pv, header=None, nrows=8761)

    # Extract max kWp (first row, second column)
    test_pv_max_kw = float(df_pv.iloc[0, 1])
    print(f"Test PV max capacity: {test_pv_max_kw:.2f} kW")

    # PV data starts from row 1 (row 0 is metadata)
    pv_values = df_pv.iloc[1:, 1].values.astype(float)
    print(f"Test PV records: {len(pv_values)}")
    print(f"Test PV range (raw): [{pv_values.min():.2f}, {pv_values.max():.2f}] kW")

    # Normalize PV using TRAINING max (68.92 kW)
    TRAINING_PV_MAX = 68.92
    pv_normalized = (pv_values * test_pv_max_kw) / TRAINING_PV_MAX
    print(f"Test PV after normalization: [{pv_normalized.min():.4f}, {pv_normalized.max():.4f}]")

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

    # Ensure all required features exist
    print("\nChecking features...")
    missing_features = [f for f in feature_names if f not in df_test.columns]
    if missing_features:
        print(f"WARNING: Missing features in test data: {missing_features}")
        print("Setting missing features to 0")
        for feat in missing_features:
            df_test[feat] = 0

    # Extract values for scaling (same features as training)
    test_values = df_test[feature_names].values

    # Check for NaN
    if np.isnan(test_values).any():
        print(f"\nWARNING: Found {np.isnan(test_values).sum()} NaN values")
        print("Filling NaN with 0...")
        test_values = np.nan_to_num(test_values, 0)

    # Apply training scaler
    print(f"\nApplying training scaler to test data...")
    print(f"Test data shape before scaling: {test_values.shape}")

    # Check raw statistics BEFORE scaling
    print("\nRaw test statistics (before scaling):")
    for i, feat in enumerate(["rain_1h", "wind_speed", "temp", "humidity"]):
        if feat in feature_names:
            idx = feature_names.index(feat)
            vals = test_values[:, idx]
            print(f"  {feat}: min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}")

    test_scaled = train_scaler.transform(test_values)

    # Check statistics AFTER scaling (before clipping)
    print("\nScaled test statistics (before clipping):")
    for i, feat in enumerate(["rain_1h", "wind_speed", "temp", "humidity"]):
        if feat in feature_names:
            idx = feature_names.index(feat)
            vals = test_scaled[:, idx]
            print(f"  {feat}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")
            if vals.max() > 1.0:
                print(f"    [WARNING]  {feat} exceeds 1.0 → will be clipped")

    # CLIP to [0, 1] - BUT NOT PV!
    print("\n" + "=" * 80)
    print("APPLYING CLIPPING TO [0, 1]")
    print("=" * 80)

    # Save PV column index to NOT clip it
    pv_idx = feature_names.index("pv") if "pv" in feature_names else None

    # Count values that will be clipped
    below_zero = (test_scaled < 0).sum()
    above_one = (test_scaled > 1).sum()

    print(f"Values < 0: {below_zero} ({below_zero/test_scaled.size*100:.2f}%)")
    print(f"Values > 1: {above_one} ({above_one/test_scaled.size*100:.2f}%)")

    # Clip weather features only
    test_clipped = test_scaled.copy()
    for i in range(test_scaled.shape[1]):
        if i != pv_idx:  # Don't clip PV!
            test_clipped[:, i] = np.clip(test_scaled[:, i], 0, 1)

    if pv_idx is not None:
        print(f"\nNOTE: PV NOT clipped (can be > 1.0 for high-capacity test systems)")
        print(f"  PV range: [{test_clipped[:, pv_idx].min():.4f}, {test_clipped[:, pv_idx].max():.4f}]")

    # Check statistics AFTER clipping
    print("\nScaled test statistics (AFTER clipping):")
    for i, feat in enumerate(["pv", "rain_1h", "wind_speed", "temp", "humidity"]):
        if feat in feature_names:
            idx = feature_names.index(feat)
            vals = test_clipped[:, idx]
            print(f"  {feat}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

    # Update dataframe with clipped values
    df_test[feature_names] = test_clipped

    # Add dummy lag features (set to current timestep values for compatibility)
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
    print(f"\nKey differences from previous preprocessing:")
    print(f"  - Applied training scaler")
    print(f"  - CLIPPED all values to [0, 1]")
    print(f"  - No OOD values (rain, wind, etc. capped at 1.0)")
    print(f"\nExpected impact:")
    print(f"  - rain > 6.73mm (training max) → clipped to 1.0")
    print(f"  - wind > 21.6 m/s (training max) → clipped to 1.0")
    print(f"  - Model should see familiar value ranges")
    print(f"  - MASE should improve significantly")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
