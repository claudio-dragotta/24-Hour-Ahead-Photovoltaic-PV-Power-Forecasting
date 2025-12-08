"""Data preprocessing script for PV forecasting.

This script loads raw PV and weather data, applies all feature engineering,
and saves the processed dataset to outputs/processed.parquet.

Features created:
- Time cyclical: hour_sin/cos, doy_sin/cos
- Solar position: zenith, azimuth angles
- Clear-sky irradiance: GHI, DNI, DHI theoretical values
- Clearness index (kc): measured_GHI / clearsky_GHI (cloud indicator)
- Weather encoding: weather_description → numerical (0-10 scale)
- Lag features: pv_lag1/24/168, ghi_lag1/24/168, etc.
- Rolling features: pv_roll3h/6h, ghi_roll3h/6h, etc.
- Time index: sequential integer for time series models
- Series ID: constant identifier for grouping

Usage:
    python scripts/preprocess_data.py

Output:
    outputs/processed.parquet (~2MB)

This file can be used by all training scripts with --processed-path argument.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pv_forecasting.pipeline import load_and_engineer_features, persist_processed


def main():
    """Main preprocessing function."""
    print("\n" + "=" * 70)
    print("PV FORECASTING - DATA PREPROCESSING")
    print("=" * 70)

    # Configuration
    raw_pv_path = Path("data/raw/pv_dataset.xlsx")
    raw_wx_path = Path("data/raw/wx_dataset.xlsx")
    output_dir = Path("outputs")
    local_timezone = "Australia/Sydney"

    print(f"\nInput files:")
    print(f"  PV data: {raw_pv_path}")
    print(f"  Weather data: {raw_wx_path}")
    print(f"  Timezone: {local_timezone}")

    # Load and engineer features
    print(f"\n{'='*70}")
    print("STEP 1: Loading raw data and engineering features...")
    print(f"{'='*70}\n")

    df = load_and_engineer_features(
        pv_path=raw_pv_path,
        wx_path=raw_wx_path,
        local_tz=local_timezone,
        lag_hours=(1, 24, 168),  # 1h, 1day, 1week lags
        rolling_hours=(3, 6),  # 3h and 6h rolling averages
        include_solar=True,  # Add solar position angles
        include_clearsky=True,  # Add clear-sky irradiance + kc
        dropna=True,  # Drop rows with NaN in critical columns
    )

    print(f"Processed {len(df)} samples with {len(df.columns)} features")

    # Display dataset info
    print(f"\n{'='*70}")
    print("STEP 2: Dataset information")
    print(f"{'='*70}\n")

    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total hours: {len(df)}")
    print(f"Total days: {len(df) / 24:.1f}")
    print(f"\nFeatures ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col:25s} (dtype: {df[col].dtype}, " f"NaN: {df[col].isna().sum()}/{len(df)})")

    # Check weather_description encoding
    print(f"\n{'='*70}")
    print("STEP 3: Verifying weather_description encoding")
    print(f"{'='*70}\n")

    if "weather_description" in df.columns:
        print(f"weather_description successfully encoded to numerical values")
        print(f"  Data type: {df['weather_description'].dtype}")
        print(f"  Unique values: {sorted(df['weather_description'].unique())}")
        print(f"  Value distribution:")
        print(df["weather_description"].value_counts().sort_index().to_string())
        print(f"\n  Encoding scheme:")
        print(f"    10.0 = clear/sunny (max PV production)")
        print(f"     8.0 = few clouds")
        print(f"     6.0 = scattered clouds")
        print(f"     4.0 = overcast/cloudy")
        print(f"     2.0 = light rain")
        print(f"     1.0 = heavy rain/storm")
        print(f"     0.0 = fog/mist")
        print(f"     5.0 = unknown/other")
    else:
        print(f"WARNING: weather_description column not found in dataset")

    # Check clearness index (kc)
    print(f"\n{'='*70}")
    print("STEP 4: Verifying clearness index (kc)")
    print(f"{'='*70}\n")

    if "kc" in df.columns:
        print(f"Clearness index (kc) successfully computed")
        print(f"  Data type: {df['kc'].dtype}")
        print(f"  Range: [{df['kc'].min():.2f}, {df['kc'].max():.2f}]")
        print(f"  Mean: {df['kc'].mean():.2f}")
        print(f"  NaN count: {df['kc'].isna().sum()} " f"({100*df['kc'].isna().sum()/len(df):.1f}%)")
        print(f"\n  kc = measured_GHI / clearsky_GHI:")
        print(f"    1.0 = perfect clear sky")
        print(f"    0.5 = 50% blocked by clouds")
        print(f"    0.0 = nighttime or fully overcast")
    else:
        print(f"WARNING: kc column not found in dataset")

    # Check for problematic columns
    print(f"\n{'='*70}")
    print("STEP 5: Checking data quality")
    print(f"{'='*70}\n")

    problems = []
    for col in df.columns:
        nan_count = df[col].isna().sum()
        nan_pct = 100 * nan_count / len(df)

        if nan_pct > 50:
            problems.append(f"  {col}: {nan_pct:.1f}% NaN (CRITICAL)")
        elif nan_pct > 10:
            print(f"  WARNING: {col}: {nan_pct:.1f}% NaN (expected for lag features)")

    if problems:
        print("Found critical issues:")
        for p in problems:
            print(p)
        print("\nWARNING: Dataset has problematic columns!")
    else:
        print("No critical data quality issues found")

    # Save to parquet
    print(f"\n{'='*70}")
    print("STEP 6: Saving processed dataset")
    print(f"{'='*70}\n")

    output_path = persist_processed(df, output_dir)
    file_size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"Saved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # Final summary
    print(f"\n{'='*70}")
    print("SUCCESS! Dataset preprocessing completed")
    print(f"{'='*70}\n")

    print(f"Next steps:")
    print(f"  1. Train LightGBM:")
    print(f"     python training_scripts/train_lgbm.py \\")
    print(f"       --processed-path {output_path}")
    print(f"\n  2. Train CNN-BiLSTM:")
    print(f"     python training_scripts/train_cnn_bilstm.py \\")
    print(f"       --processed-path {output_path}")
    print(f"\n  3. Train TFT:")
    print(f"     python training_scripts/train_tft.py \\")
    print(f"       --processed-path {output_path}")
    print(f"\n  4. Create ensemble after training all 3 models:")
    print(f"     python scripts/ensemble.py \\")
    print(f"       --lgbm outputs_lgbm/predictions_test_lgbm.csv \\")
    print(f"       --tft outputs_tft/predictions_test_tft.csv \\")
    print(f"       --bilstm outputs_cnn/predictions_test.csv \\")
    print(f"       --outdir outputs_ensemble")
    print()


if __name__ == "__main__":
    main()
