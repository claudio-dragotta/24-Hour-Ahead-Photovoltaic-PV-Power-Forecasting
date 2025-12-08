"""Generate processed parquet with lag72 (3 days) features.

This script creates a new processed.parquet file that includes lag features
for 1h, 24h, 72h (3 days), and 168h (1 week).
"""

from pathlib import Path

from pv_forecasting.pipeline import load_and_engineer_features, persist_processed


def main():
    print("=" * 60)
    print("Generating processed data with lag72 (3 days) features")
    print("=" * 60)

    # Paths
    pv_path = Path("data/raw/pv_dataset.xlsx")
    wx_path = Path("data/raw/wx_dataset.xlsx")
    output_dir = Path("outputs_lag72")

    # Feature engineering with lag72
    print("\nLoading and engineering features...")
    print("Lag hours: 1, 24, 72 (NEW!), 168")
    print("Rolling windows: 3, 6")

    df = load_and_engineer_features(
        pv_path=pv_path,
        wx_path=wx_path,
        local_tz="Australia/Sydney",
        lag_hours=[1, 24, 72, 168],  # Added lag72!
        rolling_hours=[3, 6],
    )

    print(f"\nGenerated {len(df)} samples with {len(df.columns)} features")

    # Check new lag72 features
    lag72_features = [col for col in df.columns if "lag72" in col]
    print(f"\nNew lag72 features ({len(lag72_features)}):")
    for feat in sorted(lag72_features):
        print(f"  - {feat}")

    # Save
    print(f"\nSaving to {output_dir / 'processed.parquet'}...")
    persist_processed(df, output_dir)

    print("\n" + "=" * 60)
    print("Done! New processed.parquet created with lag72 features")
    print(f"Location: {output_dir / 'processed.parquet'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
