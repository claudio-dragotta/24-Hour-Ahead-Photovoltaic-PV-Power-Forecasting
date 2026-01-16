#!/usr/bin/env python3
"""Data augmentation: add Gaussian noise to training data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_noise_augmentation(df: pd.DataFrame, noise_level: float = 0.02, n_augmented: int = 1):
    """
    Add augmented samples with Gaussian noise.

    Args:
        df: Original dataframe
        noise_level: Std of Gaussian noise (relative to feature std)
        n_augmented: Number of augmented copies per original sample

    Returns:
        Augmented dataframe (original + augmented)
    """
    print(f"Original samples: {len(df)}")

    # Features to augment (weather + solar, NOT PV target or metadata)
    metadata_cols = ["lat", "lon", "pv_kwp_max", "pv", "test_pv_kwp_max"]
    cols_to_augment = [c for c in df.columns if c not in metadata_cols]

    augmented_dfs = [df.copy()]  # Start with original

    for i in range(n_augmented):
        df_aug = df.copy()

        # Add noise to features
        for col in cols_to_augment:
            if col in df_aug.columns:
                feature_std = df[col].std()
                if feature_std > 0:
                    noise = np.random.normal(0, noise_level * feature_std, size=len(df))
                    df_aug[col] = df[col] + noise

                    # Clip to valid range [0, 1] for scaled features
                    if df_aug[col].min() >= 0 and df_aug[col].max() <= 1.1:
                        df_aug[col] = df_aug[col].clip(0, 1)

        augmented_dfs.append(df_aug)

    df_final = pd.concat(augmented_dfs, ignore_index=True)
    print(f"Augmented samples: {len(df_final)} (+{len(df_final) - len(df)} new)")

    return df_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file")
    parser.add_argument("--noise-level", type=float, default=0.02, help="Noise level (std relative to feature std)")
    parser.add_argument("--n-augmented", type=int, default=1, help="Number of augmented copies per sample")
    args = parser.parse_args()

    print("=" * 80)
    print("DATA AUGMENTATION - NOISE INJECTION")
    print("=" * 80)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Noise level: {args.noise_level} (relative to feature std)")
    print(f"Augmented copies: {args.n_augmented}")
    print("")

    # Load data
    df = pd.read_parquet(args.input)

    # Augment
    df_augmented = add_noise_augmentation(df, args.noise_level, args.n_augmented)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_augmented.to_parquet(output_path)

    print(f"\nAugmented data saved to: {output_path}")
    print(f"   Shape: {df_augmented.shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()
