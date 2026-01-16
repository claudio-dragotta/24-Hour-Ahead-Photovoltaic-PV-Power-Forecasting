from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

INPUT_CSV = Path("data/processed/merged/pv_wx_combined.csv")
OUTPUT_PARQUET = Path("data/processed/merged/pv_wx_combined.parquet")
TARGET_COLUMN = "pv"
CATEGORICAL_COLUMNS = ["weather_description"]
INPUT_LEN = 24
OUTPUT_LEN = 24
WINDOW_STEP = 1


def one_hot_encode(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)


def normalize(df, exclude_columns=None):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if exclude_columns is not None:
        cols_to_scale = [col for col in numeric_cols if col not in exclude_columns]
    else:
        cols_to_scale = numeric_cols
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df, scaler


def create_sliding_windows(df, input_len, output_len, feature_columns, target_column, step=1):
    """Build sliding windows of features and targets."""
    X = []
    y = []

    feature_values = df[feature_columns].values
    target_values = df[target_column].values

    max_start = len(df) - input_len - output_len + 1
    for start in range(0, max_start, step):
        end = start + input_len
        target_end = end + output_len
        X.append(feature_values[start:end])
        y.append(target_values[end:target_end])

    return np.array(X), np.array(y)


def main():
    # Carica il CSV
    df = pd.read_csv(INPUT_CSV, parse_dates=True, index_col=0)

    # Rimuovi colonne completamente NaN (es. dt_iso vuota dopo il merge)
    all_nan_cols = [col for col in df.columns if df[col].isna().all()]
    if all_nan_cols:
        print(f"Rimuovo colonne tutte NaN: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    # Gestione valori mancanti
    if "rain_1h" in df.columns:
        df["rain_1h"] = df["rain_1h"].fillna(0)
    if "clouds_all" in df.columns and df["clouds_all"].isnull().any():
        df["clouds_all"] = df["clouds_all"].fillna(0)

    # Rimozione colonne costanti (lat, lon e altre)
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if "lat" in constant_cols:
        print("Rimuovo colonna costante lat")
    if "lon" in constant_cols:
        print("Rimuovo colonna costante lon")
    if constant_cols:
        print(f"Rimuovo colonne costanti: {constant_cols}")
        df = df.drop(columns=constant_cols)

    # Rimozione feature sempre zero o sempre uguali
    zero_cols = [col for col in df.columns if (df[col] == 0).all()]
    if zero_cols:
        print(f"Rimuovo colonne sempre zero: {zero_cols}")
        df = df.drop(columns=zero_cols)

    # One-hot encoding su weather_description
    df = one_hot_encode(df, CATEGORICAL_COLUMNS)

    # Normalizzazione (escludi la colonna target)
    exclude_columns = [TARGET_COLUMN]
    df, scaler = normalize(df, exclude_columns=exclude_columns)

    # Sliding windows
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    X, y = create_sliding_windows(
        df, INPUT_LEN, OUTPUT_LEN, feature_columns, TARGET_COLUMN, step=WINDOW_STEP
    )

    # Salva come parquet/pickle
    out = {
        "X": X,
        "y": y,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "scaler_min": scaler.data_min_,
        "scaler_max": scaler.data_max_,
    }
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, OUTPUT_PARQUET)
    print(f"Saved preprocessed data to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
