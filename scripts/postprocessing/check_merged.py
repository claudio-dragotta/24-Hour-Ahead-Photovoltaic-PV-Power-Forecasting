from pathlib import Path

import pandas as pd

from pv_forecasting.features import standardize_feature_columns

parquet_path = Path("outputs/processed.parquet")
csv_path = Path("data/processed/merged_dataset.csv")
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
else:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

df = standardize_feature_columns(df)

print('=' * 60)
print('VERIFICA DATASET PULITO')
print('=' * 60)

print('\nPrime 15 righe:')
print(df.head(15)[['pv', 'ghi', 'temp', 'humidity']])

print('\nVerifica NaN:')
print(f'PV null: {df["pv"].isna().sum()}')
print(f'GHI null: {df["ghi"].isna().sum() if "ghi" in df.columns else 0}')
print(f'Temp null: {df["temp"].isna().sum()}')

print('\nTimestamp examples (mantiene millisecondi):')
for i in [0, 5, 10, 100, 1000, 5000]:
    print(f'  {df.index[i]}')

print(f'\nTotale righe: {len(df)}')
print(f'Colonne: {len(df.columns)}')
print(f'Periodo completo: {df.index.min()} -> {df.index.max()}')
