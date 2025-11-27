import pandas as pd

df = pd.read_csv('data/processed/merged_dataset.csv', index_col=0, parse_dates=True)

print('=' * 60)
print('VERIFICA DATASET PULITO')
print('=' * 60)

print('\nPrime 15 righe:')
print(df.head(15)[['pv', 'Ghi', 'temp', 'humidity']])

print('\nVerifica NaN:')
print(f'PV null: {df["pv"].isna().sum()}')
print(f'Ghi null: {df["Ghi"].isna().sum()}')
print(f'Temp null: {df["temp"].isna().sum()}')

print('\nTimestamp examples (mantiene millisecondi):')
for i in [0, 5, 10, 100, 1000, 5000]:
    print(f'  {df.index[i]}')

print(f'\nTotale righe: {len(df)}')
print(f'Colonne: {len(df.columns)}')
print(f'Periodo completo: {df.index.min()} -> {df.index.max()}')
