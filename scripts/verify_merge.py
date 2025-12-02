"""Verifica l'integrità dell'unione dei dataset"""
from pathlib import Path

import pandas as pd

from pv_forecasting.data import load_pv_xlsx, load_wx_xlsx, align_hourly
from pv_forecasting.features import standardize_feature_columns

print("=" * 60)
print("VERIFICA INTEGRITÀ UNIONE DATASET")
print("=" * 60)

# Carica i dataset separatamente
print("\n1. Caricamento dataset PV...")
pv = load_pv_xlsx(Path('data/raw/pv_dataset.xlsx'), 'Australia/Sydney')
print(f"   Righe PV: {len(pv)}")
print(f"   Periodo: {pv.index.min()} -> {pv.index.max()}")
print(f"   Valori non-null: {pv['pv'].notna().sum()}")
print(f"   Valori null: {pv['pv'].isna().sum()}")

print("\n2. Caricamento dataset Meteo...")
wx = load_wx_xlsx(Path('data/raw/wx_dataset.xlsx'))
print(f"   Righe WX: {len(wx)}")
print(f"   Periodo: {wx.index.min()} -> {wx.index.max()}")

print("\n3. Analisi sovrapposizione temporale...")
common_idx = pv.index.intersection(wx.index)
pv_only = pv.index.difference(wx.index)
wx_only = wx.index.difference(pv.index)

print(f"   Timestamp in comune: {len(common_idx)}")
print(f"   Solo in PV: {len(pv_only)}")
print(f"   Solo in WX: {len(wx_only)}")

if len(pv_only) > 0:
    print(f"\n   [!] Primi timestamp solo in PV:")
    print(f"      {pv_only[:5].tolist()}")

if len(wx_only) > 0:
    print(f"\n   [!] Primi timestamp solo in WX:")
    print(f"      {wx_only[:5].tolist()}")

print("\n4. Unione con align_hourly...")
merged = align_hourly(pv, wx)
merged = standardize_feature_columns(merged)
print(f"   Righe dopo unione: {len(merged)}")
print(f"   Colonne: {merged.columns.tolist()}")
print(f"   Periodo: {merged.index.min()} -> {merged.index.max()}")

print("\n5. Verifica integrità dati PV nel merged...")
print(f"   PV non-null: {merged['pv'].notna().sum()}")
print(f"   PV null: {merged['pv'].isna().sum()}")
print(f"   PV min: {merged['pv'].min():.3f}")
print(f"   PV max: {merged['pv'].max():.3f}")
print(f"   PV media: {merged['pv'].mean():.3f}")

print("\n6. Controllo continuità temporale...")
expected_hours = pd.date_range(merged.index.min(), merged.index.max(), freq='h', tz='UTC')
missing_hours = expected_hours.difference(merged.index)
print(f"   Ore attese: {len(expected_hours)}")
print(f"   Ore presenti: {len(merged)}")
print(f"   Ore mancanti: {len(missing_hours)}")

if len(missing_hours) > 0:
    print(f"   [!] Prime ore mancanti: {missing_hours[:5].tolist()}")

print("\n7. Campione dati uniti (prime 20 righe con PV > 0)...")
sample = merged[merged['pv'] > 0][['pv', 'ghi', 'temp', 'humidity']].head(20)
print(sample.to_string())

print("\n8. Verifica allineamento PV-GHI (correlazione attesa)...")
# PV dovrebbe correlere con GHI (irradianza)
for col in ["ghi", "dni", "dhi"]:
    if col not in merged.columns:
        merged[col] = pd.NA
correlation = merged[['pv', 'ghi', 'dni', 'dhi']].corr()['pv']
print(f"   Correlazione PV-GHI: {correlation['ghi']:.3f}")
print(f"   Correlazione PV-DNI: {correlation['dni']:.3f}")
print(f"   Correlazione PV-DHI: {correlation['dhi']:.3f}")

if correlation['ghi'] > 0.7:
    print("   [OK] Correlazione PV-GHI buona (> 0.7)")
else:
    print(f"   [!] Correlazione PV-GHI bassa ({correlation['ghi']:.3f})")

print("\n9. Verifica timestamp specifici...")
# Verifica un timestamp casuale
test_ts = pd.Timestamp('2010-07-15 12:00:00', tz='UTC')
if test_ts in merged.index:
    row = merged.loc[test_ts]
    print(f"   2010-07-15 12:00 UTC:")
    print(f"     PV: {row['pv']:.2f} kW")
    if "ghi" in row:
        print(f"     GHI: {row['ghi']:.2f} W/m²")
    if "temp" in row:
        print(f"     Temp: {row['temp']:.2f} K")

print("\n" + "=" * 60)
print("VERIFICA COMPLETATA")
print("=" * 60)
