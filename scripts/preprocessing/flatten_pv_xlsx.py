#!/usr/bin/env python3
"""Unisce tutte le tabelle (sheet) presenti in `data/raw/pv_dataset.xlsx`
in un unico CSV `data/processed/merged/pv_combined.csv`.

Esempio di esecuzione:
  source .venv/bin/activate
  python3 scripts/preprocessing/flatten_pv_xlsx.py

Il file risultante verrà scritto in:
  data/processed/merged/pv_combined.csv
"""
from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    try:
        import pandas as pd
    except Exception as e:
        print("Errore: pandas non è installato. Attiva il virtualenv e installa pandas.")
        raise

    pv_xlsx = Path("data/raw/pv_dataset.xlsx")
    if not pv_xlsx.exists():
        print(f"File non trovato: {pv_xlsx}")
        sys.exit(1)

    print(f"Leggo tutte le sheet da {pv_xlsx}...")
    all_sheets = pd.read_excel(pv_xlsx, sheet_name=None, header=None)

    dfs = []
    for name, df in all_sheets.items():
        # Se il file ha header non standard (metadati nella prima riga),
        # proviamo a rilevare colonne a due colonne (timestamp, pv) se header=None
        # Se il dataframe ha due colonne, assumiamo [timestamp, pv]
        df = df.copy()
        if df.shape[1] == 2:
            df.columns = ["timestamp_utc", "pv"]
        else:
            # Se ha intestazione nella prima riga, ricarichiamo con header=0
            df = pd.read_excel(pv_xlsx, sheet_name=name, header=0)
        # Convert timestamp column to datetime (se presente)
        ts_cols = [c for c in df.columns if "timestamp" in str(c).lower() or "time" in str(c).lower()]
        if ts_cols:
            df[ts_cols[0]] = pd.to_datetime(df[ts_cols[0]], errors="coerce", format="mixed")
            df = df.set_index(ts_cols[0])
        dfs.append(df)

    # Concatenate e ordinare per indice temporale se presente
    print(f"Concateno {len(dfs)} sheet...")
    combined = pd.concat(dfs, axis=0)
    try:
        combined = combined.sort_index()
    except Exception:
        pass
    # Se l'indice è temporale, rimuovi eventuali righe con timestamp NaT
    if isinstance(combined.index, pd.DatetimeIndex):
        combined = combined.loc[~combined.index.isna()]

    out_dir = Path("data/processed/merged")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_out = out_dir / "pv_combined.csv"

    print(f"Salvo CSV: {csv_out}")
    combined.to_csv(csv_out)

    print("Done.")


if __name__ == "__main__":
    main()
