#!/usr/bin/env python3
"""Unisce tutte le sheet presenti in `data/raw/wx_dataset.xlsx` in un unico foglio/file.

Salva in:
  data/processed/merged/wx_combined_single_sheet.xlsx
  data/processed/merged/wx_combined_single_sheet.csv

Esempio:
  source .venv/bin/activate
  python3 scripts/preprocessing/flatten_wx_xlsx.py
"""
from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    try:
        import pandas as pd
    except Exception:
        print("Errore: pandas non è installato. Attiva il virtualenv e installa pandas.")
        raise

    wx_xlsx = Path("data/raw/wx_dataset.xlsx")
    if not wx_xlsx.exists():
        print(f"File non trovato: {wx_xlsx}")
        sys.exit(1)

    print(f"Leggo tutte le sheet da {wx_xlsx} (preservo la colonna A senza modifiche)...")
    # Leggi tutte le sheet come stringhe per preservare esattamente il contenuto
    all_sheets = pd.read_excel(wx_xlsx, sheet_name=None, dtype=str, engine="openpyxl")

    dfs = []
    for name, df in all_sheets.items():
        # Non convertire o interpretare le colonne: mantieni i valori così come letti
        df = df.copy()
        dfs.append(df)

    print(f"Concateno {len(dfs)} sheet...")
    combined = pd.concat(dfs, axis=0, ignore_index=True)

    out_dir = Path("data/processed/merged")
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx_out = out_dir / "wx_combined_single_sheet.xlsx"
    csv_out = out_dir / "wx_combined_single_sheet.csv"

    # IMPORTANT: do not modify column A (first column) or any timestamps — keep raw values.
    # The combined dataframe preserves the original cell text as read by openpyxl/pandas.

    print(f"Salvo Excel: {xlsx_out}")
    combined.to_excel(xlsx_out, sheet_name="merged")
    print(f"Salvo CSV: {csv_out}")
    combined.to_csv(csv_out)

    print("Done.")


if __name__ == "__main__":
    main()
