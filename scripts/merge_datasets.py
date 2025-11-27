from __future__ import annotations

import argparse
from pathlib import Path

from pv_forecasting.data import load_pv_xlsx, load_wx_xlsx, align_hourly


def parse_args():
    ap = argparse.ArgumentParser(description="Merge PV and Weather datasets")
    ap.add_argument("--pv-path", type=str, default="data/raw/pv_dataset.xlsx", help="Path to PV dataset")
    ap.add_argument("--wx-path", type=str, default="data/raw/wx_dataset.xlsx", help="Path to weather dataset")
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney", help="Local timezone for PV data")
    ap.add_argument("--output", type=str, default="data/processed/merged_dataset.csv", help="Output file path (xlsx or csv)")
    return ap.parse_args()


def main():
    args = parse_args()
    
    print(f"Caricamento dataset PV da: {args.pv_path}")
    pv = load_pv_xlsx(Path(args.pv_path), args.local_tz)
    print(f"  - Righe: {len(pv)}, Colonne: {list(pv.columns)}")
    
    print(f"\nCaricamento dataset meteo da: {args.wx_path}")
    wx = load_wx_xlsx(Path(args.wx_path))
    print(f"  - Righe: {len(wx)}, Colonne: {list(wx.columns)}")
    
    print("\nUnione dei dataset...")
    merged = align_hourly(pv, wx)
    print(f"  - Dataset unito: {len(merged)} righe, {len(merged.columns)} colonne")
    print(f"  - Colonne: {list(merged.columns)}")
    print(f"  - Periodo: da {merged.index.min()} a {merged.index.max()}")
    
    output_path = Path(args.output)
    print(f"\nSalvataggio dataset unito in: {output_path}")
    
    if output_path.suffix.lower() == '.csv':
        merged.to_csv(output_path)
    elif output_path.suffix.lower() in ['.xlsx', '.xls']:
        merged.to_excel(output_path)
    else:
        # Default to CSV
        merged.to_csv(output_path)
    
    print(f"[OK] Dataset unito salvato con successo!")
    print(f"\nRiepilogo:")
    print(merged.describe())


if __name__ == "__main__":
    main()
