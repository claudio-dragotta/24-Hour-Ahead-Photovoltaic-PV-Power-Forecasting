#!/usr/bin/env python3
"""Merge PV and WX CSVs into a single file with correct timezone/DST handling.

Reads:
  data/processed/merged/pv_combined.csv
  data/processed/merged/wx_combined.csv

Writes:
  data/processed/merged/pv_wx_combined.csv

Usage:
  source .venv/bin/activate
  python3 scripts/preprocessing/merge_pv_wx.py --local-tz Australia/Sydney
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-csv", default="data/processed/merged/pv_combined.csv")
    ap.add_argument("--wx-csv", default="data/processed/merged/wx_combined.csv")
    ap.add_argument("--out", default="data/processed/merged/pv_wx_combined.csv")
    ap.add_argument("--local-tz", default="Australia/Sydney")
    args = ap.parse_args()

    try:
        import pandas as pd
    except Exception as e:
        raise SystemExit("pandas not available. Activate venv and install pandas.") from e

    from pv_forecasting.timeutils import localize_pv_index, parse_wx_index_with_tz, to_utc
    from pv_forecasting.data import align_hourly

    pv_path = Path(args.pv_csv)
    wx_path = Path(args.wx_csv)
    out_path = Path(args.out)

    if not pv_path.exists():
        raise SystemExit(f"PV CSV not found: {pv_path}")
    if not wx_path.exists():
        raise SystemExit(f"WX CSV not found: {wx_path}")

    # Read PV CSV, preserve raw strings
    pv = pd.read_csv(pv_path, dtype=str)
    pv_cols = pv.columns.tolist()
    if len(pv_cols) < 2:
        raise SystemExit("PV CSV must have at least two columns (timestamp, pv)")
    pv_ts_col = pv_cols[0]
    pv_val_col = pv_cols[1]

    # Localize PV timestamps (assumed naive local times) using robust DST handling
    pv_idx = localize_pv_index(pv[pv_ts_col], args.local_tz)
    # Use .values to avoid pandas aligning the numeric Series by its integer index
    pv_df = pd.DataFrame({"pv": pd.to_numeric(pv[pv_val_col], errors="coerce").values}, index=pv_idx)
    pv_df.index = pv_df.index.tz_convert("UTC")
    pv_df = pv_df[~pv_df.index.duplicated(keep="first")]

    # Read WX CSV and parse timezone-aware timestamps
    wx = pd.read_csv(wx_path, dtype=str)
    wx_cols = wx.columns.tolist()
    if len(wx_cols) < 2:
        raise SystemExit("WX CSV seems invalid")
    wx_ts_col = wx_cols[0]
    # parse weather timestamps with tz
    wx_idx = parse_wx_index_with_tz(wx[wx_ts_col])
    # Keep other columns. Do NOT coerce weather_description to numeric — keep as string.
    wx_data = wx.drop(columns=[wx_ts_col])
    from collections import Counter
    # Convert numeric where appropriate, but preserve columns containing 'weather' and 'description'
    for c in wx_data.columns:
        if 'weather' in c.lower() and 'description' in c.lower():
            # keep as-is (string)
            continue
        try:
            wx_data[c] = pd.to_numeric(wx_data[c], errors='coerce')
        except Exception:
            pass
    # If duplicate column labels exist (e.g., weather_description from multiple sheets),
    # coalesce duplicates into a single column by taking the first non-null value left-to-right.
    col_counts = Counter(wx_data.columns)
    if any(v > 1 for v in col_counts.values()):
        for name, cnt in list(col_counts.items()):
            if cnt <= 1:
                continue
            dup_cols = [c for c in wx_data.columns if c == name]
            # bfill across columns then take the first column (first non-null left-to-right)
            wx_data[name] = wx_data[dup_cols].bfill(axis=1).iloc[:, 0]
            # drop the duplicate columns keeping the first occurrence
            keep = []
            dropped = []
            seen = set()
            for c in wx_data.columns:
                if c in seen:
                    dropped.append(c)
                else:
                    keep.append(c)
                    seen.add(c)
            wx_data = wx_data.loc[:, keep]
    wx_df = wx_data.copy()
    wx_df.index = wx_idx

    # Align and merge using existing pipeline helper (forward-fill weather, keep PV rows)
    merged = align_hourly(pv_df, wx_df, keep_wx_future=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index_label="timestamp_utc")
    print(f"Wrote merged dataset to {out_path}")


if __name__ == "__main__":
    main()
