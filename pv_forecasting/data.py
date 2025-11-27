from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from .timeutils import localize_pv_index, parse_wx_index_with_tz, to_utc


def load_pv_xlsx(path: Path, local_tz: str) -> pd.DataFrame:
    # Read all sheets in the Excel file
    xl = pd.ExcelFile(path)
    dfs = []
    
    for sheet_name in xl.sheet_names:
        # Skip first row (header like "Max kWp  82.41"), use second row as header
        df = pd.read_excel(path, sheet_name=sheet_name, header=None, skiprows=1)
        # First column is timestamp, second is PV power
        df.columns = ['timestamp', 'pv']
        
        idx_local = localize_pv_index(df['timestamp'], local_tz)
        idx_utc = to_utc(idx_local)
        # Reset index name to avoid conflicts
        idx_utc.name = None
        pv_vals = pd.to_numeric(df['pv'], errors="coerce").values
        sheet_df = pd.DataFrame({"pv": pv_vals}, index=idx_utc)
        dfs.append(sheet_df)
    
    # Concatenate all sheets
    out = pd.concat(dfs, axis=0)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def load_wx_xlsx(path: Path) -> pd.DataFrame:
    # Read all sheets in the Excel file
    xl = pd.ExcelFile(path)
    dfs = []
    
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        cols = {c.lower(): c for c in df.columns}
        ts_col = cols.get("dt_iso") or cols.get("timestamp") or list(df.columns)[0]

        idx_parsed = parse_wx_index_with_tz(df[ts_col])
        idx_utc = to_utc(idx_parsed)

        drop_cols: Iterable[str] = [ts_col]
        feat_df = df.drop(columns=list(drop_cols), errors="ignore")
        # Cast numeric where possible
        for c in feat_df.columns:
            feat_df[c] = pd.to_numeric(feat_df[c], errors="ignore")
        sheet_df = feat_df.copy()
        sheet_df.index = idx_utc
        dfs.append(sheet_df)
    
    # Concatenate all sheets
    out = pd.concat(dfs, axis=0)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def align_hourly(pv: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    # Join datasets with forward-fill interpolation for weather data
    pv_h = pv.copy()
    wx_h = wx.copy()
    
    # Drop non-numeric columns from weather data
    numeric_cols = wx_h.select_dtypes(include=[np.number]).columns
    wx_h = wx_h[numeric_cols]
    
    # Convert to UTC without rounding
    pv_h.index = pv_h.index.tz_convert("UTC")
    wx_h.index = wx_h.index.tz_convert("UTC")

    # Outer join to get all timestamps
    df = pv_h.join(wx_h, how="outer")
    
    # Sort by index
    df = df.sort_index()
    
    # Forward-fill weather data to match PV timestamps (weather changes slowly)
    weather_cols = [c for c in df.columns if c != 'pv']
    df[weather_cols] = df[weather_cols].ffill()
    
    # Keep only rows with PV data (removes weather-only timestamps)
    df = df[df['pv'].notna()]
    
    # Remove duplicates if any (keep first occurrence)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    
    return df


def save_processed(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "processed.parquet"
    df.to_parquet(out_path)
    return out_path


def save_history(history: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "history.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return p

