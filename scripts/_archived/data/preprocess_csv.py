"""ARCHIVE: preprocess_csv.py

Original: scripts/data/preprocess_csv.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pv_forecasting.features import (
    add_time_cyclical,
    add_calendar_flags,
    add_solar_position,
    add_clearsky,
    add_kc,
    add_lags,
    add_rollings_h,
    add_rolling_vars_h,
    standardize_feature_columns,
    encode_weather_onehot,
)


def load_unified_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_utc"]) if path.exists() else pd.DataFrame()
    if df.empty:
        return df
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.rename(columns={"pv_power": "pv"})
    for c in df.columns:
        if c == "weather_description":
            continue
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass
    return df


def main():
    input_path = Path("data/raw/pv_wx_combined.csv")
    df = load_unified_csv(input_path)
    if df.empty:
        print("No input CSV found; skipping.")
        return
    df = standardize_feature_columns(df)
    df = encode_weather_onehot(df, col="weather_description")
    df = add_time_cyclical(df)
    df = add_calendar_flags(df)
    df = add_solar_position(df)
    df = add_clearsky(df)
    if "ghi" in df.columns:
        df = add_kc(df, ghi_col="ghi")
        if "kc" in df.columns:
            df["kc"] = df["kc"].fillna(0.0)
    lag_hours = (1, 24, 48, 72, 96, 168)
    rolling_hours = (3, 6, 12, 24)
    lag_cols = [c for c in ["pv", "ghi", "dni", "dhi"] if c in df.columns]
    if lag_hours and lag_cols:
        df = add_lags(df, lag_cols, list(lag_hours))
    roll_cols = [c for c in ["pv", "ghi", "dni"] if c in df.columns]
    if rolling_hours and roll_cols:
        df = add_rollings_h(df, roll_cols, list(rolling_hours))
        df = add_rolling_vars_h(df, roll_cols, list(rolling_hours))
    df = df.sort_index()
    if "rain_1h" in df.columns:
        df["rain_1h"] = df["rain_1h"].fillna(0.0)
    critical_cols = [c for c in df.columns if any(x in c for x in ["pv", "ghi", "dni", "dhi", "lag", "roll", "var"])]
    df = df.dropna(subset=critical_cols)
    df = df.copy()
    df["time_idx"] = np.arange(len(df))
    df["series_id"] = "pv_site_1"
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "processed.parquet")
    print(f"Saved processed to {out_dir / 'processed.parquet'}")


if __name__ == "__main__":
    main()
