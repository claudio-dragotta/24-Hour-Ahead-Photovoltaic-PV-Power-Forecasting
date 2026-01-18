from __future__ import annotations

from pathlib import Path
import argparse
import math

import pandas as pd
import pvlib

from pv_forecasting.features import (
    add_calendar_flags,
    add_lags,
    add_rollings_h,
    add_rolling_vars_h,
    add_time_cyclical,
    encode_weather_onehot,
    standardize_feature_columns,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Preprocess raw PV/WX data to feature Parquet.")
    ap.add_argument("--merged-csv", type=str, default="data/processed/merged/pv_wx_combined.csv")
    ap.add_argument("--out", type=str, default="data/processed/merged/pv_wx_combined.parquet")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    print(f"Loading merged CSV from {args.merged_csv} ...")
    df = pd.read_csv(args.merged_csv, parse_dates=True, index_col=0)

    # Rimuovi colonne completamente NaN
    all_nan_cols = [col for col in df.columns if df[col].isna().all()]
    if all_nan_cols:
        print(f"Rimuovo colonne tutte NaN: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    # Fill specific NaN
    if "rain_1h" in df.columns:
        df["rain_1h"] = df["rain_1h"].fillna(0)
    if "clouds_all" in df.columns and df["clouds_all"].isnull().any():
        df["clouds_all"] = df["clouds_all"].fillna(0)

    # Standardize column names (lowercase, synonyms)
    df = standardize_feature_columns(df)

    # One-hot encoding per weather_description -> wx_* colonne
    if "weather_description" in df.columns:
        df = encode_weather_onehot(df, col="weather_description")

    # Calcolo feature solari/clear-sky (richiede lat/lon)
    if "lat" not in df.columns or "lon" not in df.columns:
        raise SystemExit("lat/lon columns are required to compute solar/clear-sky features")
    try:
        lat = float(df["lat"].iloc[0])
        lon = float(df["lon"].iloc[0])
    except Exception as e:
        raise SystemExit("Unable to parse lat/lon for solar feature computation") from e
    # Usa timezone UTC (index già UTC); altitudine sconosciuta -> 0 m
    loc = pvlib.location.Location(latitude=lat, longitude=lon, tz="UTC", altitude=0)
    times = df.index
    # Solar position
    solar_pos = loc.get_solarposition(times=times)
    df["sp_zenith"] = solar_pos["zenith"].astype(float)
    df["sp_azimuth"] = solar_pos["azimuth"].astype(float)
    # Clear-sky (Ineichen)
    cs = loc.get_clearsky(times=times, model="ineichen")
    df["cs_ghi"] = cs["ghi"].astype(float)
    df["cs_dni"] = cs["dni"].astype(float)
    df["cs_dhi"] = cs["dhi"].astype(float)
    # Clearness index kc = ghi / cs_ghi
    if "ghi" in df.columns:
        kc = df["ghi"] / df["cs_ghi"].replace(0, math.nan)
        df["kc"] = kc.replace([math.inf, -math.inf], math.nan).fillna(0)

    # Time and calendar features
    df = add_time_cyclical(df)
    df = add_calendar_flags(df)

    # Lags e rolling su colonne chiave (se presenti)
    lag_cols = [c for c in ["pv", "ghi", "dni", "dhi", "temp"] if c in df.columns]
    if lag_cols:
        df = add_lags(df, lag_cols, [1, 24, 168])
    roll_cols = [c for c in ["pv", "ghi", "dni"] if c in df.columns]
    if roll_cols:
        df = add_rollings_h(df, roll_cols, [3, 6, 24, 168])
        df = add_rolling_vars_h(df, roll_cols, [3, 6, 24, 168])

    df = df.sort_index()

    # Fill NaN per preservare tutte le righe (es. prime righe con lag mancanti)
    df = df.fillna(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"Saved feature table to {out_path} with {len(df)} rows and {len(df.columns)} columns")


if __name__ == "__main__":
    main()
