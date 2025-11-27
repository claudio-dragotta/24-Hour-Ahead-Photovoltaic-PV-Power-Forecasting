from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index.tz_convert("UTC")
    hour = idx.hour.values
    day_of_year = idx.dayofyear.values
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    return df


def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for l in lags:
            out[f"{c}_lag{l}"] = out[c].shift(l)
    return out


def add_rollings(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for w in windows:
            out[f"{c}_roll{w}m"] = out[c].rolling(window=w, min_periods=max(1, w // 2)).mean()
    return out

