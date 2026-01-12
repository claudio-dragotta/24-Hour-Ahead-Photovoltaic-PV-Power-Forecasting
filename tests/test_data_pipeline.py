"""Tests for data alignment and feature engineering pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pv_forecasting import data, pipeline


def test_align_hourly_keep_wx_future():
    pv_idx = pd.date_range("2021-01-01", periods=2, freq="h", tz="UTC")
    wx_idx = pd.date_range("2021-01-01", periods=3, freq="h", tz="UTC")

    pv_df = pd.DataFrame({"pv": [1.0, 2.0]}, index=pv_idx)
    wx_df = pd.DataFrame(
        {"ghi": [100.0, 200.0, 300.0], "weather_description": ["clear", "clear", "rain"]},
        index=wx_idx,
    )

    out_keep = data.align_hourly(pv_df, wx_df, keep_wx_future=True)
    out_drop = data.align_hourly(pv_df, wx_df, keep_wx_future=False)

    assert len(out_keep) == 3
    assert pd.isna(out_keep.loc[wx_idx[2], "pv"])  # future weather rows retained with NaN pv
    assert len(out_drop) == 2
    assert out_drop.index.equals(pv_idx)  # restricted to PV timestamps when keep_wx_future=False

    # Timezone preserved and weather_description kept for encoding
    assert str(out_keep.index.tz) == "UTC"
    assert "weather_description" in out_keep.columns


def test_load_and_engineer_features_monkeypatched(monkeypatch):
    # Synthetic PV + weather data
    idx = pd.date_range("2021-01-01", periods=10, freq="h", tz="UTC")
    pv_df = pd.DataFrame({"pv": np.linspace(0.1, 1.0, len(idx))}, index=idx)
    wx_df = pd.DataFrame(
        {
            "ghi": np.linspace(100, 200, len(idx)),
            "dni": np.linspace(50, 150, len(idx)),
            "dhi": np.linspace(25, 75, len(idx)),
            "temp": np.linspace(10, 20, len(idx)),
            "humidity": np.linspace(40, 60, len(idx)),
            "wind_speed": np.linspace(1, 5, len(idx)),
            "clouds": np.linspace(0, 100, len(idx)),
            "weather_description": ["clear"] * len(idx),
        },
        index=idx,
    )

    # Monkeypatch heavy I/O and pvlib-dependent functions to lightweight stubs
    monkeypatch.setattr(pipeline, "load_pv_xlsx", lambda path, local_tz: pv_df)
    monkeypatch.setattr(pipeline, "load_wx_xlsx", lambda path: wx_df)

    def fake_add_solar_position(df):
        df = df.copy()
        df["sp_zenith"] = 0.5
        df["sp_azimuth"] = 1.0
        return df

    def fake_add_clearsky(df):
        df = df.copy()
        df["cs_ghi"] = df["ghi"] + 1
        df["cs_dni"] = df["dni"] + 1
        df["cs_dhi"] = df["dhi"] + 1
        return df

    def fake_add_kc(df, ghi_col):
        df = df.copy()
        df["kc"] = df[ghi_col] / df["cs_ghi"]
        return df

    monkeypatch.setattr(pipeline, "add_solar_position", fake_add_solar_position)
    monkeypatch.setattr(pipeline, "add_clearsky", fake_add_clearsky)
    monkeypatch.setattr(pipeline, "add_kc", fake_add_kc)

    result = pipeline.load_and_engineer_features(
        pv_path=Path("fake_pv.xlsx"),
        wx_path=Path("fake_wx.xlsx"),
        local_tz="UTC",
        lag_hours=(1, 2),
        rolling_hours=(2,),
        include_solar=True,
        include_clearsky=True,
        dropna=True,
        keep_wx_future=False,
    )

    # After dropping lag-induced NaN rows, length should shrink by max lag (2)
    assert len(result) == len(idx) - 2

    # Index should be UTC and sorted
    assert str(result.index.tz) == "UTC"
    assert result.index.is_monotonic_increasing

    # Core engineered columns present and non-null
    required_cols = [
        "pv",
        "ghi",
        "pv_lag1",
        "pv_lag2",
        "pv_roll2h",
        "kc",
        "time_idx",
        "series_id",
        "sp_zenith",
    ]
    for col in required_cols:
        assert col in result.columns
        assert not result[col].isna().any(), f"{col} contains NaN values"

    # time_idx sequential and series_id constant
    assert result["time_idx"].tolist() == list(range(len(result)))
    assert result["series_id"].nunique() == 1
    assert result["series_id"].iloc[0] == "pv_site_1"
