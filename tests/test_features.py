"""Tests for pv_forecasting.features module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_forecasting.features import (
    add_clearsky,
    add_kc,
    add_lags,
    add_rollings_h,
    add_solar_position,
    add_time_cyclical,
    standardize_feature_columns,
)


class TestTimeCyclical:
    """Tests for add_time_cyclical function."""

    def test_add_time_cyclical(self, sample_merged_data):
        """Test that cyclical time features are added correctly."""
        df = add_time_cyclical(sample_merged_data)

        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        assert "doy_sin" in df.columns
        assert "doy_cos" in df.columns

        # Check bounds
        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()
        assert df["doy_sin"].between(-1, 1).all()
        assert df["doy_cos"].between(-1, 1).all()

    def test_cyclical_properties(self, sample_merged_data):
        """Test mathematical properties of cyclical features."""
        df = add_time_cyclical(sample_merged_data)

        # sin^2 + cos^2 = 1 for hour features
        hour_identity = df["hour_sin"] ** 2 + df["hour_cos"] ** 2
        np.testing.assert_array_almost_equal(hour_identity, 1.0, decimal=5)

        # sin^2 + cos^2 = 1 for doy features
        doy_identity = df["doy_sin"] ** 2 + df["doy_cos"] ** 2
        np.testing.assert_array_almost_equal(doy_identity, 1.0, decimal=5)


class TestLagFeatures:
    """Tests for add_lags function."""

    def test_add_lags_single_column(self, sample_merged_data):
        """Test adding lag features for a single column."""
        df = add_lags(sample_merged_data, cols=["pv"], lags=[1, 24])

        assert "pv_lag1" in df.columns
        assert "pv_lag24" in df.columns

        # Check that lag values are correct
        assert pd.isna(df["pv_lag1"].iloc[0])
        assert df["pv_lag1"].iloc[1] == pytest.approx(df["pv"].iloc[0])

    def test_add_lags_multiple_columns(self, sample_merged_data):
        """Test adding lags for multiple columns."""
        df = add_lags(sample_merged_data, cols=["pv", "ghi"], lags=[1, 24])

        assert "pv_lag1" in df.columns
        assert "pv_lag24" in df.columns
        assert "ghi_lag1" in df.columns
        assert "ghi_lag24" in df.columns

    def test_lags_preserve_shape(self, sample_merged_data):
        """Test that lags preserve dataframe shape (rows)."""
        df = add_lags(sample_merged_data, cols=["pv"], lags=[1])

        assert len(df) == len(sample_merged_data)


class TestRollingFeatures:
    """Tests for add_rollings_h function."""

    def test_add_rollings_basic(self, sample_merged_data):
        """Test adding rolling window features."""
        df = add_rollings_h(sample_merged_data, cols=["pv"], windows=[3, 6])

        assert "pv_roll3h" in df.columns
        assert "pv_roll6h" in df.columns

    def test_rolling_calculation(self, sample_merged_data):
        """Test that rolling means are calculated correctly."""
        df = add_rollings_h(sample_merged_data, cols=["pv"], windows=[3])

        # Check rolling mean calculation (after enough data)
        idx = 5
        expected_mean = sample_merged_data["pv"].iloc[idx - 2 : idx + 1].mean()
        assert df["pv_roll3h"].iloc[idx] == pytest.approx(expected_mean, abs=1e-6)

    def test_rolling_with_nan(self, sample_merged_data):
        """Test rolling with NaN values at the start."""
        df = add_rollings_h(sample_merged_data, cols=["pv"], windows=[3])

        # First value should not be NaN (min_periods allows partial windows)
        assert not pd.isna(df["pv_roll3h"].iloc[0])


class TestStandardizeColumns:
    """Tests for standardize_feature_columns function."""

    def test_standardize_lowercase(self):
        """Test that columns are standardized to lowercase."""
        df = pd.DataFrame({"GHI": [1, 2], "DNI": [3, 4], "DHI": [5, 6]})
        result = standardize_feature_columns(df)

        assert "ghi" in result.columns
        assert "dni" in result.columns
        assert "dhi" in result.columns

    def test_standardize_synonyms(self):
        """Test that synonyms are mapped correctly."""
        df = pd.DataFrame({"Temperature": [20, 25], "cloud_cover": [50, 60]})
        result = standardize_feature_columns(df)

        assert "temp" in result.columns
        assert "clouds" in result.columns

    def test_preserve_values(self):
        """Test that values are preserved during standardization."""
        df = pd.DataFrame({"GHI": [100, 200], "Temperature": [20, 25]})
        result = standardize_feature_columns(df)

        np.testing.assert_array_equal(result["ghi"].values, [100, 200])
        np.testing.assert_array_equal(result["temp"].values, [20, 25])


class TestSolarPosition:
    """Tests for add_solar_position function."""

    def test_solar_position_adds_columns(self, sample_merged_data):
        """Test that solar position columns are added."""
        try:
            df = add_solar_position(sample_merged_data, lat=-33.86, lon=151.21)
            assert "sp_zenith" in df.columns
            assert "sp_azimuth" in df.columns
        except ImportError:
            pytest.skip("pvlib not available")

    def test_solar_position_bounds(self, sample_merged_data):
        """Test that solar position values are within valid ranges."""
        try:
            df = add_solar_position(sample_merged_data, lat=-33.86, lon=151.21)
            assert df["sp_zenith"].between(0, 180).all()
            assert df["sp_azimuth"].between(0, 360).all()
        except ImportError:
            pytest.skip("pvlib not available")

    def test_solar_position_requires_timezone(self):
        """Test that solar position requires timezone-aware index."""
        df = pd.DataFrame({"pv": [1, 2, 3]}, index=pd.date_range("2010-01-01", periods=3, freq="H"))
        with pytest.raises(ValueError, match="timezone-aware"):
            add_solar_position(df, lat=-33.86, lon=151.21)


class TestClearsky:
    """Tests for add_clearsky function."""

    def test_clearsky_adds_columns(self, sample_merged_data):
        """Test that clearsky columns are added."""
        try:
            df = add_clearsky(sample_merged_data, lat=-33.86, lon=151.21)
            assert "cs_ghi" in df.columns
            assert "cs_dni" in df.columns
            assert "cs_dhi" in df.columns
        except ImportError:
            pytest.skip("pvlib not available")

    def test_clearsky_positive_values(self, sample_merged_data):
        """Test that clearsky irradiance values are non-negative."""
        try:
            df = add_clearsky(sample_merged_data, lat=-33.86, lon=151.21)
            assert (df["cs_ghi"] >= 0).all()
            assert (df["cs_dni"] >= 0).all()
            assert (df["cs_dhi"] >= 0).all()
        except ImportError:
            pytest.skip("pvlib not available")


class TestClearnessIndex:
    """Tests for add_kc function."""

    def test_kc_calculation(self, sample_merged_data):
        """Test clearness index calculation."""
        try:
            # First add clearsky
            df = add_clearsky(sample_merged_data, lat=-33.86, lon=151.21)
            df = add_kc(df, ghi_col="ghi")

            assert "kc" in df.columns

            # kc should be between 0 and ~1.5 (can exceed 1 due to reflections)
            assert df["kc"].min() >= 0
            assert df["kc"].max() < 5  # Reasonable upper bound
        except ImportError:
            pytest.skip("pvlib not available")

    def test_kc_with_zero_clearsky(self):
        """Test kc behavior when clearsky is zero (night time)."""
        dates = pd.date_range("2010-01-01", periods=10, freq="H", tz="UTC")
        df = pd.DataFrame({"ghi": [100] * 10, "cs_ghi": [0] * 10}, index=dates)

        result = add_kc(df, ghi_col="ghi")

        # When clearsky is zero (night), kc should be NaN (undefined)
        assert "kc" in result.columns
        assert result["kc"].isna().all()  # All NaN at night is correct behavior
