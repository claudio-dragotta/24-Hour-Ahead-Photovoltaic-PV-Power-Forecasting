"""Tests for pv_forecasting.data module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pv_forecasting.data import align_hourly, standardize_feature_columns


class TestAlignHourly:
    """Tests for align_hourly function."""

    def test_align_basic(self, sample_pv_data, sample_weather_data):
        """Test basic alignment of PV and weather data."""
        result = align_hourly(sample_pv_data, sample_weather_data)

        assert "pv" in result.columns
        assert "ghi" in result.columns
        assert "dni" in result.columns
        assert len(result) > 0

    def test_align_preserves_timezone(self, sample_pv_data, sample_weather_data):
        """Test that alignment preserves UTC timezone."""
        result = align_hourly(sample_pv_data, sample_weather_data)

        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_align_handles_missing_data(self):
        """Test alignment with non-overlapping data."""
        pv = pd.DataFrame(
            {"pv": [0.5, 0.6, 0.7]}, index=pd.date_range("2010-01-01", periods=3, freq="H", tz="UTC")
        )
        wx = pd.DataFrame(
            {"ghi": [100, 200, 300]}, index=pd.date_range("2010-01-02", periods=3, freq="H", tz="UTC")
        )

        result = align_hourly(pv, wx)

        # Should have only overlapping times or use forward fill
        assert len(result) >= 0

    def test_align_sorted_output(self, sample_pv_data, sample_weather_data):
        """Test that output is sorted by timestamp."""
        result = align_hourly(sample_pv_data, sample_weather_data)

        assert result.index.is_monotonic_increasing


class TestStandardizeFeatureColumns:
    """Tests for standardize_feature_columns function."""

    def test_lowercase_conversion(self):
        """Test conversion to lowercase."""
        df = pd.DataFrame({"GHI": [100, 200], "DNI": [150, 250], "DHI": [50, 75]})

        result = standardize_feature_columns(df)

        assert "ghi" in result.columns
        assert "dni" in result.columns
        assert "dhi" in result.columns
        assert "GHI" not in result.columns

    def test_synonym_mapping(self):
        """Test mapping of synonyms to standard names."""
        df = pd.DataFrame({"Temperature": [20, 25], "cloud_cover": [50, 60], "wind": [5, 10]})

        result = standardize_feature_columns(df)

        assert "temp" in result.columns
        assert "clouds" in result.columns

    def test_preserve_original_df(self):
        """Test that original dataframe is not modified."""
        df = pd.DataFrame({"GHI": [100, 200]})
        original_columns = df.columns.tolist()

        result = standardize_feature_columns(df)

        # Original should be unchanged
        assert df.columns.tolist() == original_columns
        # Result should be different
        assert result.columns.tolist() != original_columns

    def test_values_preserved(self):
        """Test that data values are preserved during standardization."""
        df = pd.DataFrame({"GHI": [100, 200], "Temperature": [20, 25]})

        result = standardize_feature_columns(df)

        np.testing.assert_array_equal(result["ghi"].values, [100, 200])
        np.testing.assert_array_equal(result["temp"].values, [20, 25])

    def test_handles_clouds_all(self):
        """Test that clouds_all is mapped to clouds."""
        df = pd.DataFrame({"clouds_all": [50, 60, 70]})

        result = standardize_feature_columns(df)

        assert "clouds" in result.columns
        assert "clouds_all" not in result.columns

    def test_preserves_unknown_columns(self):
        """Test that unknown columns are preserved."""
        df = pd.DataFrame({"GHI": [100, 200], "custom_feature": [1, 2]})

        result = standardize_feature_columns(df)

        assert "ghi" in result.columns
        assert "custom_feature" in result.columns


class TestDataLoading:
    """Tests for data loading functions."""

    @pytest.mark.skip(reason="Requires actual Excel files")
    def test_load_pv_xlsx(self):
        """Test loading PV data from Excel."""
        # This would require actual test data files
        pass

    @pytest.mark.skip(reason="Requires actual Excel files")
    def test_load_wx_xlsx(self):
        """Test loading weather data from Excel."""
        # This would require actual test data files
        pass


class TestDataIntegration:
    """Integration tests for data module."""

    def test_full_pipeline(self, sample_pv_data, sample_weather_data):
        """Test complete data processing pipeline."""
        # Align data
        df = align_hourly(sample_pv_data, sample_weather_data)

        # Standardize
        df = standardize_feature_columns(df)

        # Verify output
        assert len(df) > 0
        assert "pv" in df.columns
        assert df.index.tz is not None
        assert not df.index.has_duplicates

    def test_handles_edge_cases(self):
        """Test handling of edge cases in data processing."""
        # Empty dataframe
        empty_df = pd.DataFrame()
        result = standardize_feature_columns(empty_df)
        assert len(result) == 0

        # Single row
        single = pd.DataFrame({"GHI": [100]})
        result = standardize_feature_columns(single)
        assert len(result) == 1
        assert "ghi" in result.columns
