"""Tests for pv_forecasting.pipeline module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pv_forecasting.pipeline import DEFAULT_LAG_HOURS, DEFAULT_ROLLING_HOURS, load_and_engineer_features


class TestPipelineConstants:
    """Tests for pipeline constants."""

    def test_default_lag_hours(self):
        """Test default lag hours are defined."""
        assert len(DEFAULT_LAG_HOURS) > 0
        assert 1 in DEFAULT_LAG_HOURS
        assert 24 in DEFAULT_LAG_HOURS

    def test_default_rolling_hours(self):
        """Test default rolling hours are defined."""
        assert len(DEFAULT_ROLLING_HOURS) > 0
        assert all(w > 0 for w in DEFAULT_ROLLING_HOURS)


class TestLoadAndEngineerFeatures:
    """Tests for load_and_engineer_features function."""

    @pytest.mark.skip(reason="Requires actual data files")
    def test_basic_feature_engineering(self, tmp_path):
        """Test basic feature engineering pipeline."""
        # Would need actual test data files
        pass

    @pytest.mark.skip(reason="Requires actual data files")
    def test_with_custom_lags(self, tmp_path):
        """Test feature engineering with custom lag hours."""
        pass

    @pytest.mark.skip(reason="Requires actual data files")
    def test_with_custom_rolling_windows(self, tmp_path):
        """Test feature engineering with custom rolling windows."""
        pass

    @pytest.mark.skip(reason="Requires actual data files")
    def test_solar_features_optional(self, tmp_path):
        """Test that solar features can be disabled."""
        pass

    @pytest.mark.skip(reason="Requires actual data files")
    def test_clearsky_features_optional(self, tmp_path):
        """Test that clearsky features can be disabled."""
        pass


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering."""

    def test_feature_names_convention(self, sample_training_data):
        """Test that engineered features follow naming convention."""
        df = sample_training_data

        # Check for cyclical features
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        assert "doy_sin" in df.columns
        assert "doy_cos" in df.columns

        # Check for time_idx and series_id
        assert "time_idx" in df.columns
        assert "series_id" in df.columns

    def test_no_duplicate_columns(self, sample_training_data):
        """Test that there are no duplicate column names."""
        df = sample_training_data

        assert len(df.columns) == len(set(df.columns))

    def test_numeric_features(self, sample_training_data):
        """Test that all features (except series_id) are numeric."""
        df = sample_training_data

        for col in df.columns:
            if col != "series_id":
                assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric"

    def test_time_idx_sequential(self, sample_training_data):
        """Test that time_idx is sequential."""
        df = sample_training_data

        expected = list(range(len(df)))
        actual = df["time_idx"].tolist()

        assert actual == expected

    def test_series_id_constant(self, sample_training_data):
        """Test that series_id is constant."""
        df = sample_training_data

        assert df["series_id"].nunique() == 1
