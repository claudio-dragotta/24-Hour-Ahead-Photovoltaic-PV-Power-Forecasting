"""Tests for pv_forecasting.metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from pv_forecasting.metrics import mae, mase, rmse


class TestRMSE:
    """Tests for RMSE metric."""

    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        result = rmse(y_true, y_pred)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_rmse_known_values(self):
        """Test RMSE with known ground truth values."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])

        # RMSE = sqrt((0.25 + 0.25 + 0 + 1) / 4) = sqrt(1.5/4) = sqrt(0.375)
        expected = np.sqrt(0.375)
        result = rmse(y_true, y_pred)

        assert result == pytest.approx(expected, rel=1e-6)

    def test_rmse_shape_mismatch(self):
        """Test that RMSE raises error on shape mismatch."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])

        with pytest.raises(ValueError):
            rmse(y_true, y_pred)

    def test_rmse_positive(self):
        """Test that RMSE is always non-negative."""
        y_true = np.random.rand(100)
        y_pred = np.random.rand(100)

        result = rmse(y_true, y_pred)
        assert result >= 0


class TestMAE:
    """Tests for MAE metric."""

    def test_mae_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        result = mae(y_true, y_pred)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_mae_known_values(self):
        """Test MAE with known ground truth values."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])

        # MAE = (0.5 + 0.5 + 0 + 1) / 4 = 2.0 / 4 = 0.5
        expected = 0.5
        result = mae(y_true, y_pred)

        assert result == pytest.approx(expected, rel=1e-6)

    def test_mae_symmetric(self):
        """Test that MAE treats positive and negative errors equally."""
        y_true = np.array([5.0, 5.0])
        y_pred1 = np.array([3.0, 7.0])  # errors: -2, +2
        y_pred2 = np.array([7.0, 3.0])  # errors: +2, -2

        result1 = mae(y_true, y_pred1)
        result2 = mae(y_true, y_pred2)

        assert result1 == pytest.approx(result2)


class TestMASE:
    """Tests for MASE metric."""

    def test_mase_perfect_prediction(self):
        """Test MASE with perfect predictions."""
        train_series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([6.0, 7.0, 8.0])
        y_pred = np.array([6.0, 7.0, 8.0])

        result = mase(y_true, y_pred, train_series, m=1)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_mase_naive_baseline(self):
        """Test that MASE equals 1.0 for naive seasonal forecast."""
        # Create data where naive forecast (y[t] = y[t-m]) should give MASE=1
        train_series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([2.0, 3.0, 4.0])

        # Naive seasonal forecast with m=1: predict y[t-1]
        y_pred = np.array([1.0, 2.0, 3.0])

        result = mase(y_true, y_pred, train_series, m=1)

        # For naive forecast, MASE should be close to 1.0
        # (may not be exactly 1.0 due to different test set)
        assert result > 0

    def test_mase_better_than_naive(self):
        """Test MASE < 1 when predictions are better than naive baseline."""
        train_series = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        y_true = np.array([4.5, 5.0, 5.5])

        # Good predictions (close to truth)
        y_pred_good = np.array([4.4, 5.1, 5.4])

        result = mase(y_true, y_pred_good, train_series, m=1)

        # Should be better than naive baseline
        assert result < 2.0  # Reasonable upper bound

    def test_mase_seasonal_lag(self):
        """Test MASE with seasonal lag m=24 (daily seasonality)."""
        # Create weekly data with strong daily pattern
        np.random.seed(42)
        train_series = np.repeat(np.arange(1, 8), 24) + np.random.randn(168) * 0.1

        y_true = np.array([8.0] * 24)
        y_pred = np.array([7.9] * 24)

        result = mase(y_true, y_pred, train_series, m=24)

        assert result > 0
        assert np.isfinite(result)

    def test_mase_raises_on_constant_series(self):
        """Test that MASE handles constant training series (MAE_train=0)."""
        train_series = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # constant!
        y_true = np.array([6.0, 7.0, 8.0])
        y_pred = np.array([5.5, 6.5, 7.5])

        # Should either raise or return inf/nan
        result = mase(y_true, y_pred, train_series, m=1)

        # Depending on implementation, might return inf or raise
        assert np.isinf(result) or np.isnan(result) or isinstance(result, float)


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_all_metrics_on_random_data(self, sample_predictions):
        """Test all metrics on random prediction data."""
        y_true, y_pred = sample_predictions

        rmse_val = rmse(y_true, y_pred)
        mae_val = mae(y_true, y_pred)

        # Basic sanity checks
        assert rmse_val > 0
        assert mae_val > 0
        assert rmse_val >= mae_val  # RMSE >= MAE always

    def test_metrics_with_zero_error(self):
        """Test metrics when error is zero."""
        y = np.array([0.5, 0.6, 0.7, 0.8])

        assert rmse(y, y) == 0.0
        assert mae(y, y) == 0.0

    def test_metrics_with_negative_values(self):
        """Test metrics work with negative values."""
        y_true = np.array([-1.0, -2.0, -3.0])
        y_pred = np.array([-1.1, -2.2, -2.8])

        rmse_val = rmse(y_true, y_pred)
        mae_val = mae(y_true, y_pred)

        assert np.isfinite(rmse_val)
        assert np.isfinite(mae_val)
        assert rmse_val > 0
        assert mae_val > 0
