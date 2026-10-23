"""Unit tests for model comparison orchestration."""

import numpy as np
import pandas as pd
import pytest

from forecasting.comparison import ComparisonReport, ModelComparisonItem
from forecasting.service import ForecastingService


def test_comparison_report_structure():
    """Test that ComparisonReport has required fields."""
    report = ComparisonReport(
        comparison_id="test123",
        model_results={},
        horizon=10,
        evaluation_origins=[pd.Timestamp("2020-01-01")],
        settings={"n_windows": 3},
        data_fingerprint="abc123",
        timestamp="2024-01-01T00:00:00",
    )

    assert report.comparison_id == "test123"
    assert report.horizon == 10
    assert len(report.evaluation_origins) == 1
    assert report.data_fingerprint == "abc123"
    assert isinstance(report.model_results, dict)


def test_model_comparison_item_structure():
    """Test that ModelComparisonItem has required fields."""
    item = ModelComparisonItem(
        model_key="test_model",
        display_name="Test Model",
        forecast=None,
        backtest=None,
        error="Test error",
        elapsed_seconds=1.5,
        context_used=None,
        warnings=["warning1"],
    )

    assert item.model_key == "test_model"
    assert item.display_name == "Test Model"
    assert item.error == "Test error"
    assert item.elapsed_seconds == 1.5
    assert item.context_used is None
    assert item.warnings == ["warning1"]


def test_comparison_service_compare_models_requires_models():
    """Test that compare_models raises error with empty model list."""
    service = ForecastingService()

    # Create minimal parsed data
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
        'y': np.random.randn(100)
    })

    from forecasting.data.parsing import ParsedData
    parsed_data = ParsedData(
        df=df,
        date_column='ds',
        target_columns=['y'],
        past_covariate_columns=[],
        known_covariate_columns=[],
        prep_report={},
        raw_signature="abc123",
        observation_mask=pd.Series([True] * 100),
    )

    with pytest.raises(ValueError, match="At least one model must be selected"):
        service.compare_models(
            model_keys=[],
            parsed_data=parsed_data,
            horizon=10,
        )


def test_comparison_service_compare_models_insufficient_data():
    """Test that compare_models raises error with insufficient data."""
    service = ForecastingService()

    # Create minimal parsed data with insufficient data
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=30, freq='D'),
        'y': np.random.randn(30)
    })

    from forecasting.data.parsing import ParsedData
    parsed_data = ParsedData(
        df=df,
        date_column='ds',
        target_columns=['y'],
        past_covariate_columns=[],
        known_covariate_columns=[],
        prep_report={},
        raw_signature="abc123",
        observation_mask=pd.Series([True] * 30),
    )

    with pytest.raises(ValueError, match="Insufficient data for comparison"):
        service.compare_models(
            model_keys=["stl"],  # Use STL, which does not require model weights
            parsed_data=parsed_data,
            horizon=10,
        )


def test_comparison_service_progress_callback():
    """Test that progress callback is called during comparison."""
    service = ForecastingService()
    progress_calls = []

    def progress_callback(model_key, status):
        progress_calls.append((model_key, status))

    # Create sufficient parsed data
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
        'y': np.random.randn(100)
    })

    from forecasting.data.parsing import ParsedData
    parsed_data = ParsedData(
        df=df,
        date_column='ds',
        target_columns=['y'],
        past_covariate_columns=[],
        known_covariate_columns=[],
        prep_report={},
        raw_signature="abc123",
        observation_mask=pd.Series([True] * 100),
    )

    try:
        service.compare_models(
            model_keys=["stl"],
            parsed_data=parsed_data,
            horizon=10,
            progress_callback=progress_callback,
        )
    except Exception:
        # May fail if model not available, but callback should still be called
        pass

    # Callback should have been called at least once
    assert len(progress_calls) > 0
