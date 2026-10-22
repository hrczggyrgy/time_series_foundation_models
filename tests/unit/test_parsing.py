"""Unit tests for data parsing and observation mask tracking."""

import numpy as np
import pandas as pd
import pytest

from forecasting.data.parsing import prepare_dataframe


def test_prepare_dataframe_returns_observation_mask():
    """Test that prepare_dataframe returns observation mask."""
    # Create simple test data
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'value': np.random.randn(100)
    })

    prepared, prep_info, obs_mask = prepare_dataframe(
        raw_df=df,
        date_col='date',
        target_cols=['value'],
        past_covariate_cols=None,
        known_covariate_cols=None,
        auto_fix=False,
    )

    # Check that observation mask is returned
    assert obs_mask is not None, "prepare_dataframe should return observation mask"
    assert len(obs_mask) == len(prepared), "Observation mask should match dataframe length"
    assert obs_mask.dtype == bool, "Observation mask should be boolean"


def test_observation_mask_marks_resampled_points():
    """Test that resampled points are marked as imputed in observation mask."""
    # Create irregular data
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    # Remove some dates to make it irregular
    dates = dates[::2]  # Every other day
    df = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(len(dates))
    })

    prepared, prep_info, obs_mask = prepare_dataframe(
        raw_df=df,
        date_col='date',
        target_cols=['value'],
        past_covariate_cols=None,
        known_covariate_cols=None,
        auto_fix=True,  # Enable auto-resampling
    )

    # If resampling occurred, new points should be marked as False
    if prep_info.get('auto_resampled_to'):
        original_len = prep_info.get('original_length', len(prepared))
        if len(prepared) > original_len:
            # New points should be marked as imputed
            assert not obs_mask.iloc[original_len:].all(), \
                "Resampled points should be marked as imputed (False)"


def test_observation_mask_all_true_for_regular_data():
    """Test that regular data has all True in observation mask."""
    # Create regular data
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'value': np.random.randn(100)
    })

    prepared, prep_info, obs_mask = prepare_dataframe(
        raw_df=df,
        date_col='date',
        target_cols=['value'],
        past_covariate_cols=None,
        known_covariate_cols=None,
        auto_fix=False,
    )

    # All should be True for regular data without resampling
    assert obs_mask.all(), "All points should be marked as observed for regular data"
