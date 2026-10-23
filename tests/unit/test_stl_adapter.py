"""Tests for the STL decomposition baseline adapter."""

import numpy as np
import pandas as pd
import pytest

from forecasting.adapters.stl import STLAdapter
from forecasting.contracts import ForecastRequest


def _request(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    horizon: int = 5,
) -> ForecastRequest:
    return ForecastRequest(
        target_history=values,
        history_index=dates,
        target_names=tuple(f"target_{i}" for i in range(values.shape[0])),
        horizon=horizon,
        future_index=pd.date_range(
            dates[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        ),
    )


def test_stl_forecasts_trend_and_repeated_season_for_each_target():
    steps = np.arange(84, dtype=float)
    seasonal = np.sin(2 * np.pi * steps / 7)
    history = np.vstack(
        (
            0.4 * steps + seasonal,
            -0.2 * steps + 2 * seasonal,
        )
    )
    request = _request(
        history,
        pd.date_range("2024-01-01", periods=84, freq="D"),
        horizon=14,
    )

    result = STLAdapter()._predict_impl(request, None)

    assert result.point.shape == (2, 14)
    assert result.quantiles is None
    assert result.context_used == 84
    assert result.metadata == {"baseline_type": "stl", "seasonal_period": 7}
    assert np.all(np.isfinite(result.point))
    assert result.point[0, 7:].mean() > result.point[0, :7].mean()
    assert result.point[1, 7:].mean() < result.point[1, :7].mean()


def test_stl_reduces_seasonal_period_to_fit_two_cycles():
    steps = np.arange(8, dtype=float)
    history = (steps + np.sin(2 * np.pi * steps / 4)).reshape(1, -1)
    request = _request(history, pd.date_range("2024-01-01", periods=8, freq="D"))

    result = STLAdapter()._predict_impl(request, None)

    assert result.metadata["seasonal_period"] == 4
    assert any("reduced from 7 to 4" in warning for warning in result.warnings)


def test_stl_rejects_non_finite_history():
    history = np.arange(16, dtype=float).reshape(1, -1)
    history[0, 4] = np.nan
    request = _request(history, pd.date_range("2024-01-01", periods=16, freq="D"))

    with pytest.raises(ValueError, match="finite target history"):
        STLAdapter()._predict_impl(request, None)
