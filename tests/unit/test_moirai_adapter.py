"""Unit tests for the Moirai adapter output mapping."""

import sys
from types import ModuleType

import numpy as np
import pandas as pd

from forecasting.adapters.moirai import MoiraiAdapter
from forecasting.contracts import ForecastRequest, RuntimeConfig


def test_moirai_maps_forecast_quantiles_and_reuses_predictor(monkeypatch):
    request = ForecastRequest(
        target_history=np.arange(12, dtype=float).reshape(1, -1),
        history_index=pd.date_range("2024-01-01", periods=12, freq="D"),
        target_names=("target",),
        horizon=3,
        future_index=pd.date_range("2024-01-13", periods=3, freq="D"),
    )
    levels = request.quantile_levels
    created_predictors = []

    class ListDataset(list):
        def __init__(self, entries, freq):
            super().__init__(entries)
            self.freq = freq

    class Forecast:
        def quantile(self, level):
            return np.full(3, level)

    class Predictor:
        def predict(self, dataset):
            assert dataset.freq == "D"
            assert dataset[0]["target"].shape == (12,)
            return iter([Forecast()])

    class MoiraiForecast:
        def __init__(self, **kwargs):
            assert kwargs["prediction_length"] == 3
            assert kwargs["context_length"] == 12

        def create_predictor(self, **kwargs):
            assert kwargs == {"batch_size": 1, "device": "cpu"}
            predictor = Predictor()
            created_predictors.append(predictor)
            return predictor

    gluonts_common = ModuleType("gluonts.dataset.common")
    gluonts_common.ListDataset = ListDataset
    uni2ts_moirai = ModuleType("uni2ts.model.moirai")
    uni2ts_moirai.MoiraiForecast = MoiraiForecast
    monkeypatch.setitem(sys.modules, "gluonts.dataset.common", gluonts_common)
    monkeypatch.setitem(sys.modules, "uni2ts.model.moirai", uni2ts_moirai)

    adapter = MoiraiAdapter()
    adapter._runtime_config = RuntimeConfig(device="cpu")

    result = adapter._predict_impl(request, object())
    second_result = adapter._predict_impl(request, object())

    assert result.point.shape == (1, 3)
    assert result.quantiles is not None
    assert result.quantiles.shape == (1, 3, len(levels))
    np.testing.assert_allclose(result.point, 0.5)
    np.testing.assert_allclose(result.quantiles[0, 0], levels)
    assert result.context_used == 12
    assert len(created_predictors) == 1
    np.testing.assert_array_equal(second_result.point, result.point)
