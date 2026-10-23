from types import ModuleType

import numpy as np
import pandas as pd

from forecasting.adapters.tirex2 import TiRex2Adapter
from forecasting.contracts import ForecastRequest


def _request() -> ForecastRequest:
    history = pd.date_range("2024-01-01", periods=12, freq="D")
    return ForecastRequest(
        target_history=np.vstack((np.arange(12), np.arange(12) * 2)),
        history_index=history,
        target_names=("first", "second"),
        horizon=3,
        future_index=pd.date_range("2024-01-13", periods=3, freq="D"),
    )


def test_tirex2_converts_quantile_axis_to_common_contract(monkeypatch):
    request = _request()
    module = ModuleType("tirex2")

    class TimeseriesType:
        def __init__(self, **kwargs):
            self.values = kwargs

    module.TimeseriesType = TimeseriesType
    monkeypatch.setitem(__import__("sys").modules, "tirex2", module)

    class Model:
        def forecast(self, timeseries, prediction_length, output_type):
            assert timeseries[0].values["target"].shape == (2, 12)
            assert prediction_length == 3
            assert output_type == "numpy"
            return [np.tile(np.arange(9, dtype=float)[None, :, None], (2, 1, 3))]

    result = TiRex2Adapter()._predict_impl(request, Model())

    assert result.point.shape == (2, 3)
    assert result.quantiles is not None
    assert result.quantiles.shape == (2, 3, 9)
    np.testing.assert_allclose(result.point, 4.0)


def test_tirex2_passes_historical_and_future_known_covariates(monkeypatch):
    request = _request()
    known_history = np.arange(12, dtype=float).reshape(1, -1)
    known_future = np.array([[12.0, 13.0, 14.0]])
    request = ForecastRequest(
        target_history=request.target_history[:1],
        history_index=request.history_index,
        target_names=("first",),
        horizon=request.horizon,
        future_index=request.future_index,
        known_covariates_history=known_history,
        known_covariates_future=known_future,
    )
    module = ModuleType("tirex2")

    class TimeseriesType:
        def __init__(self, **kwargs):
            self.values = kwargs

    module.TimeseriesType = TimeseriesType
    monkeypatch.setitem(__import__("sys").modules, "tirex2", module)

    class Model:
        def forecast(self, timeseries, prediction_length, output_type):
            covariates = timeseries[0].values["future_covariates"]
            np.testing.assert_array_equal(
                covariates.cpu().numpy(),
                np.concatenate((known_history, known_future), axis=1),
            )
            return [np.zeros((1, 9, prediction_length))]

    TiRex2Adapter()._predict_impl(request, Model())
