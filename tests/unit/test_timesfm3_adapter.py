from types import SimpleNamespace

import numpy as np
import pandas as pd

from forecasting.adapters.timesfm3 import TimesFM3Adapter
from forecasting.contracts import ForecastRequest


def test_timesfm3_maps_native_forecast_to_common_contract():
    history = pd.date_range("2024-01-01", periods=12, freq="D")
    request = ForecastRequest(
        target_history=np.vstack((np.arange(12), np.arange(12) * 2)),
        history_index=history,
        target_names=("first", "second"),
        horizon=3,
        future_index=pd.date_range("2024-01-13", periods=3, freq="D"),
    )
    levels = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))

    class Model:
        config = SimpleNamespace(quantiles=levels)

        def predict(self, context, horizon, **kwargs):
            assert context.shape == (2, 12)
            assert horizon == 3
            assert kwargs["return_quantiles"] is True
            quantiles = np.broadcast_to(levels, (2, 3, len(levels))).copy()
            point = np.full((2, 3), 0.5)
            return SimpleNamespace(forecast=point, quantiles=quantiles)

    result = TimesFM3Adapter()._predict_impl(request, Model())

    assert result.point.shape == (2, 3)
    assert result.quantiles is not None
    assert result.quantiles.shape == (2, 3, 9)
    np.testing.assert_array_equal(result.quantile_levels, levels)
