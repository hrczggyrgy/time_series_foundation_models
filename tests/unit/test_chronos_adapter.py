import numpy as np
import pandas as pd
import torch

from forecasting.adapters.chronos import ChronosAdapter
from forecasting.contracts import ForecastRequest


def test_chronos_uses_samples_for_quantiles():
    history = pd.date_range("2024-01-01", periods=12, freq="D")
    request = ForecastRequest(
        target_history=np.vstack((np.arange(12), np.arange(12) * 2)),
        history_index=history,
        target_names=("first", "second"),
        horizon=3,
        future_index=pd.date_range("2024-01-13", periods=3, freq="D"),
        quantile_levels=(0.1, 0.5, 0.9),
    )

    class Model:
        def predict(self, contexts, prediction_length, num_samples):
            assert tuple(contexts.shape) == (2, 12)
            assert prediction_length == 3
            assert num_samples == 100
            return torch.arange(100, dtype=torch.float32).reshape(1, 100, 1).expand(2, 100, 3)

    result = ChronosAdapter()._predict_impl(request, Model())

    assert result.point.shape == (2, 3)
    assert result.quantiles is not None
    assert result.quantiles.shape == (2, 3, 3)
    assert np.all(np.diff(result.quantiles, axis=-1) >= 0)
    assert result.point[0, 0] == np.quantile(np.arange(100), 0.5)
