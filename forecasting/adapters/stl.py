"""STL decomposition baseline for univariate and independent multivariate series."""

from __future__ import annotations

from typing import Any

import numpy as np

from forecasting.adapters.base import BaseAdapter
from forecasting.contracts import (
    ForecastRequest,
    ForecastResult,
    ModelCapabilities,
    RuntimeConfig,
)


class STLAdapter(BaseAdapter):
    """Forecast by extrapolating the STL trend and repeating its final season."""

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_key="stl",
            display_name="STL Decomposition",
            checkpoint_id="",
            checkpoint_revision="",
            supports_cpu=True,
            supports_cuda=False,
            supports_mps=False,
            supports_multivariate=True,
            supports_past_covariates=False,
            supports_known_future_covariates=False,
            native_quantile_levels=(),
            max_context_length=None,
            min_context_length=4,
            max_horizon=None,
            compiles_context_horizon=False,
            supports_batch_inference=True,
            license_notice="No model weights — statsmodels STL decomposition baseline.",
        )

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        return None

    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        from statsmodels.tsa.seasonal import STL

        history = self._ensure_2d_targets(request.target_history).astype(float)
        period, warnings = self._seasonal_period(request)
        point = np.empty((request.n_targets, request.horizon), dtype=float)
        trend_window = min(history.shape[1], max(period * 2, period + 2))
        future_steps = np.arange(1, request.horizon + 1, dtype=float)

        for target_idx, series in enumerate(history):
            if not np.all(np.isfinite(series)):
                raise ValueError("STL baseline requires finite target history values.")
            decomposition = STL(
                series,
                period=period,
                robust=True,
            ).fit()
            recent_trend = decomposition.trend[-trend_window:]
            trend_x = np.arange(trend_window, dtype=float)
            slope, intercept = np.polyfit(trend_x, recent_trend, 1)
            extrapolated_trend = intercept + slope * (trend_window - 1 + future_steps)
            seasonal_pattern = decomposition.seasonal[-period:]
            seasonal_forecast = np.resize(seasonal_pattern, request.horizon)
            point[target_idx] = extrapolated_trend + seasonal_forecast

        warnings.append(
            "STL baseline returns deterministic point forecasts without prediction intervals."
        )
        return self._build_result(
            point=point,
            quantiles=None,
            quantile_levels=np.asarray(request.quantile_levels, dtype=float),
            request=request,
            point_definition="extrapolated_stl_trend_plus_repeated_season",
            context_used=request.n_history,
            warnings=warnings,
            metadata={"baseline_type": "stl", "seasonal_period": period},
        )

    @staticmethod
    def _seasonal_period(request: ForecastRequest) -> tuple[int, list[str]]:
        frequency = request.history_index.inferred_freq
        if frequency is None:
            frequency = request.history_index.freqstr
        frequency = (frequency or "").upper()
        warnings = []
        if frequency.startswith(("T", "MIN")):
            requested_period = 60
        elif frequency.startswith("H"):
            requested_period = 24
        elif frequency.startswith(("D", "B")):
            requested_period = 7
        elif frequency.startswith("W"):
            requested_period = 52
        elif frequency.startswith("M"):
            requested_period = 12
        elif frequency.startswith("Q"):
            requested_period = 4
        else:
            requested_period = 7
            warnings.append("Could not infer frequency; STL uses a weekly seasonal period.")

        period = min(requested_period, max(2, request.n_history // 2))
        if period != requested_period:
            warnings.append(
                f"Seasonal period reduced from {requested_period} to {period} "
                "to fit at least two cycles in the available history."
            )
        return period, warnings
