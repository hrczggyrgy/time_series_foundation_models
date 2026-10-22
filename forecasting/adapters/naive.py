"""Naive baseline adapters — zero dependencies, deterministic forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecasting.adapters.base import BaseAdapter
from forecasting.contracts import (
    ForecastRequest,
    ForecastResult,
    ModelCapabilities,
    RuntimeConfig,
)


class NaiveAdapter(BaseAdapter):
    """Naive forecast: repeat last observed value."""

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_key="naive",
            display_name="Naive (Last Value)",
            checkpoint_id="",
            checkpoint_revision="",
            supports_cpu=True,
            supports_cuda=False,
            supports_mps=False,
            supports_multivariate=False,
            supports_past_covariates=False,
            supports_known_future_covariates=False,
            native_quantile_levels=(),
            max_context_length=None,
            min_context_length=1,
            max_horizon=None,
            compiles_context_horizon=False,
            supports_batch_inference=True,
            license_policy="permissive",
            license_notice="No model weights — algorithmic baseline.",
        )

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        # No model to load
        return None

    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        n_targets = request.n_targets
        horizon = request.horizon

        # Get last value for each target
        target_history = self._ensure_2d_targets(request.target_history)  # [targets, history]
        last_values = target_history[:, -1]  # [targets]

        # Repeat for horizon
        point = np.tile(last_values[:, np.newaxis], (1, horizon))  # [targets, horizon]

        # No probabilistic output for naive baseline
        quantiles = None
        quantile_levels = np.array(request.quantile_levels)

        warnings = ["Naive baseline produces deterministic forecasts only (no prediction intervals)."]

        return self._build_result(
            point=point,
            quantiles=quantiles,
            quantile_levels=quantile_levels,
            request=request,
            point_definition="last_observed_value",
            context_used=1,
            warnings=warnings,
            metadata={"baseline_type": "naive"},
        )


class SeasonalNaiveAdapter(BaseAdapter):
    """Seasonal naive forecast: repeat value from same seasonal period."""

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_key="seasonal_naive",
            display_name="Seasonal Naive",
            checkpoint_id="",
            checkpoint_revision="",
            supports_cpu=True,
            supports_cuda=False,
            supports_mps=False,
            supports_multivariate=False,
            supports_past_covariates=False,
            supports_known_future_covariates=False,
            native_quantile_levels=(),
            max_context_length=None,
            min_context_length=2,
            max_horizon=None,
            compiles_context_horizon=False,
            supports_batch_inference=True,
            license_policy="permissive",
            license_notice="No model weights — algorithmic baseline.",
        )

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        return None

    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        n_targets = request.n_targets
        horizon = request.horizon

        target_history = self._ensure_2d_targets(request.target_history)  # [targets, history]
        history_index = request.history_index

        # Infer seasonal period from frequency
        seasonal_period = self._infer_seasonal_period(history_index)

        if seasonal_period is None or seasonal_period >= len(history_index):
            warnings = [
                "Could not infer seasonal period; falling back to naive (last value)."
            ]
            # Fall back to naive
            last_values = target_history[:, -1]
            point = np.tile(last_values[:, np.newaxis], (1, horizon))
        else:
            warnings = [
                f"Using seasonal period of {seasonal_period} steps "
                f"(inferred from data frequency)."
            ]
            # Seasonal naive: value from same position in previous season
            point = np.zeros((n_targets, horizon))
            for h in range(horizon):
                hist_idx = -(seasonal_period - (h % seasonal_period))
                if abs(hist_idx) <= target_history.shape[1]:
                    point[:, h] = target_history[:, hist_idx]
                else:
                    # Not enough history, use last available
                    point[:, h] = target_history[:, -1]

        quantiles = None
        quantile_levels = np.array(request.quantile_levels)
        warnings.append("Seasonal naive baseline produces deterministic forecasts only.")

        return self._build_result(
            point=point,
            quantiles=quantiles,
            quantile_levels=quantile_levels,
            request=request,
            point_definition="seasonal_repetition",
            context_used=seasonal_period if seasonal_period else 1,
            warnings=warnings,
            metadata={
                "baseline_type": "seasonal_naive",
                "seasonal_period": seasonal_period,
            },
        )

    def _infer_seasonal_period(self, history_index: pd.DatetimeIndex) -> int | None:
        """Infer seasonal period from datetime index frequency."""
        if len(history_index) < 3:
            return None

        try:
            freq = pd.infer_freq(history_index)
            if freq is None:
                return None

            freq = freq.upper()
            # Map pandas frequency to seasonal periods
            if freq.startswith(("T", "MIN")):
                return 60  # Hourly seasonality in minutely data
            elif freq.startswith("H"):
                return 24  # Daily seasonality in hourly data
            elif freq.startswith("D") or freq == "B":
                return 7  # Weekly seasonality in daily data
            elif freq.startswith("W"):
                return 52  # Yearly seasonality in weekly data
            elif freq.startswith("M"):
                return 12  # Yearly seasonality in monthly data
            elif freq.startswith("Q"):
                return 4  # Yearly seasonality in quarterly data
            elif freq.startswith(("A", "Y")):
                return 1  # No intra-year seasonality in annual data
        except Exception:
            pass

        return None


def create_naive_adapter() -> NaiveAdapter:
    return NaiveAdapter()


def create_seasonal_naive_adapter() -> SeasonalNaiveAdapter:
    return SeasonalNaiveAdapter()