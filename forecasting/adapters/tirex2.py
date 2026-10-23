"""TiRex-2 adapter for zero-shot univariate and multivariate forecasting."""

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
from forecasting.exceptions import ModelLoadError, ModelNotAvailableError

_QUANTILE_LEVELS = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))


class TiRex2Adapter(BaseAdapter):
    """Adapter for the NX-AI TiRex-2 model."""

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_key="tirex_2",
            display_name="TiRex-2",
            checkpoint_id="NX-AI/TiRex-2",
            checkpoint_revision="main",
            supports_cpu=True,
            supports_cuda=True,
            supports_multivariate=True,
            supports_past_covariates=True,
            supports_known_future_covariates=True,
            native_quantile_levels=tuple(_QUANTILE_LEVELS.tolist()),
            max_context_length=32768,
            max_horizon=1024,
            supports_batch_inference=True,
            license_notice="Apache-2.0",
        )

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        try:
            from tirex2 import load_model
        except ImportError as exc:
            raise ModelNotAvailableError(
                self.capabilities.model_key,
                "tirex-2>=0.4.0",
                details={"import_error": str(exc)},
            ) from exc

        try:
            return load_model(
                self.capabilities.checkpoint_id,
                device=runtime_config.resolved_device(),
            )
        except Exception as exc:
            raise ModelLoadError(
                self.capabilities.model_key,
                f"Failed to load checkpoint: {exc}",
                original_error=exc,
            ) from exc

    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        import torch
        from tirex2 import TimeseriesType

        target = torch.as_tensor(
            self._ensure_2d_targets(request.target_history), dtype=torch.float32
        )
        past_covariates = (
            torch.as_tensor(request.past_covariates, dtype=torch.float32)
            if request.past_covariates is not None
            else None
        )
        future_covariates = None
        if request.known_covariates_future is not None:
            if request.known_covariates_history is None:
                raise ValueError(
                    "TiRex-2 requires known-covariate history when future covariates are supplied."
                )
            future_covariates = torch.as_tensor(
                np.concatenate(
                    (
                        request.known_covariates_history,
                        request.known_covariates_future,
                    ),
                    axis=1,
                ),
                dtype=torch.float32,
            )

        series = TimeseriesType(
            target=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        forecasts = model.forecast(
            [series],
            prediction_length=request.horizon,
            output_type="numpy",
        )
        forecast = np.asarray(forecasts[0], dtype=float)
        expected_shape = (
            request.n_targets,
            len(_QUANTILE_LEVELS),
            request.horizon,
        )
        if forecast.shape != expected_shape:
            raise ValueError(
                f"TiRex-2 returned shape {forecast.shape}; expected {expected_shape}"
            )

        native_quantiles, warnings = self._validate_quantile_ordering(
            np.moveaxis(forecast, 1, -1), _QUANTILE_LEVELS
        )
        requested_levels = np.asarray(request.quantile_levels, dtype=float)
        if not np.array_equal(requested_levels, _QUANTILE_LEVELS):
            quantiles = self._interpolate_quantiles(
                native_quantiles, _QUANTILE_LEVELS, requested_levels
            )
        else:
            quantiles = native_quantiles
        point = self._interpolate_quantiles(
            native_quantiles, _QUANTILE_LEVELS, np.asarray([0.5])
        )[:, :, 0]

        return self._build_result(
            point=point,
            quantiles=quantiles,
            quantile_levels=requested_levels,
            request=request,
            point_definition="median",
            context_used=request.n_history,
            warnings=warnings,
            metadata={},
        )

    @staticmethod
    def _interpolate_quantiles(
        quantiles: np.ndarray, source_levels: np.ndarray, target_levels: np.ndarray
    ) -> np.ndarray:
        result = np.empty((*quantiles.shape[:2], len(target_levels)), dtype=float)
        for target_idx in range(quantiles.shape[0]):
            for step_idx in range(quantiles.shape[1]):
                result[target_idx, step_idx] = np.interp(
                    target_levels,
                    source_levels,
                    quantiles[target_idx, step_idx],
                )
        return result
