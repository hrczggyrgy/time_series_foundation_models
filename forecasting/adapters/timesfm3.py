"""TimesFM 3.0 adapter; install with the separate TimesFM 3.0 profile."""

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


class TimesFM3Adapter(BaseAdapter):
    """Adapter for Google TimesFM 3.0 using its PyTorch forecaster API."""

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_key="timesfm_3",
            display_name="TimesFM 3.0 (330M)",
            checkpoint_id="google/timesfm-3.0-pytorch",
            checkpoint_revision="main",
            supports_cpu=True,
            supports_cuda=True,
            supports_mps=False,
            supports_multivariate=True,
            supports_past_covariates=True,
            supports_known_future_covariates=True,
            native_quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            max_context_length=15360,
            max_horizon=1024,
            supports_batch_inference=True,
            license_notice=(
                "TimesFM Non-Commercial License v1.0: self-hosted weights are restricted "
                "to non-commercial, non-production use."
            ),
        )

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        try:
            from timesfm3.torch import TimesFM3Forecaster
        except ImportError as exc:
            raise ModelNotAvailableError(
                self.capabilities.model_key,
                "timesfm[torch]>=3.0.0,<4.0",
                details={"import_error": str(exc)},
            ) from exc

        try:
            return TimesFM3Forecaster.from_pretrained(
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
        past_future_covariates = None
        if request.known_covariates_future is not None:
            if request.known_covariates_history is None:
                raise ValueError(
                    "TimesFM 3.0 requires known-covariate history when future "
                    "covariates are supplied."
                )
            past_future_covariates = np.concatenate(
                (
                    request.known_covariates_history,
                    request.known_covariates_future,
                ),
                axis=1,
            )

        output = model.predict(
            context=self._ensure_2d_targets(request.target_history),
            horizon=request.horizon,
            past_only_covariates=request.past_covariates,
            past_future_covariates=past_future_covariates,
            return_quantiles=True,
            sort_quantiles=True,
            padding_mode="none",
        )
        if output.forecast is None or output.quantiles is None:
            raise ValueError("TimesFM 3.0 did not return point and quantile forecasts.")

        point = np.asarray(output.forecast, dtype=float)
        native_levels = np.asarray(model.config.quantiles, dtype=float)
        quantiles = np.asarray(output.quantiles, dtype=float)
        expected_shape = (request.n_targets, request.horizon, len(native_levels))
        if quantiles.shape != expected_shape:
            raise ValueError(
                f"TimesFM 3.0 returned quantiles with shape {quantiles.shape}; "
                f"expected {expected_shape}"
            )
        if point.shape != (request.n_targets, request.horizon):
            raise ValueError(
                f"TimesFM 3.0 returned point shape {point.shape}; "
                f"expected {(request.n_targets, request.horizon)}"
            )

        quantiles, warnings = self._validate_quantile_ordering(
            quantiles, native_levels
        )
        requested_levels = np.asarray(request.quantile_levels, dtype=float)
        if not np.array_equal(requested_levels, native_levels):
            quantiles = self._interpolate_quantiles(
                quantiles, native_levels, requested_levels
            )
        return self._build_result(
            point=point,
            quantiles=quantiles,
            quantile_levels=requested_levels,
            request=request,
            point_definition="median",
            context_used=min(
                request.n_history,
                self.capabilities.max_context_length or request.n_history,
            ),
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
