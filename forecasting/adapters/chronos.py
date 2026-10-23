"""Chronos-T5 adapter using sampled forecast trajectories."""

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


class ChronosAdapter(BaseAdapter):
    """Adapter for Amazon's Apache-2.0 Chronos-T5 Small checkpoint."""

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_key="chronos",
            display_name="Chronos-T5 Small",
            checkpoint_id="amazon/chronos-t5-small",
            checkpoint_revision="main",
            supports_cpu=True,
            supports_cuda=True,
            supports_multivariate=True,
            native_quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            max_context_length=512,
            supports_batch_inference=True,
            license_notice="Apache-2.0",
        )

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        try:
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise ModelNotAvailableError(
                self.capabilities.model_key,
                "chronos-forecasting>=1.4.0,<3.0.0",
                details={"import_error": str(exc)},
            ) from exc

        try:
            return ChronosPipeline.from_pretrained(
                self.capabilities.checkpoint_id,
                device_map=runtime_config.resolved_device(),
            )
        except Exception as exc:
            raise ModelLoadError(
                self.capabilities.model_key,
                f"Failed to load checkpoint: {exc}",
                original_error=exc,
            ) from exc

    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        import torch

        values = self._ensure_2d_targets(request.target_history)
        contexts = torch.as_tensor(values, dtype=torch.float32)
        with torch.inference_mode():
            samples = model.predict(
                contexts,
                prediction_length=request.horizon,
                num_samples=100,
            )

        samples_array = np.asarray(samples, dtype=float)
        expected_shape_start = (request.n_targets, 100)
        if (
            samples_array.ndim != 3
            or samples_array.shape[:2] != expected_shape_start
            or samples_array.shape[2] < request.horizon
        ):
            raise ValueError(
                "Chronos returned samples with shape "
                f"{samples_array.shape}; expected ({request.n_targets}, 100, >= {request.horizon})"
            )
        samples_array = samples_array[:, :, : request.horizon]

        requested_levels = np.asarray(request.quantile_levels, dtype=float)
        quantiles = np.quantile(samples_array, requested_levels, axis=1)
        quantiles = np.moveaxis(quantiles, 0, -1)
        quantiles, warnings = self._validate_quantile_ordering(
            quantiles, requested_levels
        )
        point = np.quantile(samples_array, 0.5, axis=1)
        return self._build_result(
            point=point,
            quantiles=quantiles,
            quantile_levels=requested_levels,
            request=request,
            point_definition="sample_median",
            context_used=min(
                request.n_history,
                self.capabilities.max_context_length or request.n_history,
            ),
            warnings=warnings,
            metadata={"sample_count": samples_array.shape[1]},
        )
