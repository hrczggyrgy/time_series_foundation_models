"""Moirai 1.1 adapter for the separate Uni2TS/PyTorch 2.4 deployment profile."""

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


class MoiraiAdapter(BaseAdapter):
    """Adapter for Salesforce Moirai 1.1-R Small."""

    def __init__(self) -> None:
        super().__init__()
        self._predictor_cache: dict[tuple[int, int, str], Any] = {}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_key="moirai",
            display_name="Moirai 1.1 (Small)",
            checkpoint_id="Salesforce/moirai-1.1-R-small",
            checkpoint_revision="main",
            supports_cpu=True,
            supports_cuda=True,
            supports_multivariate=False,
            native_quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            max_context_length=2000,
            supports_batch_inference=False,
            license_notice=(
                "Review the Salesforce model-card license for weights; "
                "Uni2TS source code is Apache-2.0."
            ),
        )

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        try:
            from uni2ts.model.moirai import MoiraiModule
        except ImportError as exc:
            raise ModelNotAvailableError(
                self.capabilities.model_key,
                "uni2ts",
                details={"import_error": str(exc)},
            ) from exc

        try:
            return MoiraiModule.from_pretrained(self.capabilities.checkpoint_id)
        except Exception as exc:
            raise ModelLoadError(
                self.capabilities.model_key,
                f"Failed to load checkpoint: {exc}",
                original_error=exc,
            ) from exc

    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai import MoiraiForecast

        if request.n_targets != 1:
            raise ValueError("Moirai 1.1 adapter currently supports one target series at a time.")
        if request.past_covariates is not None or request.known_covariates_future is not None:
            raise ValueError("Moirai adapter currently does not support covariates.")
        if self._runtime_config is None:
            raise RuntimeError("Runtime configuration is unavailable; load the adapter first.")

        context_limit = self.capabilities.max_context_length or request.n_history
        context_length = min(request.n_history, context_limit)
        frequency = request.history_index.inferred_freq or "D"
        device = self._runtime_config.resolved_device()
        cache_key = (request.horizon, context_length, device)
        predictor = self._predictor_cache.get(cache_key)
        if predictor is None:
            forecast_model = MoiraiForecast(
                module=model,
                prediction_length=request.horizon,
                context_length=context_length,
                patch_size="auto",
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            predictor = forecast_model.create_predictor(
                batch_size=1,
                device=device,
            )
            self._predictor_cache[cache_key] = predictor

        dataset = ListDataset(
            [
                {
                    "start": request.history_index[-context_length],
                    "target": np.asarray(request.target_history).reshape(-1)[-context_length:],
                }
            ],
            freq=frequency,
        )
        forecast = next(iter(predictor.predict(dataset)))
        requested_levels = np.asarray(request.quantile_levels, dtype=float)
        quantiles = np.column_stack(
            [np.asarray(forecast.quantile(float(level)), dtype=float) for level in requested_levels]
        )
        quantiles = quantiles.reshape(1, request.horizon, len(requested_levels))
        quantiles, warnings = self._validate_quantile_ordering(
            quantiles, requested_levels
        )
        point = np.asarray(forecast.quantile(0.5), dtype=float).reshape(1, request.horizon)
        return self._build_result(
            point=point,
            quantiles=quantiles,
            quantile_levels=requested_levels,
            request=request,
            point_definition="median",
            context_used=context_length,
            warnings=warnings,
            metadata={"sample_count": 100},
        )

    def close(self) -> None:
        self._predictor_cache.clear()
        super().close()
