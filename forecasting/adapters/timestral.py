"""Timestral adapter family — Modified MIT with $20M revenue cap."""

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


# Checkpoint configurations
TIMESTRAL_CHECKPOINTS = {
    "timestral_5m": {
        "checkpoint_id": "mistralai/Timestral-5M",
        "display_name": "Timestral 5M",
    },
    "timestral_28m": {
        "checkpoint_id": "mistralai/Timestral-28M",
        "display_name": "Timestral 28M",
    },
    "timestral_123m": {
        "checkpoint_id": "mistralai/Timestral-123M",
        "display_name": "Timestral 123M",
    },
    "timestral_209m": {
        "checkpoint_id": "mistralai/Timestral-209M",
        "display_name": "Timestral 209M",
    },
}


class TimestralAdapter(BaseAdapter):
    """Adapter for Timestral family (5M, 28M, 123M, 209M).

    All checkpoints share the same API via timestral.TimestralForecaster.
    Checkpoint is determined by model_key from registry.
    """

    def __init__(self) -> None:
        super().__init__()
        self._checkpoint_id = ""
        self._model_key = ""
        self._display_name = ""

    def _set_checkpoint(self, model_key: str) -> None:
        """Set checkpoint from model_key."""
        if model_key not in TIMESTRAL_CHECKPOINTS:
            raise ValueError(f"Unknown Timestral model key: {model_key}")
        cfg = TIMESTRAL_CHECKPOINTS[model_key]
        self._checkpoint_id = cfg["checkpoint_id"]
        self._model_key = model_key
        self._display_name = cfg["display_name"]

    @property
    def capabilities(self) -> ModelCapabilities:
        if not self._model_key:
            # Return default capabilities for registry listing
            return ModelCapabilities(
                model_key="timestral_base",
                display_name="Timestral (base)",
                checkpoint_id="",
                checkpoint_revision="main",
                supports_cpu=True,
                supports_cuda=True,
                supports_mps=False,
                supports_multivariate=False,  # Currently forecasts each target independently
                supports_past_covariates=False,  # Not yet implemented
                supports_known_future_covariates=False,  # Not yet implemented
                native_quantile_levels=(
                    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                    0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
                ),
                max_context_length=32768,
                min_context_length=1,
                max_horizon=1024,
                compiles_context_horizon=False,
                supports_batch_inference=True,
                license_policy="restricted",
                license_notice=(
                    "Modified MIT License with revenue restriction: "
                    "You are not authorized to use this model if your company's (or employer's) "
                    "global consolidated monthly revenue exceeds $20 million. "
                    "This applies to the model, derivatives, modifications, and combined works. "
                    "Contact sales@mistral.ai for commercial license. "
                    "Inference code is Apache-2.0."
                ),
            )

        return ModelCapabilities(
            model_key=self._model_key,
            display_name=self._display_name,
            checkpoint_id=self._checkpoint_id,
            checkpoint_revision="main",
            supports_cpu=True,
            supports_cuda=True,
            supports_mps=False,
            supports_multivariate=False,  # Currently forecasts each target independently
            supports_past_covariates=False,  # Not yet implemented
            supports_known_future_covariates=False,  # Not yet implemented
            native_quantile_levels=(
                0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
            ),
            max_context_length=32768,
            min_context_length=1,
            max_horizon=1024,
            compiles_context_horizon=False,
            supports_batch_inference=True,
            license_policy="restricted",
            license_notice=(
                "Modified MIT License with revenue restriction: "
                "You are not authorized to use this model if your company's (or employer's) "
                "global consolidated monthly revenue exceeds $20 million. "
                "This applies to the model, derivatives, modifications, and combined works. "
                "Contact sales@mistral.ai for commercial license. "
                "Inference code is Apache-2.0."
            ),
        )

    def load(self, runtime_config: RuntimeConfig) -> None:
        """Load model with checkpoint from runtime_config.extra['model_key']."""
        model_key = runtime_config.extra.get("model_key", "timestral_209m")
        self._set_checkpoint(model_key)
        super().load(runtime_config)

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        try:
            import torch
            from timestral import TimestralForecaster
        except ImportError as e:
            raise ModelNotAvailableError(
                self._model_key or "timestral",
                "timestral",
                details={"import_error": str(e)},
            ) from e

        device = runtime_config.resolved_device()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        try:
            model = TimestralForecaster.from_pretrained(
                self._checkpoint_id,
                device=device,
            )
        except Exception as e:
            raise ModelLoadError(
                self._model_key,
                f"Failed to load checkpoint: {e}",
                original_error=e,
            ) from e

        return model

    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        import torch

        horizon = request.horizon
        n_targets = request.n_targets
        n_history = request.n_history

        target_history = self._ensure_2d_targets(request.target_history)  # [targets, history]

        all_points = []
        all_quantiles = []
        warnings = []

        requested_levels = np.array(request.quantile_levels)

        for t in range(n_targets):
            context = target_history[t].astype(np.float32)

            # Get forecast
            forecast = model.forecast(context, prediction_length=horizon)

            # Extract median (point): [1, 1, horizon] -> [horizon]
            point = forecast.median.squeeze().cpu().numpy()
            if point.ndim == 0:
                point = np.array([point])
            all_points.append(point)

            # Get quantiles at requested levels
            q_vals = []
            for q in requested_levels:
                try:
                    q_tensor = forecast.quantile(float(q))  # [1, 1, horizon]
                    q_val = q_tensor.squeeze().cpu().numpy()
                    if q_val.ndim == 0:
                        q_val = np.array([q_val])
                    q_vals.append(q_val)
                except Exception as e:
                    raise ValueError(
                        f"Timestral failed to compute quantile {q:.0%}. "
                        f"This may indicate an incompatibility with the requested quantile levels. "
                        f"Error: {e}"
                    ) from e

            # q_vals is list of [horizon] arrays
            q_arr = np.column_stack(q_vals)  # [horizon, n_quantiles]
            all_quantiles.append(q_arr)

        point = np.stack(all_points)  # [targets, horizon]
        quantiles = np.stack(all_quantiles)  # [targets, horizon, n_quantiles]

        # Validate quantile ordering
        quantiles, q_warnings = self._validate_quantile_ordering(
            quantiles, requested_levels
        )
        warnings.extend(q_warnings)

        # Note: Timestral returns quantiles at exactly the requested levels
        # No interpolation needed if we requested the exact levels

        return self._build_result(
            point=point,
            quantiles=quantiles,
            quantile_levels=requested_levels,
            request=request,
            point_definition="median",
            context_used=min(n_history, self.capabilities.max_context_length or n_history),
            warnings=warnings,
            metadata={
                "checkpoint": self._checkpoint_id,
            },
        )


# Factory functions for each checkpoint
def create_timestral_209m_adapter() -> TimestralAdapter:
    return TimestralAdapter(
        checkpoint_id="mistralai/Timestral-209M",
        model_key="timestral_209m",
        display_name="Timestral 209M",
    )


def create_timestral_28m_adapter() -> TimestralAdapter:
    return TimestralAdapter(
        checkpoint_id="mistralai/Timestral-28M",
        model_key="timestral_28m",
        display_name="Timestral 28M",
    )


def create_timestral_123m_adapter() -> TimestralAdapter:
    return TimestralAdapter(
        checkpoint_id="mistralai/Timestral-123M",
        model_key="timestral_123m",
        display_name="Timestral 123M",
    )


def create_timestral_5m_adapter() -> TimestralAdapter:
    return TimestralAdapter(
        checkpoint_id="mistralai/Timestral-5M",
        model_key="timestral_5m",
        display_name="Timestral 5M",
    )