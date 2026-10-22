"""Base adapter class with common functionality."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from forecasting.contracts import (
    ForecastAdapter,
    ForecastRequest,
    ForecastResult,
    ModelCapabilities,
    RuntimeConfig,
)
from forecasting.exceptions import InferenceError, ModelLoadError


class BaseAdapter(ABC, ForecastAdapter):
    """Base class for forecast adapters with shared utilities."""

    def __init__(self) -> None:
        self._loaded = False
        self._runtime_config: RuntimeConfig | None = None

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return static capabilities."""
        pass

    @abstractmethod
    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        """Load the actual model. Return model handle."""
        pass

    @abstractmethod
    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        """Core prediction logic."""
        pass

    def load(self, runtime_config: RuntimeConfig) -> None:
        """Load model with runtime configuration."""
        if self._loaded and self._runtime_config == runtime_config:
            return

        try:
            model = self._load_model(runtime_config)
            self._model = model
            self._runtime_config = runtime_config
            self._loaded = True
        except Exception as e:
            raise ModelLoadError(
                self.capabilities.model_key,
                f"Failed to load model: {e}",
                original_error=e,
            ) from e

    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Generate forecast for a single request."""
        if not self._loaded:
            raise InferenceError(
                self.capabilities.model_key,
                "Model not loaded. Call load() first.",
            )

        # Validate capabilities
        can_handle, reasons = self.capabilities.can_handle_request(request)
        if not can_handle:
            raise InferenceError(
                self.capabilities.model_key,
                f"Request not supported: {'; '.join(reasons)}",
                request_info={"reasons": reasons},
            )

        try:
            return self._predict_impl(request, self._model)
        except Exception as e:
            raise InferenceError(
                self.capabilities.model_key,
                f"Prediction failed: {e}",
                request_info={
                    "n_targets": request.n_targets,
                    "n_history": request.n_history,
                    "horizon": request.horizon,
                },
                original_error=e,
            ) from e

    def predict_batch(
        self, requests: list[ForecastRequest]
    ) -> list[ForecastResult]:
        """Default: sequential prediction. Override for true batching."""
        return [self.predict(r) for r in requests]

    def close(self) -> None:
        """Release resources."""
        self._loaded = False
        self._model = None
        self._runtime_config = None

    # --- Shared utilities ---

    def _ensure_2d_targets(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is [targets, history]."""
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    def _ensure_3d_quantiles(
        self, quantiles: np.ndarray, n_targets: int, horizon: int, n_quantiles: int
    ) -> np.ndarray:
        """Ensure quantiles array is [targets, horizon, quantiles]."""
        if quantiles.ndim == 2:
            # [horizon, quantiles] -> [1, horizon, quantiles]
            quantiles = quantiles.reshape(1, *quantiles.shape)
        elif quantiles.ndim == 1:
            # [horizon] -> [1, horizon, 1] (degenerate)
            quantiles = quantiles.reshape(1, -1, 1)
        if quantiles.shape != (n_targets, horizon, n_quantiles):
            raise ValueError(
                f"Quantiles shape {quantiles.shape} != expected ({n_targets}, {horizon}, {n_quantiles})"
            )
        return quantiles

    def _validate_quantile_ordering(
        self, quantiles: np.ndarray, quantile_levels: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Check for quantile crossing, repair if possible."""
        warnings = []
        n_targets, horizon, n_quantiles = quantiles.shape

        for t in range(n_targets):
            for h in range(horizon):
                q_vals = quantiles[t, h]
                diffs = np.diff(q_vals)
                if np.any(diffs < -1e-10):
                    # Crossing detected
                    warnings.append(
                        f"Target {t}, step {h}: quantile crossing detected "
                        f"(levels={quantile_levels}, values={q_vals})"
                    )
                    # Isotonic regression repair (pool adjacent violators)
                    q_vals = self._isotonic_regression(q_vals)
                    quantiles[t, h] = q_vals

        return quantiles, warnings

    def _isotonic_regression(self, y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators Algorithm for isotonic regression."""
        n = len(y)
        if n <= 1:
            return y

        # PAVA
        blocks = [{"start": i, "end": i, "value": y[i]} for i in range(n)]

        i = 0
        while i < len(blocks) - 1:
            if blocks[i]["value"] <= blocks[i + 1]["value"] + 1e-12:
                i += 1
            else:
                # Merge blocks
                merged = {
                    "start": blocks[i]["start"],
                    "end": blocks[i + 1]["end"],
                    "value": (
                        blocks[i]["value"] * (blocks[i]["end"] - blocks[i]["start"] + 1)
                        + blocks[i + 1]["value"] * (blocks[i + 1]["end"] - blocks[i + 1]["start"] + 1)
                    )
                    / (
                        (blocks[i]["end"] - blocks[i]["start"] + 1)
                        + (blocks[i + 1]["end"] - blocks[i + 1]["start"] + 1)
                    ),
                }
                blocks[i : i + 2] = [merged]
                if i > 0:
                    i -= 1

        # Expand back
        result = np.empty(n)
        for block in blocks:
            result[block["start"] : block["end"] + 1] = block["value"]
        return result

    def _build_result(
        self,
        point: np.ndarray,
        quantiles: np.ndarray | None,
        quantile_levels: np.ndarray,
        request: ForecastRequest,
        point_definition: str,
        context_used: int,
        warnings: list[str],
        metadata: dict[str, Any],
    ) -> ForecastResult:
        """Construct standardized ForecastResult."""
        return ForecastResult(
            point=point,
            quantiles=quantiles,
            quantile_levels=quantile_levels,
            target_names=request.target_names,
            future_index=request.future_index,
            model_key=self.capabilities.model_key,
            checkpoint_id=self.capabilities.checkpoint_id,
            checkpoint_revision=self.capabilities.checkpoint_revision,
            point_definition=point_definition,
            context_used=context_used,
            warnings=tuple(warnings),
            metadata=metadata,
        )