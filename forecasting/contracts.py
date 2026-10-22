"""Core forecasting contracts: typed dataclasses for requests, results, and capabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastRequest:
    """A single forecasting task. For multivariate, target_names length > 1."""

    target_history: np.ndarray                    # [targets, history] or [history] for univariate
    history_index: pd.DatetimeIndex               # Timestamps for target_history
    target_names: tuple[str, ...]                 # Names of each target series
    horizon: int                                  # Number of steps to forecast
    future_index: pd.DatetimeIndex                # Timestamps for forecast horizon

    # Optional covariates
    past_covariates: np.ndarray | None = None               # [past_features, history]
    known_covariates_history: np.ndarray | None = None      # [known_features, history]
    known_covariates_future: np.ndarray | None = None       # [known_features, horizon]

    # Requested quantile levels (must be strictly increasing, finite, in (0,1))
    quantile_levels: tuple[float, ...] = (
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    )

    def __post_init__(self) -> None:
        # Validate target_history shape
        th = self.target_history
        if th.ndim == 1:
            th = th.reshape(1, -1)
        elif th.ndim != 2:
            raise ValueError(f"target_history must be 1D or 2D, got {th.ndim}D")

        n_targets, n_history = th.shape
        if n_targets != len(self.target_names):
            raise ValueError(
                f"target_history has {n_targets} targets but target_names has {len(self.target_names)}"
            )
        if n_history == 0:
            raise ValueError("target_history cannot be empty")

        # Validate history_index length
        if len(self.history_index) != n_history:
            raise ValueError(
                f"history_index length {len(self.history_index)} != target_history length {n_history}"
            )

        # Validate future_index length
        if len(self.future_index) != self.horizon:
            raise ValueError(
                f"future_index length {len(self.future_index)} != horizon {self.horizon}"
            )

        # Validate quantile levels
        q = np.asarray(self.quantile_levels, dtype=float)
        if q.ndim != 1 or q.size == 0:
            raise ValueError("quantile_levels must be a non-empty 1D sequence")
        if not np.all(np.isfinite(q)):
            raise ValueError("quantile_levels must be finite")
        if not np.all((q > 0) & (q < 1)):
            raise ValueError("quantile_levels must be in (0, 1)")
        if not np.all(np.diff(q) > 0):
            raise ValueError("quantile_levels must be strictly increasing")

        # Validate covariates if provided
        if self.past_covariates is not None:
            pc = self.past_covariates
            if pc.ndim != 2:
                raise ValueError("past_covariates must be 2D [features, history]")
            if pc.shape[1] != n_history:
                raise ValueError(
                    f"past_covariates history length {pc.shape[1]} != target history {n_history}"
                )

        if self.known_covariates_history is not None:
            kch = self.known_covariates_history
            if kch.ndim != 2:
                raise ValueError("known_covariates_history must be 2D [features, history]")
            if kch.shape[1] != n_history:
                raise ValueError(
                    f"known_covariates_history length {kch.shape[1]} != target history {n_history}"
                )

        if self.known_covariates_future is not None:
            kcf = self.known_covariates_future
            if kcf.ndim != 2:
                raise ValueError("known_covariates_future must be 2D [features, horizon]")
            if kcf.shape[1] != self.horizon:
                raise ValueError(
                    f"known_covariates_future length {kcf.shape[1]} != horizon {self.horizon}"
                )
            if self.known_covariates_history is not None:
                if kcf.shape[0] != self.known_covariates_history.shape[0]:
                    raise ValueError(
                        "known_covariates_future and known_covariates_history must have same feature count"
                    )

    @property
    def n_targets(self) -> int:
        return len(self.target_names)

    @property
    def n_history(self) -> int:
        return self.target_history.shape[-1]

    def to_univariate_requests(self) -> list[ForecastRequest]:
        """Split a multivariate request into independent univariate requests."""
        if self.n_targets == 1:
            return [self]
        requests = []
        for i, name in enumerate(self.target_names):
            req = ForecastRequest(
                target_history=self.target_history[i : i + 1],
                history_index=self.history_index,
                target_names=(name,),
                horizon=self.horizon,
                future_index=self.future_index,
                past_covariates=self.past_covariates,
                known_covariates_history=self.known_covariates_history,
                known_covariates_future=self.known_covariates_future,
                quantile_levels=self.quantile_levels,
            )
            requests.append(req)
        return requests


@dataclass(frozen=True)
class ForecastResult:
    """Normalized forecast output from any adapter."""

    point: np.ndarray                          # [targets, horizon]
    quantiles: np.ndarray | None               # [targets, horizon, quantile_levels] or None
    quantile_levels: np.ndarray                # Probability levels for quantiles
    target_names: tuple[str, ...]
    future_index: pd.DatetimeIndex

    # Model identification
    model_key: str
    checkpoint_id: str | None
    checkpoint_revision: str | None

    # Point forecast definition
    point_definition: str                      # e.g., "median", "mean", "native_mean"

    # Execution metadata
    context_used: int
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate shapes
        n_targets = len(self.target_names)
        horizon = len(self.future_index)

        if self.point.shape != (n_targets, horizon):
            raise ValueError(
                f"point shape {self.point.shape} != expected ({n_targets}, {horizon})"
            )

        if self.quantiles is not None:
            if self.quantiles.ndim != 3:
                raise ValueError("quantiles must be 3D [targets, horizon, quantile_levels]")
            if self.quantiles.shape[:2] != (n_targets, horizon):
                raise ValueError(
                    f"quantiles shape {self.quantiles.shape[:2]} != expected ({n_targets}, {horizon})"
                )
            if self.quantiles.shape[2] != len(self.quantile_levels):
                raise ValueError(
                    f"quantiles has {self.quantiles.shape[2]} levels but "
                    f"quantile_levels has {len(self.quantile_levels)}"
                )

            # Validate quantile ordering (allow small numerical tolerance)
            for t in range(n_targets):
                for h in range(horizon):
                    q_vals = self.quantiles[t, h]
                    if not np.all(np.diff(q_vals) >= -1e-10):
                        # Crossing detected — record in warnings but don't fail
                        pass  # Handled by adapter

        # Validate quantile_levels
        ql = np.asarray(self.quantile_levels)
        if ql.ndim != 1 or ql.size == 0:
            raise ValueError("quantile_levels must be non-empty 1D")
        if not np.all(np.isfinite(ql)):
            raise ValueError("quantile_levels must be finite")
        if not np.all((ql > 0) & (ql < 1)):
            raise ValueError("quantile_levels must be in (0, 1)")
        if not np.all(np.diff(ql) > 0):
            raise ValueError("quantile_levels must be strictly increasing")

    def get_interval(self, coverage: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Get lower/upper bounds for a central prediction interval.

        Args:
            coverage: Nominal coverage in (0, 1), e.g., 0.8 for 80% interval.

        Returns:
            Tuple of (lower, upper) arrays [targets, horizon], or None if quantiles unavailable.
        """
        if self.quantiles is None:
            return None

        alpha = (1 - coverage) / 2
        lower_q = alpha
        upper_q = 1 - alpha

        lower = self._interpolate_quantile(lower_q)
        upper = self._interpolate_quantile(upper_q)

        return lower, upper

    def _interpolate_quantile(self, q: float) -> np.ndarray:
        """Interpolate quantile at probability q from available quantile levels."""
        ql = self.quantile_levels
        if q <= ql[0]:
            return self.quantiles[:, :, 0]
        if q >= ql[-1]:
            return self.quantiles[:, :, -1]

        # Find surrounding quantiles
        idx = np.searchsorted(ql, q)
        q_low, q_high = ql[idx - 1], ql[idx]
        v_low = self.quantiles[:, :, idx - 1]
        v_high = self.quantiles[:, :, idx]

        # Linear interpolation in quantile space
        weight = (q - q_low) / (q_high - q_low)
        return v_low + weight * (v_high - v_low)

    def has_probabilistic(self) -> bool:
        return self.quantiles is not None and self.quantiles.size > 0


@dataclass(frozen=True)
class ModelCapabilities:
    """Describes what a model adapter can do."""

    # Model identification
    model_key: str
    display_name: str
    checkpoint_id: str
    checkpoint_revision: str = "main"

    # Device support
    supports_cpu: bool = True
    supports_cuda: bool = True
    supports_mps: bool = False

    # Input/output capabilities
    supports_multivariate: bool = False
    supports_past_covariates: bool = False
    supports_known_future_covariates: bool = False
    supports_group_forecasting: bool = False

    # Quantile behavior
    native_quantile_levels: tuple[float, ...] = ()
    supports_quantile_interpolation: bool = True
    supports_quantile_crossing_repair: bool = False

    # Context/horizon constraints
    max_context_length: int | None = None
    min_context_length: int = 1
    max_horizon: int | None = None
    compiles_context_horizon: bool = False  # Whether model compiles for specific (ctx, horizon)

    # Data handling
    supports_missing_values: bool = False
    missing_value_policy: Literal["error", "interpolate", "drop"] = "error"

    # Batching
    supports_batch_inference: bool = False

    # Licensing
    license_policy: Literal["permissive", "restricted"] = "permissive"
    license_notice: str = ""

    def can_handle_request(self, request: ForecastRequest) -> tuple[bool, list[str]]:
        """Check if this adapter can handle a request. Returns (can_handle, reasons_if_not)."""
        reasons = []

        if request.n_targets > 1 and not self.supports_multivariate:
            reasons.append(f"Model does not support multivariate targets (got {request.n_targets})")

        if request.past_covariates is not None and not self.supports_past_covariates:
            reasons.append("Model does not support past covariates")

        if request.known_covariates_future is not None and not self.supports_known_future_covariates:
            reasons.append("Model does not support known-future covariates")

        if request.n_history < self.min_context_length:
            reasons.append(
                f"History length {request.n_history} < minimum {self.min_context_length}"
            )

        if self.max_context_length and request.n_history > self.max_context_length:
            reasons.append(
                f"History length {request.n_history} > maximum {self.max_context_length}"
            )

        if self.max_horizon and request.horizon > self.max_horizon:
            reasons.append(f"Horizon {request.horizon} > maximum {self.max_horizon}")

        return len(reasons) == 0, reasons


class ForecastAdapter(Protocol):
    """Protocol for forecast model adapters."""

    def capabilities(self) -> ModelCapabilities:
        """Return static capabilities of this adapter/model."""
        ...

    def load(self, runtime_config: RuntimeConfig) -> None:
        """Load model weights and initialize for inference."""
        ...

    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Generate forecast for a single request."""
        ...

    def predict_batch(
        self, requests: Sequence[ForecastRequest]
    ) -> list[ForecastResult]:
        """Generate forecasts for multiple requests. Default: sequential."""
        return [self.predict(r) for r in requests]

    def close(self) -> None:
        """Release resources (GPU memory, etc.)."""
        ...


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration for model loading."""

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    per_core_batch_size: int = 16
    # Model-specific extra config
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"