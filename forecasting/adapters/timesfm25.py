"""TimesFM 2.5 adapter — preserves existing app.py behavior."""

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
from forecasting.exceptions import ModelLoadError, ModelNotAvailableError


class TimesFM25Adapter(BaseAdapter):
    """Adapter for TimesFM 2.5 (200M) — Apache-2.0 licensed."""

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_key="timesfm_2_5",
            display_name="TimesFM 2.5 (200M)",
            checkpoint_id="google/timesfm-2.5-200m-pytorch",
            checkpoint_revision="main",
            supports_cpu=True,
            supports_cuda=True,
            supports_mps=False,
            supports_multivariate=False,
            supports_past_covariates=False,
            supports_known_future_covariates=True,  # Limited support
            native_quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            max_context_length=16384,
            min_context_length=32,
            max_horizon=512,
            compiles_context_horizon=True,
            supports_batch_inference=True,
            license_policy="permissive",
            license_notice="Apache-2.0 (weights and code)",
        )

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        try:
            import timesfm
            import torch
        except ImportError as e:
            raise ModelNotAvailableError(
                self.capabilities.model_key,
                "timesfm[torch]",
                details={"import_error": str(e)},
            ) from e

        # Set threads for CPU
        device = runtime_config.resolved_device()
        if device == "cpu":
            torch.set_num_threads(4)
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        try:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.capabilities.checkpoint_id,
                torch_compile=runtime_config.compile,
            )
        except TypeError:
            # Fallback if torch_compile not supported
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.capabilities.checkpoint_id
            )
        except Exception as e:
            raise ModelLoadError(
                self.capabilities.model_key,
                f"Failed to load checkpoint: {e}",
                original_error=e,
            ) from e

        return model

    def _compile_model(
        self,
        model: Any,
        context_len: int,
        horizon: int,
        infer_is_positive: bool,
    ) -> tuple[int, int, bool]:
        """Compile model with TimesFM 2.5 ForecastConfig."""
        import timesfm

        # Round to 32-step increments (TimesFM 2.5 requirement)
        rounded_context = int(
            min(
                self.capabilities.max_context_length,
                max(self.capabilities.min_context_length, ((context_len + 31) // 32) * 32),
            )
        )
        rounded_horizon = int(max(32, ((horizon + 31) // 32) * 32))

        config = timesfm.ForecastConfig(
            max_context=rounded_context,
            max_horizon=rounded_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=infer_is_positive,
            fix_quantile_crossing=True,
        )

        if hasattr(model, "compile"):
            model.compile(config)

        return rounded_context, rounded_horizon, infer_is_positive

    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        # TimesFM 2.5 only supports univariate
        if request.n_targets > 1:
            # Process each target independently
            results = []
            for i in range(request.n_targets):
                univariate_req = ForecastRequest(
                    target_history=request.target_history[i : i + 1],
                    history_index=request.history_index,
                    target_names=(request.target_names[i],),
                    horizon=request.horizon,
                    future_index=request.future_index,
                    past_covariates=request.past_covariates,
                    known_covariates_history=request.known_covariates_history,
                    known_covariates_future=request.known_covariates_future,
                    quantile_levels=request.quantile_levels,
                )
                results.append(self._predict_univariate(univariate_req, model))
            return self._merge_results(results, request)

        return self._predict_univariate(request, model)

    def _predict_univariate(
        self, request: ForecastRequest, model: Any
    ) -> ForecastResult:
        values = self._ensure_2d_targets(request.target_history)[0]  # [history]
        horizon = request.horizon
        all_non_negative = bool(np.all(values >= 0))

        # Determine context length
        max_ctx = self.capabilities.max_context_length
        ctx = min(len(values), max_ctx)

        # Compile model
        compiled_ctx, compiled_horizon, infer_is_positive = self._compile_model(
            model, ctx, max(horizon, 128), all_non_negative
        )

        # Prepare input
        input_values = values[-ctx:].astype(np.float32)

        # Run forecast
        point, quantiles = model.forecast(horizon=horizon, inputs=[input_values])

        point_arr = np.asarray(point[0], dtype=float)  # [horizon]
        q_arr = (
            np.asarray(quantiles[0], dtype=float)
            if quantiles is not None
            else np.column_stack([point_arr] * 9)  # 9 quantiles (0.1-0.9)
        )

        # q_arr shape: [horizon, 9] for quantiles 0.1-0.9
        # But TimesFM 2.5 returns 10 columns: mean + 9 quantiles
        # Verify and adjust
        if q_arr.shape[1] == 10:
            # First column is mean, quantiles are 0.1-0.9 at indices 1-9
            mean_col = q_arr[:, 0]
            q_arr = q_arr[:, 1:]  # Keep only quantiles
        elif q_arr.shape[1] != 9:
            # Unexpected shape - raise error instead of fabricating quantiles
            raise ValueError(
                f"TimesFM 2.5 returned unexpected quantile shape: {q_arr.shape}. "
                f"Expected 9 quantile columns (0.1-0.9) or 10 columns (mean + 9 quantiles). "
                f"Cannot fabricate probabilistic forecasts from point predictions."
            )

        # Reshape to [1, horizon, 9]
        q_arr = q_arr.reshape(1, horizon, 9)
        point_arr = point_arr.reshape(1, horizon)

        # Validate quantile ordering
        q_arr, warnings = self._validate_quantile_ordering(
            q_arr, np.array(self.capabilities.native_quantile_levels)
        )

        # Interpolate to requested quantile levels if different
        requested_levels = np.array(request.quantile_levels)
        native_levels = np.array(self.capabilities.native_quantile_levels)

        # Check if requested levels are outside native range
        if requested_levels[0] < native_levels[0] or requested_levels[-1] > native_levels[-1]:
            warnings.append(
                f"TimesFM 2.5 only supports quantiles {native_levels[0]:.0%} to {native_levels[-1]:.0%}. "
                f"Requested levels outside this range will be clamped to nearest available quantile. "
                f"This means intervals like 90% (P05-P95) will effectively be narrower than advertised."
            )

        if not np.array_equal(requested_levels, native_levels):
            q_arr = self._interpolate_quantiles(q_arr, native_levels, requested_levels)
            warnings.append(
                f"Interpolated quantiles from native {native_levels} to requested {requested_levels}"
            )

        return self._build_result(
            point=point_arr,
            quantiles=q_arr,
            quantile_levels=requested_levels,
            request=request,
            point_definition="median",  # TimesFM API returns median as point forecast
            context_used=ctx,
            warnings=warnings,
            metadata={
                "compiled_context": compiled_ctx,
                "compiled_horizon": compiled_horizon,
                "infer_is_positive": infer_is_positive,
            },
        )

    def _interpolate_quantiles(
        self, quantiles: np.ndarray, from_levels: np.ndarray, to_levels: np.ndarray
    ) -> np.ndarray:
        """Interpolate quantiles from one set of levels to another."""
        n_targets, horizon, _ = quantiles.shape
        result = np.empty((n_targets, horizon, len(to_levels)))

        for t in range(n_targets):
            for h in range(horizon):
                q_vals = quantiles[t, h]
                for i, q in enumerate(to_levels):
                    if q <= from_levels[0]:
                        result[t, h, i] = q_vals[0]
                    elif q >= from_levels[-1]:
                        result[t, h, i] = q_vals[-1]
                    else:
                        idx = np.searchsorted(from_levels, q)
                        q_low, q_high = from_levels[idx - 1], from_levels[idx]
                        v_low, v_high = q_vals[idx - 1], q_vals[idx]
                        weight = (q - q_low) / (q_high - q_low)
                        result[t, h, i] = v_low + weight * (v_high - v_low)

        return result

    def _merge_results(
        self, results: list[ForecastResult], request: ForecastRequest
    ) -> ForecastResult:
        """Merge multiple univariate results into one multivariate result."""
        point = np.vstack([r.point for r in results])  # [targets, horizon]
        quantiles = np.vstack([r.quantiles for r in results])  # [targets, horizon, q]
        all_warnings = []
        for r in results:
            all_warnings.extend(r.warnings)

        return self._build_result(
            point=point,
            quantiles=quantiles,
            quantile_levels=request.quantile_levels,
            request=request,
            point_definition="median",
            context_used=min(r.context_used for r in results),
            warnings=all_warnings,
            metadata={"merged_from_univariate": True},
        )


# For backward compatibility with existing app.py
def create_timesfm25_adapter() -> TimesFM25Adapter:
    """Factory function for TimesFM 2.5 adapter."""
    return TimesFM25Adapter()