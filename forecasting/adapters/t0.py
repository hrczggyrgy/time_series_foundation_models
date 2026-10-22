"""t0-gamma adapter — Apache-2.0 licensed, multivariate + covariates."""

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


class T0GammaAdapter(BaseAdapter):
    """Adapter for t0-gamma — Apache-2.0 weights, supports multivariate & covariates."""

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_key="t0_gamma",
            display_name="t0-gamma",
            checkpoint_id="theforecastingcompany/t0-gamma",
            checkpoint_revision="main",
            supports_cpu=True,
            supports_cuda=True,
            supports_mps=False,  # UNVERIFIED for torch; MLX runtime available separately
            supports_multivariate=True,
            supports_past_covariates=False,  # Not yet implemented - requires context+horizon alignment
            supports_known_future_covariates=False,  # Not yet implemented - requires context+horizon alignment
            supports_group_forecasting=True,
            native_quantile_levels=(
                0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99
            ),  # 21 quantiles: P01, P05-P95 in 5% increments, P99
            max_context_length=512,
            min_context_length=1,
            max_horizon=1024,
            compiles_context_horizon=False,
            supports_batch_inference=True,
            license_policy="permissive",
            license_notice="Apache-2.0 (weights and code). Requires tfc-t0[torch]>=0.6.0.",
        )

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        try:
            import torch
            from t0 import T0Forecaster
        except ImportError as e:
            raise ModelNotAvailableError(
                self.capabilities.model_key,
                "tfc-t0[torch]>=0.6.0",
                details={"import_error": str(e)},
            ) from e

        device = runtime_config.resolved_device()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        try:
            model = T0Forecaster.from_pretrained(
                self.capabilities.checkpoint_id
            ).eval()
            model.to(device)
        except Exception as e:
            raise ModelLoadError(
                self.capabilities.model_key,
                f"Failed to load checkpoint: {e}",
                original_error=e,
            ) from e

        return model

    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        import torch

        device = next(model.parameters()).device
        horizon = request.horizon
        n_targets = request.n_targets
        n_history = request.n_history

        # Prepare target history: [batch, context] or [batch, features, context]
        target_history = self._ensure_2d_targets(request.target_history)  # [targets, history]
        context_tensor = torch.from_numpy(target_history.astype(np.float32)).to(device)

        # t0 expects [batch, features, context] for multivariate
        if context_tensor.ndim == 2:
            context_tensor = context_tensor.unsqueeze(1)  # [targets, 1, history]

        # Prepare covariates if provided
        # t0-gamma expects future_covariates to include both historical and future known covariates
        future_covariates = None
        
        # Build combined covariates: past_covariates + known_covariates
        # t0 expects [batch, features, context + horizon]
        covariate_parts = []
        
        if request.past_covariates is not None:
            covariate_parts.append(("past", request.past_covariates))
        
        if request.known_covariates_history is not None:
            covariate_parts.append(("known_hist", request.known_covariates_history))
        
        if request.known_covariates_future is not None:
            covariate_parts.append(("known_future", request.known_covariates_future))
        
        if covariate_parts:
            # Build combined covariate tensor
            # First, align all historical covariates to context length
            context_len = context_tensor.shape[-1]
            combined_hist = []
            combined_future = []
            
            for name, cov in covariate_parts:
                if name in ("past", "known_hist"):
                    # Historical covariate - use up to context_len
                    if cov.shape[-1] >= context_len:
                        combined_hist.append(cov[:, -context_len:])
                    else:
                        # Pad if shorter (shouldn't happen with proper prep)
                        pad_width = context_len - cov.shape[-1]
                        combined_hist.append(np.pad(cov, ((0, 0), (pad_width, 0)), mode='edge'))
                elif name == "known_future":
                    combined_future.append(cov)
            
            if combined_hist:
                hist_combined = np.concatenate(combined_hist, axis=0)  # [total_features, context]
                if combined_future:
                    future_combined = np.concatenate(combined_future, axis=0)
                    # Concatenate historical + future for full covariate tensor
                    full_combined = np.concatenate([hist_combined, future_combined], axis=1)
                else:
                    full_combined = hist_combined
            else:
                # Only future covariates provided
                full_combined = np.concatenate(combined_future, axis=0)
            
            future_covariates = torch.from_numpy(full_combined.astype(np.float32)).to(device)
            if future_covariates.ndim == 2:
                future_covariates = future_covariates.unsqueeze(0).repeat(n_targets, 1, 1)

        # Request quantile levels
        quantile_levels = np.array(request.quantile_levels, dtype=np.float32)

        warnings = []

        with torch.no_grad():
            out = model.predict(
                context_tensor,
                horizon=horizon,
                quantile_levels=quantile_levels,
                future_covariates=future_covariates,
            )

        # out.quantiles: [batch, horizon, n_quantiles] or [batch, features, horizon, n_quantiles]
        # out.median: [batch, horizon] or [batch, features, horizon]
        quantiles = out.quantiles.cpu().numpy()
        point = out.median.cpu().numpy()

        # Handle shapes
        if quantiles.ndim == 4:
            # [batch, features, horizon, quantiles] -> [targets, horizon, quantiles]
            quantiles = quantiles.transpose(0, 2, 3, 1).reshape(n_targets, horizon, -1)
            point = point.transpose(0, 2, 1).reshape(n_targets, horizon)
        elif quantiles.ndim == 3:
            # [batch, horizon, quantiles] -> [1, horizon, quantiles] for univariate
            if n_targets == 1:
                quantiles = quantiles.reshape(1, horizon, -1)
                point = point.reshape(1, horizon)
            else:
                # Multivariate but quantiles missing feature dim - shouldn't happen
                pass

        # Validate quantile ordering
        quantiles, q_warnings = self._validate_quantile_ordering(
            quantiles, quantile_levels
        )
        warnings.extend(q_warnings)

        # T0 returns quantiles at the exact requested levels - no interpolation needed
        # The model handles interpolation internally based on its trained quantiles
        requested_levels = np.array(request.quantile_levels)

        return self._build_result(
            point=point,
            quantiles=quantiles,
            quantile_levels=requested_levels,
            request=request,
            point_definition="median",
            context_used=min(n_history, self.capabilities.max_context_length or n_history),
            warnings=warnings,
            metadata={
                "device": str(device),
                "native_quantile_levels": self.capabilities.native_quantile_levels,
            },
        )


def create_t0_gamma_adapter() -> T0GammaAdapter:
    return T0GammaAdapter()