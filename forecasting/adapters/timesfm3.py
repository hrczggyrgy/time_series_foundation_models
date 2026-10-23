"""TimesFM 3.0 adapter stub — Not yet implemented (Non-Commercial License)."""

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
    """Adapter for TimesFM 3.0 (330M) — Non-Commercial License.

    STUB: Not yet implemented. TimesFM 3.0 requires separate installation
    of timesfm[torch]>=3.0.0 and has a different API (TimesFM3Evaluator).
    """

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_key="timesfm_3",
            display_name="TimesFM 3.0 (330M)",
            checkpoint_id="google/timesfm-3.0-pytorch",
            checkpoint_revision="main",
            supports_cpu=True,
            supports_cuda=True,
            supports_mps=False,  # UNVERIFIED
            supports_multivariate=True,
            supports_past_covariates=True,
            supports_known_future_covariates=True,
            native_quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            max_context_length=32768,
            min_context_length=32,
            max_horizon=1024,
            compiles_context_horizon=False,
            supports_batch_inference=True,
            license_notice=(
                "TimesFM Non-Commercial License v1.0: "
                "Weights restricted to non-commercial, non-production use only. "
                "Commercial/production use requires authorized Google Cloud services (BigQuery ML). "
                "Source code is Apache-2.0."
            ),
        )

    def _load_model(self, runtime_config: RuntimeConfig) -> Any:
        raise ModelNotAvailableError(
            self.capabilities.model_key,
            "timesfm[torch]>=3.0.0",
            details={
                "reason": "TimesFM 3.0 adapter not yet implemented. "
                "Install timesfm[torch]>=3.0.0 and implement the adapter using TimesFM3Evaluator API."
            },
        )

    def _predict_impl(self, request: ForecastRequest, model: Any) -> ForecastResult:
        raise NotImplementedError("TimesFM 3.0 adapter not yet implemented")


def create_timesfm3_adapter() -> TimesFM3Adapter:
    return TimesFM3Adapter()