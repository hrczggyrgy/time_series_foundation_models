"""Registry of available forecasting models."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from forecasting.contracts import ForecastAdapter, ModelCapabilities, RuntimeConfig
from forecasting.exceptions import ModelNotAvailableError


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Registry entry for a model adapter."""

    model_key: str
    display_name: str
    adapter_class: str  # Fully qualified class name for lazy loading
    capabilities: ModelCapabilities
    # Optional: minimum package version, extra dependencies
    min_package_version: str = ""
    extra_deps: tuple[str, ...] = ()
    required_module: str = ""
    deployment_profiles: tuple[str, ...] = ()
    experimental: bool = False  # If True, adapter not yet fully implemented


class ModelRegistry:
    """Central registry of all available model adapters."""

    def __init__(self):
        self._entries: dict[str, ModelRegistryEntry] = {}
        self._instances: dict[str, ForecastAdapter] = {}

        # Register built-in models
        self._register_builtin()

    def _register_builtin(self) -> None:
        """Register all known model adapters."""

        # TimesFM 2.5 — Permissive (Apache-2.0)
        from forecasting.contracts import ModelCapabilities

        self.register(
            ModelRegistryEntry(
                model_key="timesfm_2_5",
                display_name="TimesFM 2.5 (200M)",
                adapter_class="forecasting.adapters.timesfm25.TimesFM25Adapter",
                capabilities=ModelCapabilities(
                    model_key="timesfm_2_5",
                    display_name="TimesFM 2.5 (200M)",
                    checkpoint_id="google/timesfm-2.5-200m-pytorch",
                    checkpoint_revision="main",
                    supports_cpu=True,
                    supports_cuda=True,
                    supports_mps=False,  # UNVERIFIED
                    supports_multivariate=False,
                    supports_past_covariates=False,
                    supports_known_future_covariates=False,
                    native_quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    max_context_length=16384,
                    min_context_length=32,
                    max_horizon=512,
                    compiles_context_horizon=True,
                    supports_batch_inference=True,
                    license_notice="Apache-2.0 (weights and code)",
                ),
                min_package_version="timesfm[torch]==2.0.2",
                required_module="timesfm",
                deployment_profiles=("timesfm25",),
            )
        )

        # TimesFM 3.0 — separate deployment profile (timesfm 2.5 and 3.0 conflict)
        self.register(
            ModelRegistryEntry(
                model_key="timesfm_3",
                display_name="TimesFM 3.0 (330M)",
                adapter_class="forecasting.adapters.timesfm3.TimesFM3Adapter",
                capabilities=ModelCapabilities(
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
                    min_context_length=32,
                    max_horizon=1024,
                    compiles_context_horizon=False,
                    supports_batch_inference=True,
                    license_notice=(
                        "TimesFM Non-Commercial License v1.0: "
                        "Weights restricted to non-commercial, non-production use only. "
                        "Commercial/production use requires authorized Google Cloud "
                        "services (BigQuery ML). "
                        "Source code is Apache-2.0. Install in the TimesFM 3.0 profile."
                    ),
                ),
                min_package_version="timesfm[torch]>=3.0.0",
                required_module="timesfm3",
                deployment_profiles=("timesfm3",),
            )
        )

        self.register(
            ModelRegistryEntry(
                model_key="tirex_2",
                display_name="TiRex-2",
                adapter_class="forecasting.adapters.tirex2.TiRex2Adapter",
                capabilities=ModelCapabilities(
                    model_key="tirex_2",
                    display_name="TiRex-2",
                    checkpoint_id="NX-AI/TiRex-2",
                    supports_cpu=True,
                    supports_cuda=True,
                    supports_multivariate=True,
                    supports_past_covariates=True,
                    supports_known_future_covariates=True,
                    native_quantile_levels=(
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
                    ),
                    max_context_length=32768,
                    max_horizon=1024,
                    supports_batch_inference=True,
                    license_notice="Apache-2.0",
                ),
                min_package_version="tirex-2>=0.4.0",
                required_module="tirex2",
                deployment_profiles=("timesfm25", "timesfm3"),
            )
        )

        self.register(
            ModelRegistryEntry(
                model_key="chronos",
                display_name="Chronos-T5 Small",
                adapter_class="forecasting.adapters.chronos.ChronosAdapter",
                capabilities=ModelCapabilities(
                    model_key="chronos",
                    display_name="Chronos-T5 Small",
                    checkpoint_id="amazon/chronos-t5-small",
                    supports_cpu=True,
                    supports_cuda=True,
                    supports_multivariate=True,
                    native_quantile_levels=(
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
                    ),
                    max_context_length=512,
                    supports_batch_inference=True,
                    license_notice="Apache-2.0",
                ),
                min_package_version="chronos-forecasting>=1.4.0,<3.0.0",
                required_module="chronos",
                deployment_profiles=("timesfm25", "timesfm3", "moirai"),
            )
        )

        self.register(
            ModelRegistryEntry(
                model_key="moirai",
                display_name="Moirai 1.1 (Small)",
                adapter_class="forecasting.adapters.moirai.MoiraiAdapter",
                capabilities=ModelCapabilities(
                    model_key="moirai",
                    display_name="Moirai 1.1 (Small)",
                    checkpoint_id="Salesforce/moirai-1.1-R-small",
                    supports_cpu=True,
                    supports_cuda=True,
                    native_quantile_levels=(
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
                    ),
                    max_context_length=2000,
                    supports_batch_inference=False,
                    license_notice=(
                        "Review the Salesforce model-card license for weights; "
                        "Uni2TS source code is Apache-2.0."
                    ),
                ),
                min_package_version="uni2ts",
                required_module="uni2ts",
                deployment_profiles=("moirai",),
            )
        )

        # Timestral 209M — Restricted (Modified MIT with $20M revenue cap)
        self.register(
            ModelRegistryEntry(
                model_key="timestral_209m",
                display_name="Timestral 209M — Restricted License",
                adapter_class="forecasting.adapters.timestral.TimestralAdapter",
                capabilities=ModelCapabilities(
                    model_key="timestral_209m",
                    display_name="Timestral 209M",
                    checkpoint_id="mistralai/Timestral-209M",
                    checkpoint_revision="main",
                    supports_cpu=True,
                    supports_cuda=True,
                    supports_mps=False,  # UNVERIFIED
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
                    license_notice=(
                        "Modified MIT License with revenue restriction: "
                        "You are not authorized to use this model if your company's (or employer's) "
                        "global consolidated monthly revenue exceeds $20 million. "
                        "This applies to the model, derivatives, modifications, and combined works. "
                        "Contact sales@mistral.ai for commercial license. "
                        "Inference code is Apache-2.0."
                    ),
                ),
                min_package_version="timestral>=0.1.0",
                required_module="timestral",
                deployment_profiles=("timesfm25", "timesfm3"),
            )
        )

        # Timestral 28M — Restricted (Modified MIT with $20M revenue cap)
        self.register(
            ModelRegistryEntry(
                model_key="timestral_28m",
                display_name="Timestral 28M — Restricted License",
                adapter_class="forecasting.adapters.timestral.TimestralAdapter",
                capabilities=ModelCapabilities(
                    model_key="timestral_28m",
                    display_name="Timestral 28M",
                    checkpoint_id="mistralai/Timestral-28M",
                    checkpoint_revision="main",
                    supports_cpu=True,
                    supports_cuda=True,
                    supports_mps=False,  # UNVERIFIED
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
                    license_notice=(
                        "Modified MIT License with revenue restriction: "
                        "You are not authorized to use this model if your company's (or employer's) "
                        "global consolidated monthly revenue exceeds $20 million. "
                        "This applies to the model, derivatives, modifications, and combined works. "
                        "Contact sales@mistral.ai for commercial license. "
                        "Inference code is Apache-2.0."
                    ),
                ),
                min_package_version="timestral>=0.1.0",
                required_module="timestral",
                deployment_profiles=("timesfm25", "timesfm3"),
            )
        )

        # Timestral 123M — Restricted (Modified MIT with $20M revenue cap)
        self.register(
            ModelRegistryEntry(
                model_key="timestral_123m",
                display_name="Timestral 123M — Restricted License",
                adapter_class="forecasting.adapters.timestral.TimestralAdapter",
                capabilities=ModelCapabilities(
                    model_key="timestral_123m",
                    display_name="Timestral 123M",
                    checkpoint_id="mistralai/Timestral-123M",
                    checkpoint_revision="main",
                    supports_cpu=True,
                    supports_cuda=True,
                    supports_mps=False,  # UNVERIFIED
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
                    license_notice=(
                        "Modified MIT License with revenue restriction: "
                        "You are not authorized to use this model if your company's (or employer's) "
                        "global consolidated monthly revenue exceeds $20 million. "
                        "This applies to the model, derivatives, modifications, and combined works. "
                        "Contact sales@mistral.ai for commercial license. "
                        "Inference code is Apache-2.0."
                    ),
                ),
                min_package_version="timestral>=0.1.0",
                required_module="timestral",
                deployment_profiles=("timesfm25", "timesfm3"),
            )
        )

        # Timestral 5M — Restricted (Modified MIT with $20M revenue cap)
        self.register(
            ModelRegistryEntry(
                model_key="timestral_5m",
                display_name="Timestral 5M — Restricted License",
                adapter_class="forecasting.adapters.timestral.TimestralAdapter",
                capabilities=ModelCapabilities(
                    model_key="timestral_5m",
                    display_name="Timestral 5M",
                    checkpoint_id="mistralai/Timestral-5M",
                    checkpoint_revision="main",
                    supports_cpu=True,
                    supports_cuda=True,
                    supports_mps=False,  # UNVERIFIED
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
                    license_notice=(
                        "Modified MIT License with revenue restriction: "
                        "You are not authorized to use this model if your company's (or employer's) "
                        "global consolidated monthly revenue exceeds $20 million. "
                        "This applies to the model, derivatives, modifications, and combined works. "
                        "Contact sales@mistral.ai for commercial license. "
                        "Inference code is Apache-2.0."
                    ),
                ),
                min_package_version="timestral>=0.1.0",
                required_module="timestral",
                deployment_profiles=("timesfm25", "timesfm3"),
            )
        )

        # t0-gamma — Permissive (Apache-2.0)
        self.register(
            ModelRegistryEntry(
                model_key="t0_gamma",
                display_name="t0-gamma (Apache-2.0)",
                adapter_class="forecasting.adapters.t0.T0GammaAdapter",
                capabilities=ModelCapabilities(
                    model_key="t0_gamma",
                    display_name="t0-gamma",
                    checkpoint_id="theforecastingcompany/t0-gamma",
                    checkpoint_revision="main",
                    supports_cpu=True,
                    supports_cuda=True,
                    supports_mps=False,  # UNVERIFIED for torch; MLX available separately
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
                    license_notice="Apache-2.0 (weights and code). Requires tfc-t0[torch]>=0.6.0.",
                ),
                min_package_version="tfc-t0[torch]>=0.6.0",
                required_module="t0",
                deployment_profiles=("timesfm25", "timesfm3"),
            )
        )

        # STL decomposition baseline — deterministic and dependency-light
        self.register(
            ModelRegistryEntry(
                model_key="stl",
                display_name="STL Decomposition",
                adapter_class="forecasting.adapters.stl.STLAdapter",
                capabilities=ModelCapabilities(
                    model_key="stl",
                    display_name="STL Decomposition",
                    checkpoint_id="",
                    checkpoint_revision="",
                    supports_cpu=True,
                    supports_cuda=False,
                    supports_mps=False,
                    supports_multivariate=True,
                    supports_past_covariates=False,
                    supports_known_future_covariates=False,
                    native_quantile_levels=(),
                    max_context_length=None,
                    min_context_length=4,
                    max_horizon=None,
                    compiles_context_horizon=False,
                    supports_batch_inference=True,
                    license_notice="No model weights — statsmodels STL decomposition baseline.",
                ),
            )
        )

    def register(self, entry: ModelRegistryEntry) -> None:
        """Register a model adapter entry."""
        self._entries[entry.model_key] = entry

    def get_entry(self, model_key: str) -> ModelRegistryEntry | None:
        """Get registry entry by key."""
        return self._entries.get(model_key)

    def list_available(self, include_experimental: bool = False) -> list[ModelRegistryEntry]:
        """List registered models.

        Args:
            include_experimental: If True, include experimental/unimplemented models

        Returns:
            List of available model registry entries
        """
        available = []
        active_profile = os.getenv("TSFM_MODEL_PROFILE")
        for entry in self._entries.values():
            if (
                entry.experimental or entry.capabilities.experimental
            ) and not include_experimental:
                continue
            if (
                active_profile
                and entry.deployment_profiles
                and active_profile not in entry.deployment_profiles
            ):
                continue
            if entry.required_module and importlib.util.find_spec(entry.required_module) is None:
                continue
            available.append(entry)
        return available

    def get_adapter(
        self,
        model_key: str,
        runtime_config: RuntimeConfig,
    ) -> ForecastAdapter:
        """Get or create adapter instance for a model.

        Args:
            model_key: Model identifier
            runtime_config: Runtime configuration
        """
        entry = self._entries.get(model_key)
        if not entry:
            raise ModelNotAvailableError(model_key, "unknown model key")

        # Create runtime config with model_key for adapters that need it (e.g., Timestral)
        adapted_runtime = runtime_config
        if model_key.startswith("timestral_"):
            extra = dict(runtime_config.extra)
            extra["model_key"] = model_key
            adapted_runtime = RuntimeConfig(
                device=runtime_config.device,
                dtype=runtime_config.dtype,
                compile=runtime_config.compile,
                compile_mode=runtime_config.compile_mode,
                per_core_batch_size=runtime_config.per_core_batch_size,
                extra=extra,
            )

        # Return cached instance if compatible
        cache_key = self._cache_key(model_key, adapted_runtime)
        if cache_key in self._instances:
            return self._instances[cache_key]

        # Lazy load adapter class
        adapter = self._load_adapter_class(entry.adapter_class)
        adapter.load(adapted_runtime)
        self._instances[cache_key] = adapter
        return adapter

    def _cache_key(self, model_key: str, runtime_config: RuntimeConfig) -> str:
        """Generate cache key including all config that affects model behavior."""
        parts = [
            model_key,
            runtime_config.resolved_device(),
            runtime_config.dtype,
            str(runtime_config.compile),
            runtime_config.compile_mode,
            str(runtime_config.per_core_batch_size),
        ]
        # Include extra config if any
        if runtime_config.extra:
            extra_str = "|".join(f"{k}={v}" for k, v in sorted(runtime_config.extra.items()))
            parts.append(extra_str)
        return ":".join(parts)

    def _load_adapter_class(self, class_path: str) -> ForecastAdapter:
        """Dynamically import and instantiate adapter class."""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            adapter_class = getattr(module, class_name)
            return adapter_class()
        except ImportError as e:
            raise ModelNotAvailableError(
                class_path,
                module_path,
                details={"import_error": str(e)},
            ) from e
        except AttributeError as e:
            raise ModelNotAvailableError(
                class_path,
                class_name,
                details={"attribute_error": str(e)},
            ) from e

    def clear_cache(self) -> None:
        """Clear all cached adapter instances."""
        for adapter in self._instances.values():
            try:
                adapter.close()
            except Exception:
                pass
        self._instances.clear()


# Global registry instance (for model caching only)
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry (shared model cache, no policy)."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_registry() -> None:
    """Reset global registry (mainly for testing)."""
    global _registry
    if _registry:
        _registry.clear_cache()
    _registry = None