"""Custom exceptions for the forecasting module."""

from __future__ import annotations

from typing import Any


class ForecastingError(Exception):
    """Base exception for all forecasting errors."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class ModelLoadError(ForecastingError):
    """Failed to load model weights or initialize runtime."""

    def __init__(
        self,
        model_key: str,
        message: str,
        *,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(f"[{model_key}] {message}", details=details)
        self.model_key = model_key
        self.original_error = original_error


class ModelNotAvailableError(ForecastingError):
    """Model package not installed or not importable."""

    def __init__(self, model_key: str, package_name: str, *, details: dict[str, Any] | None = None):
        super().__init__(
            f"Model '{model_key}' requires package '{package_name}' which is not installed. "
            f"Install with: pip install {package_name}",
            details=details,
        )
        self.model_key = model_key
        self.package_name = package_name


class CapabilityNotSupportedError(ForecastingError):
    """Request requires capability the model doesn't support."""

    def __init__(
        self,
        model_key: str,
        capability: str,
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            f"Model '{model_key}' does not support capability: {capability}",
            details=details,
        )
        self.model_key = model_key
        self.capability = capability


class LicensePolicyError(ForecastingError):
    """Model not allowed under current license policy."""

    def __init__(
        self,
        model_key: str,
        policy: str,
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            f"Model '{model_key}' is not allowed under license policy '{policy}'. "
            f"Change policy to 'research' to enable restricted models.",
            details=details,
        )
        self.model_key = model_key
        self.policy = policy


class LicenseAcknowledgmentRequiredError(ForecastingError):
    """Restricted model requires explicit license acknowledgment."""

    def __init__(
        self,
        model_key: str,
        notice: str,
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            f"Model '{model_key}' requires license acknowledgment:\n{notice}",
            details=details,
        )
        self.model_key = model_key
        self.notice = notice


class InferenceError(ForecastingError):
    """Error during forecast generation."""

    def __init__(
        self,
        model_key: str,
        message: str,
        *,
        request_info: dict[str, Any] | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(f"[{model_key}] Inference failed: {message}", details=details)
        self.model_key = model_key
        self.request_info = request_info or {}
        self.original_error = original_error


class QuantileError(ForecastingError):
    """Quantile validation or computation error."""

    def __init__(
        self,
        message: str,
        *,
        quantile_levels: list[float] | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details=details)
        self.quantile_levels = quantile_levels


class DataValidationError(ForecastingError):
    """Input data validation failure."""

    def __init__(
        self,
        message: str,
        *,
        column: str | None = None,
        row_indices: list[int] | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details=details)
        self.column = column
        self.row_indices = row_indices


class BacktestError(ForecastingError):
    """Backtesting execution error."""

    def __init__(
        self,
        message: str,
        *,
        window: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details=details)
        self.window = window


class CacheError(ForecastingError):
    """Cache key generation or invalidation error."""

    def __init__(
        self,
        message: str,
        *,
        cache_key: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details=details)
        self.cache_key = cache_key