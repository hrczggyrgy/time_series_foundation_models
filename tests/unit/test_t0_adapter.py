"""Unit tests for T0-Gamma adapter quantile handling."""

import numpy as np
import pytest

from forecasting.adapters.t0 import T0GammaAdapter


def test_t0_native_quantile_levels():
    """Test that native quantile levels match official documentation."""
    adapter = T0GammaAdapter()
    caps = adapter.capabilities

    # Official T0 levels: P01, P05-P95 in 5% increments, P99
    expected = (
        0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99
    )

    assert caps.native_quantile_levels == expected, \
        f"Native quantile levels should match official T0 documentation"


def test_t0_covariate_support_disabled():
    """Test that covariate support is marked as False until implemented."""
    adapter = T0GammaAdapter()
    caps = adapter.capabilities

    assert not caps.supports_past_covariates, \
        "Past covariates should be disabled until context+horizon alignment is implemented"
    assert not caps.supports_known_future_covariates, \
        "Known future covariates should be disabled until context+horizon alignment is implemented"


def test_t0_no_interpolation_method():
    """Test that T0 adapter no longer has interpolation method (it was removed)."""
    adapter = T0GammaAdapter()

    # The _interpolate_quantiles method should not exist
    assert not hasattr(adapter, '_interpolate_quantiles'), \
        "T0 adapter should not have _interpolate_quantiles method - T0 handles interpolation internally"
