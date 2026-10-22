"""Unit tests for T0-Gamma adapter quantile handling."""

import numpy as np
import pytest

from forecasting.adapters.t0 import T0GammaAdapter


def test_t0_native_quantile_levels():
    """Test that native quantile levels match official documentation."""
    adapter = T0GammaAdapter()
    caps = adapter.capabilities

    # Updated T0 levels: P01, P025, P05-P95 in 5% increments, P975, P99
    expected = (
        0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
        0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
    )

    assert caps.native_quantile_levels == expected, \
        f"Native quantile levels should match official T0 documentation"


def test_t0_covariate_support_enabled():
    """Test that covariate support is now enabled."""
    adapter = T0GammaAdapter()
    caps = adapter.capabilities

    assert caps.supports_past_covariates, \
        "Past covariates should be enabled"
    assert caps.supports_known_future_covariates, \
        "Known future covariates should be enabled"
    assert caps.supports_multivariate, \
        "Multivariate should be enabled"


def test_t0_has_interpolation_method():
    """Test that T0 adapter now has interpolation method for non-native quantile levels."""
    adapter = T0GammaAdapter()

    # The _interpolate_quantiles method should exist for interpolation
    assert hasattr(adapter, '_interpolate_quantiles'), \
        "T0 adapter should have _interpolate_quantiles method for quantile interpolation"