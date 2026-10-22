"""Evaluation metrics for forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PointMetrics:
    """Point forecast accuracy metrics."""

    mae: float
    rmse: float
    mape: float | None
    bias: float
    mase: float | None
    mase_denominator: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape,
            "Bias": self.bias,
            "MASE": self.mase,
            "MASE_Denominator": self.mase_denominator,
        }


@dataclass(frozen=True)
class ProbabilisticMetrics:
    """Probabilistic forecast accuracy metrics."""

    interval_coverage: dict[float, float]  # coverage_pct -> actual_coverage
    interval_width: dict[float, float]     # coverage_pct -> mean_width
    mean_pinball_loss: float | None
    pinball_losses: dict[float, float]     # quantile -> loss
    crps: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "Interval_Coverage": self.interval_coverage,
            "Interval_Width": self.interval_width,
            "Mean_Pinball_Loss": self.mean_pinball_loss,
            "Pinball_Losses": self.pinball_losses,
            "CRPS": self.crps,
        }


def compute_point_metrics(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    training_actuals: np.ndarray | None = None,
    seasonal_period: int | None = None,
) -> PointMetrics:
    """Compute point forecast metrics.

    Args:
        actuals: Ground truth values [horizon] or [targets, horizon]
        forecasts: Point forecasts [horizon] or [targets, horizon]
        training_actuals: Training history for MASE denominator
        seasonal_period: Seasonal period for MASE

    Returns:
        PointMetrics with all computed metrics.
    """
    actuals = np.asarray(actuals, dtype=float).ravel()
    forecasts = np.asarray(forecasts, dtype=float).ravel()

    if len(actuals) != len(forecasts):
        raise ValueError("actuals and forecasts must have same length")

    # Remove NaN pairs
    mask = np.isfinite(actuals) & np.isfinite(forecasts)
    actuals = actuals[mask]
    forecasts = forecasts[mask]

    if len(actuals) == 0:
        return PointMetrics(
            mae=math.nan, rmse=math.nan, mape=math.nan, bias=math.nan, mase=math.nan, mase_denominator=math.nan
        )

    errors = actuals - forecasts
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))

    # MAPE (handle zeros)
    denom = np.where(np.abs(actuals) < 1e-12, np.nan, np.abs(actuals))
    ape = np.abs(errors / denom) * 100
    mape_series = ape[np.isfinite(ape)]
    mape = float(mape_series.mean()) if len(mape_series) > 0 else math.nan

    # MASE
    mase = math.nan
    mase_denom = math.nan
    if training_actuals is not None and len(training_actuals) > 1:
        train = np.asarray(training_actuals, dtype=float).ravel()
        if seasonal_period and seasonal_period < len(train):
            # Seasonal naive denominator
            seasonal_errors = np.abs(train[seasonal_period:] - train[:-seasonal_period])
            mase_denom = float(np.mean(seasonal_errors))
        else:
            # Naive denominator (random walk)
            naive_errors = np.abs(np.diff(train))
            mase_denom = float(np.mean(naive_errors))

        if mase_denom > 1e-12:
            mase = mae / mase_denom
        else:
            mase = math.nan

    return PointMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        bias=bias,
        mase=mase,
        mase_denominator=mase_denom if not math.isnan(mase_denom) else None,
    )


def compute_pinball_loss(actuals: np.ndarray, quantiles: np.ndarray, level: float) -> float:
    """Compute pinball loss for a single quantile level."""
    actuals = np.asarray(actuals, dtype=float).ravel()
    quantiles = np.asarray(quantiles, dtype=float).ravel()

    if len(actuals) != len(quantiles):
        raise ValueError("actuals and quantiles must have same length")

    mask = np.isfinite(actuals) & np.isfinite(quantiles)
    actuals = actuals[mask]
    quantiles = quantiles[mask]

    if len(actuals) == 0:
        return math.nan

    diff = actuals - quantiles
    loss = np.where(diff >= 0, level * diff, (level - 1) * diff)
    return float(np.mean(loss))


def compute_interval_metrics(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage: float,
) -> tuple[float, float]:
    """Compute interval coverage and mean width."""
    actuals = np.asarray(actuals, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()

    mask = np.isfinite(actuals) & np.isfinite(lower) & np.isfinite(upper)
    actuals = actuals[mask]
    lower = lower[mask]
    upper = upper[mask]

    if len(actuals) == 0:
        return math.nan, math.nan

    coverage_actual = float(np.mean((actuals >= lower) & (actuals <= upper)) * 100)
    mean_width = float(np.mean(upper - lower))

    return coverage_actual, mean_width


def compute_crps(actuals: np.ndarray, quantiles: np.ndarray, quantile_levels: np.ndarray) -> float:
    """Compute CRPS via numerical integration over quantile levels.

    Uses the identity: CRPS = 2 * ∫_0^1 (τ - 1_{y ≤ q_τ}) * (q_τ - y) dτ
    Approximated via trapezoidal rule over provided quantile levels.

    Note: This integrates only over the provided quantile levels (typically 0.1-0.9).
    The tails (0 to q_min and q_max to 1) are not included unless quantile_levels
    explicitly covers them. For standard decile forecasts, this is an approximation.

    Args:
        actuals: Ground truth [horizon] or [targets, horizon]
        quantiles: Quantile forecasts [horizon, n_levels] or [targets, horizon, n_levels]
        quantile_levels: Probability levels for quantiles [n_levels] (must be strictly increasing)

    Returns:
        Mean CRPS across all valid forecast points.
    """
    actuals = np.asarray(actuals, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    quantile_levels = np.asarray(quantile_levels, dtype=float)

    if quantile_levels.ndim != 1 or quantile_levels.size < 2:
        return math.nan
    if not np.all(np.diff(quantile_levels) > 0):
        return math.nan  # Quantile levels must be strictly increasing

    # Ensure 3D: [targets, horizon, n_levels]
    if quantiles.ndim == 2:
        quantiles = quantiles.reshape(1, *quantiles.shape)
    if actuals.ndim == 1:
        actuals = actuals.reshape(1, -1)

    n_targets, horizon, n_levels = quantiles.shape
    if n_levels != len(quantile_levels):
        return math.nan
    if actuals.shape != (n_targets, horizon):
        return math.nan

    # Trapezoidal integration for each target/horizon point
    crps_values = []

    for t in range(n_targets):
        for h in range(horizon):
            y = actuals[t, h]
            q_vals = quantiles[t, h]

            if not np.isfinite(y) or not np.all(np.isfinite(q_vals)):
                continue

            # Pinball loss at each quantile level
            # Standard formula: if y >= q_τ: τ * (y - q_τ), else: (τ - 1) * (y - q_τ)
            diff = y - q_vals
            pinball = np.where(diff >= 0,
                             quantile_levels * diff,
                             (quantile_levels - 1) * diff)

            # CRPS = 2 * integral of pinball loss over τ in [0,1]
            # Trapezoidal rule
            crps_val = 2 * np.trapz(pinball, quantile_levels)
            crps_values.append(crps_val)

    return float(np.mean(crps_values)) if crps_values else math.nan


def compute_all_probabilistic_metrics(
    actuals: np.ndarray,
    quantiles: np.ndarray,  # [targets, horizon, n_levels]
    quantile_levels: np.ndarray,
    point_forecasts: np.ndarray | None = None,
    coverage_levels: tuple[float, ...] = (0.5, 0.6, 0.8, 0.9),
) -> ProbabilisticMetrics:
    """Compute all probabilistic metrics for a forecast."""
    actuals = np.asarray(actuals, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    quantile_levels = np.asarray(quantile_levels, dtype=float)

    # Ensure 3D
    if quantiles.ndim == 2:
        quantiles = quantiles.reshape(1, *quantiles.shape)
    if actuals.ndim == 1:
        actuals = actuals.reshape(1, -1)

    n_targets, horizon, n_levels = quantiles.shape

    # Pinball losses at each requested quantile level
    pinball_losses = {}
    for i, q in enumerate(quantile_levels):
        loss = compute_pinball_loss(
            actuals.ravel(), quantiles[:, :, i].ravel(), q
        )
        pinball_losses[float(q)] = loss

    mean_pinball = float(np.mean([v for v in pinball_losses.values() if not math.isnan(v)])) \
        if any(not math.isnan(v) for v in pinball_losses.values()) else math.nan

    # CRPS
    crps = compute_crps(actuals, quantiles, quantile_levels)

    # Interval coverage and width
    interval_coverage = {}
    interval_width = {}
    for cov in coverage_levels:
        alpha = (1 - cov) / 2
        lower_q, upper_q = alpha, 1 - alpha

        # Interpolate bounds
        lower = _interpolate_quantile(quantiles, quantile_levels, lower_q)
        upper = _interpolate_quantile(quantiles, quantile_levels, upper_q)

        cov_actual, width = compute_interval_metrics(actuals, lower, upper, cov)
        interval_coverage[cov] = cov_actual
        interval_width[cov] = width

    return ProbabilisticMetrics(
        interval_coverage=interval_coverage,
        interval_width=interval_width,
        mean_pinball_loss=mean_pinball,
        pinball_losses=pinball_losses,
        crps=crps,
    )


def _interpolate_quantile(
    quantiles: np.ndarray, levels: np.ndarray, target_q: float
) -> np.ndarray:
    """Interpolate quantile at target probability."""
    if target_q <= levels[0]:
        return quantiles[:, :, 0]
    if target_q >= levels[-1]:
        return quantiles[:, :, -1]

    idx = np.searchsorted(levels, target_q)
    q_low, q_high = levels[idx - 1], levels[idx]
    v_low = quantiles[:, :, idx - 1]
    v_high = quantiles[:, :, idx]
    weight = (target_q - q_low) / (q_high - q_low)
    return v_low + weight * (v_high - v_low)


def aggregate_metrics_across_windows(
    window_metrics: list[PointMetrics],
) -> PointMetrics:
    """Aggregate point metrics across backtest windows (weighted by sample size)."""
    # This is a simplified aggregation; in practice, weight by n_obs per window
    if not window_metrics:
        return PointMetrics(math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    return PointMetrics(
        mae=float(np.mean([m.mae for m in window_metrics if not math.isnan(m.mae)])),
        rmse=float(np.mean([m.rmse for m in window_metrics if not math.isnan(m.rmse)])),
        mape=float(np.nanmean([m.mape for m in window_metrics])) if any(not math.isnan(m.mape) for m in window_metrics) else math.nan,
        bias=float(np.mean([m.bias for m in window_metrics if not math.isnan(m.bias)])),
        mase=float(np.nanmean([m.mase for m in window_metrics])) if any(not math.isnan(m.mase) for m in window_metrics) else math.nan,
        mase_denominator=None,  # Not meaningful aggregated
    )