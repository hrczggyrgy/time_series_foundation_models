"""Backtesting engine with leakage-free splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from forecasting.contracts import ForecastAdapter, ForecastRequest, ForecastResult
from forecasting.data.parsing import infer_frequency_safe
from forecasting.evaluation.metrics import (
    PointMetrics,
    ProbabilisticMetrics,
    compute_all_probabilistic_metrics,
    compute_point_metrics,
)
from forecasting.exceptions import BacktestError


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for rolling-origin backtesting."""

    horizon: int
    n_windows: int
    step_size: int | None = None  # None = horizon (non-overlapping)
    expanding_window: bool = True  # True = expanding, False = sliding
    min_training_length: int = 32
    max_training_length: int | None = None  # None = use all available


@dataclass(frozen=True)
class BacktestWindow:
    """Single backtest window result."""

    window_id: int
    train_end_idx: int
    train_end_date: pd.Timestamp
    test_dates: pd.DatetimeIndex
    test_actuals: np.ndarray  # [targets, horizon] or [horizon]
    point_forecasts: np.ndarray  # [targets, horizon] or [horizon]
    quantile_forecasts: np.ndarray | None  # [targets, horizon, n_levels] or [horizon, n_levels]
    quantile_levels: np.ndarray
    point_metrics: PointMetrics
    probabilistic_metrics: ProbabilisticMetrics | None
    model_key: str
    context_used: int


@dataclass(frozen=True)
class BacktestResult:
    """Complete backtest result across all windows."""

    windows: list[BacktestWindow]
    aggregate_point_metrics: PointMetrics
    aggregate_probabilistic_metrics: ProbabilisticMetrics | None
    config: BacktestConfig
    model_key: str
    target_names: tuple[str, ...]


def generate_backtest_splits(
    n_obs: int,
    horizon: int,
    n_windows: int,
    step_size: int | None = None,
    expanding: bool = True,
    min_train: int = 32,
    max_train: int | None = None,
) -> list[tuple[int, int]]:
    """Generate (train_end_idx, test_end_idx) pairs for rolling-origin backtest.

    Returns list of (train_end, test_end) where:
    - train_end: index of last training observation (exclusive for test)
    - test_end: index of last test observation (exclusive)
    - test window is [train_end, test_end) of length horizon
    """
    if step_size is None:
        step_size = horizon

    need = min_train + horizon
    if n_obs < need:
        return []

    n_windows = min(n_windows, max(1, n_obs - need + 1))
    splits = []

    for w in range(n_windows):
        # Work backwards from the end
        test_end = n_obs - w * step_size
        train_end = test_end - horizon

        if train_end < min_train:
            break

        if max_train and train_end > max_train:
            if expanding:
                train_end = min(train_end, max_train)
            else:
                train_end = max_train

        if train_end >= min_train:
            splits.append((train_end, test_end))

    # Reverse to chronological order
    splits.reverse()
    return splits


def run_backtest(
    adapter: ForecastAdapter,
    target_history: np.ndarray,  # [targets, history] or [history]
    history_index: pd.DatetimeIndex,
    target_names: tuple[str, ...],
    config: BacktestConfig,
    past_covariates: np.ndarray | None = None,
    known_covariates_history: np.ndarray | None = None,
    known_covariates_future: np.ndarray | None = None,
    quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> BacktestResult:
    """Run rolling-origin backtest with leakage-free splits.

    For each window:
    1. Select training history up to train_end
    2. Fit any data-dependent preprocessing on training data ONLY
    3. Generate forecast for horizon
    4. Align with held-out actuals
    5. Score only observed targets
    """
    # Ensure 2D target_history
    if target_history.ndim == 1:
        target_history = target_history.reshape(1, -1)

    n_targets, n_obs = target_history.shape

    # Validate covariates
    if past_covariates is not None and past_covariates.shape[1] != n_obs:
        raise BacktestError("past_covariates history length mismatch", details={"expected": n_obs, "got": past_covariates.shape[1]})
    if known_covariates_history is not None and known_covariates_history.shape[1] != n_obs:
        raise BacktestError("known_covariates_history length mismatch")
    if known_covariates_future is not None and known_covariates_future.shape[1] != config.horizon:
        raise BacktestError("known_covariates_future horizon mismatch")

    # Generate splits
    splits = generate_backtest_splits(
        n_obs=n_obs,
        horizon=config.horizon,
        n_windows=config.n_windows,
        step_size=config.step_size,
        expanding=config.expanding_window,
        min_train=config.min_training_length,
        max_train=config.max_training_length,
    )

    if not splits:
        raise BacktestError(
            f"Insufficient data for backtesting: need at least {config.min_training_length + config.horizon} observations, have {n_obs}"
        )

    windows = []

    for window_id, (train_end, test_end) in enumerate(splits):
        # Training data
        train_history = target_history[:, :train_end]
        train_index = history_index[:train_end]

        # Test data
        test_actuals = target_history[:, train_end:test_end]
        test_dates = history_index[train_end:test_end]

        # Covariates for this window
        train_past_cov = past_covariates[:, :train_end] if past_covariates is not None else None
        train_known_hist = known_covariates_history[:, :train_end] if known_covariates_history is not None else None
        test_known_future = known_covariates_future if known_covariates_future is not None else None

        # Build request
        request = ForecastRequest(
            target_history=train_history,
            history_index=train_index,
            target_names=target_names,
            horizon=config.horizon,
            future_index=test_dates,
            past_covariates=train_past_cov,
            known_covariates_history=train_known_hist,
            known_covariates_future=test_known_future,
            quantile_levels=quantile_levels,
        )

        # Generate forecast
        result: ForecastResult = adapter.predict(request)

        # Compute metrics
        point_metrics = compute_point_metrics(
            actuals=test_actuals,
            forecasts=result.point,
            training_actuals=train_history,
            seasonal_period=_infer_seasonal_period(train_index),
        )

        prob_metrics = None
        if result.has_probabilistic():
            prob_metrics = compute_all_probabilistic_metrics(
                actuals=test_actuals,
                quantiles=result.quantiles,
                quantile_levels=result.quantile_levels,
                point_forecasts=result.point,
            )

        windows.append(BacktestWindow(
            window_id=window_id + 1,
            train_end_idx=train_end,
            train_end_date=history_index[train_end - 1],
            test_dates=test_dates,
            test_actuals=test_actuals,
            point_forecasts=result.point,
            quantile_forecasts=result.quantiles,
            quantile_levels=result.quantile_levels,
            point_metrics=point_metrics,
            probabilistic_metrics=prob_metrics,
            model_key=result.model_key,
            context_used=result.context_used,
        ))

    # Aggregate metrics
    agg_point = aggregate_window_metrics(windows)
    agg_prob = aggregate_probabilistic_metrics(windows) if windows[0].probabilistic_metrics else None

    return BacktestResult(
        windows=windows,
        aggregate_point_metrics=agg_point,
        aggregate_probabilistic_metrics=agg_prob,
        config=config,
        model_key=windows[0].model_key,
        target_names=target_names,
    )


def _infer_seasonal_period(index: pd.DatetimeIndex) -> int | None:
    """Infer seasonal period from datetime index."""
    freq = infer_frequency_safe(pd.Series(index))
    if freq == "D":
        return 7
    elif freq == "H":
        return 24
    elif freq == "W":
        return 52
    elif freq == "M":
        return 12
    elif freq == "Q":
        return 4
    return None


def aggregate_window_metrics(windows: list[BacktestWindow]) -> PointMetrics:
    """Aggregate point metrics across windows."""
    from forecasting.evaluation.metrics import aggregate_metrics_across_windows
    return aggregate_metrics_across_windows([w.point_metrics for w in windows])


def aggregate_probabilistic_metrics(windows: list[BacktestWindow]) -> ProbabilisticMetrics:
    """Aggregate probabilistic metrics across windows."""
    # Simple mean aggregation
    first = windows[0].probabilistic_metrics
    if first is None:
        return None

    n = len(windows)
    cov_keys = first.interval_coverage.keys()
    width_keys = first.interval_width.keys()
    pinball_keys = first.pinball_losses.keys()

    agg_cov = {k: float(np.mean([w.probabilistic_metrics.interval_coverage[k] for w in windows])) for k in cov_keys}
    agg_width = {k: float(np.mean([w.probabilistic_metrics.interval_width[k] for w in windows])) for k in width_keys}
    agg_pinball = {k: float(np.mean([w.probabilistic_metrics.pinball_losses[k] for w in windows])) for k in pinball_keys}
    agg_crps = float(np.mean([w.probabilistic_metrics.crps for w in windows if w.probabilistic_metrics.crps is not None and not math.isnan(w.probabilistic_metrics.crps)]))

    # Mean pinball loss
    valid_pinball = [v for v in agg_pinball.values() if not math.isnan(v)]
    mean_pinball = float(np.mean(valid_pinball)) if valid_pinball else math.nan

    return ProbabilisticMetrics(
        interval_coverage=agg_cov,
        interval_width=agg_width,
        mean_pinball_loss=mean_pinball,
        pinball_losses=agg_pinball,
        crps=agg_crps,
    )


# Need to import math
import math