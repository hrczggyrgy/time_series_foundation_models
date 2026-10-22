"""Export utilities for forecasts and backtest results."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from forecasting.contracts import ForecastRequest, ForecastResult
from forecasting.evaluation.backtesting import BacktestResult, BacktestWindow


def export_forecast_long(
    result: ForecastResult,
    request: ForecastRequest,
    run_id: str,
    actuals: np.ndarray | None = None,
) -> pd.DataFrame:
    """Export forecast in long format (one row per target, horizon step, quantile)."""
    n_targets, horizon = result.point.shape
    n_quantiles = len(result.quantile_levels) if result.quantiles is not None else 0

    rows = []
    for t in range(n_targets):
        target_name = result.target_names[t]
        for h in range(horizon):
            timestamp = result.future_index[h]
            row_base = {
                "run_id": run_id,
                "model_key": result.model_key,
                "checkpoint_id": result.checkpoint_id,
                "checkpoint_revision": result.checkpoint_revision,
                "target": target_name,
                "forecast_origin": request.history_index[-1].isoformat(),
                "timestamp": timestamp.isoformat(),
                "horizon_step": h + 1,
                "point": float(result.point[t, h]),
                "point_definition": result.point_definition,
                "context_used": result.context_used,
            }

            # Add actuals if available
            if actuals is not None:
                actuals_2d = np.atleast_2d(actuals)
                if actuals_2d.shape == (n_targets, horizon):
                    row_base["actual"] = float(actuals_2d[t, h])

            # Point forecast row
            rows.append(row_base.copy())

            # Quantile rows
            if result.quantiles is not None:
                for q_idx, q_level in enumerate(result.quantile_levels):
                    q_row = row_base.copy()
                    q_row.update({
                        "quantile_level": float(q_level),
                        "quantile_value": float(result.quantiles[t, h, q_idx]),
                    })
                    rows.append(q_row)

    return pd.DataFrame(rows)


def export_forecast_wide(
    result: ForecastResult,
    request: ForecastRequest,
    run_id: str,
    actuals: np.ndarray | None = None,
    coverage_levels: tuple[float, ...] = (0.5, 0.6, 0.8),
) -> pd.DataFrame:
    """Export forecast in wide format (one row per target, horizon step)."""
    n_targets, horizon = result.point.shape

    rows = []
    for t in range(n_targets):
        target_name = result.target_names[t]
        for h in range(horizon):
            row = {
                "run_id": run_id,
                "model_key": result.model_key,
                "target": target_name,
                "forecast_origin": request.history_index[-1].isoformat(),
                "timestamp": result.future_index[h].isoformat(),
                "horizon_step": h + 1,
                "point": float(result.point[t, h]),
                "point_definition": result.point_definition,
                "context_used": result.context_used,
            }

            if actuals is not None:
                actuals_2d = np.atleast_2d(actuals)
                if actuals_2d.shape == (n_targets, horizon):
                    row["actual"] = float(actuals_2d[t, h])

            # Add interval columns
            if result.quantiles is not None:
                for cov in coverage_levels:
                    alpha = (1 - cov) / 2
                    lower, upper = result.get_interval(cov)
                    if lower is not None:
                        row[f"lower_{int(cov*100)}"] = float(lower[t, h])
                        row[f"upper_{int(cov*100)}"] = float(upper[t, h])

            rows.append(row)

    return pd.DataFrame(rows)


def export_metadata_json(
    result: ForecastResult,
    request: ForecastRequest,
    run_id: str,
    data_fingerprint: str,
    git_commit: str | None = None,
    app_version: str = "0.2.0",
    preprocessing_info: dict[str, Any] | None = None,
    backtest_config: dict[str, Any] | None = None,
    timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Create metadata JSON for reproducibility."""
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "app_version": app_version,
        "git_commit": git_commit,
        "model": {
            "key": result.model_key,
            "checkpoint_id": result.checkpoint_id,
            "checkpoint_revision": result.checkpoint_revision,
            "point_definition": result.point_definition,
        },
        "data": {
            "fingerprint": data_fingerprint,
            "n_targets": len(result.target_names),
            "target_names": list(result.target_names),
            "history_length": request.n_history,
            "horizon": request.horizon,
            "frequency": infer_frequency_str(request.history_index),
            "timezone": str(request.history_index.tz) if request.history_index.tz else "naive",
            "quantile_levels": [float(q) for q in result.quantile_levels],
            "preprocessing": preprocessing_info or {},
        },
        "forecast": {
            "context_used": result.context_used,
            "warnings": list(result.warnings),
            "metadata": result.metadata,
        },
        "backtest": backtest_config or {},
        "timing": timing or {},
        "environment": {
            "python_version": "3.10+",
        },
    }


def export_backtest_long(
    backtest: BacktestResult,
    run_id: str,
) -> pd.DataFrame:
    """Export backtest results in long format."""
    rows = []

    for window in backtest.windows:
        n_targets, horizon = window.point_forecasts.shape
        n_quantiles = len(window.quantile_levels) if window.quantile_forecasts is not None else 0

        for t in range(n_targets):
            target_name = backtest.target_names[t]
            for h in range(horizon):
                row = {
                    "run_id": run_id,
                    "model_key": window.model_key,
                    "window": window.window_id,
                    "train_end_date": window.train_end_date.isoformat(),
                    "target": target_name,
                    "timestamp": window.test_dates[h].isoformat(),
                    "horizon_step": h + 1,
                    "actual": float(window.test_actuals[t, h]) if window.test_actuals.ndim == 2 else float(window.test_actuals[h]),
                    "point_forecast": float(window.point_forecasts[t, h]) if window.point_forecasts.ndim == 2 else float(window.point_forecasts[h]),
                    "abs_error": abs(
                        float(window.test_actuals[t, h] if window.test_actuals.ndim == 2 else window.test_actuals[h])
                        - float(window.point_forecasts[t, h] if window.point_forecasts.ndim == 2 else window.point_forecasts[h])
                    ),
                }

                if window.quantile_forecasts is not None:
                    for q_idx, q_level in enumerate(window.quantile_levels):
                        q_val = window.quantile_forecasts[t, h, q_idx] if window.quantile_forecasts.ndim == 3 else window.quantile_forecasts[h, q_idx]
                        row[f"quantile_{q_level}"] = float(q_val)

                rows.append(row)

    return pd.DataFrame(rows)


def export_backtest_summary(
    backtest: BacktestResult,
    run_id: str,
) -> pd.DataFrame:
    """Export per-window and aggregate metrics summary."""
    rows = []

    # Per-window metrics
    for window in backtest.windows:
        pm = window.point_metrics
        row = {
            "run_id": run_id,
            "model_key": window.model_key,
            "window": window.window_id,
            "train_end_date": window.train_end_date.isoformat(),
            "MAE": pm.mae,
            "RMSE": pm.rmse,
            "MAPE": pm.mape,
            "Bias": pm.bias,
            "MASE": pm.mase,
        }

        if window.probabilistic_metrics:
            pr = window.probabilistic_metrics
            for cov, val in pr.interval_coverage.items():
                row[f"Coverage_{int(cov*100)}"] = val
            for cov, val in pr.interval_width.items():
                row[f"Width_{int(cov*100)}"] = val
            row["Mean_Pinball"] = pr.mean_pinball_loss
            row["CRPS"] = pr.crps

        rows.append(row)

    # Aggregate row
    if len(rows) > 1:
        agg = backtest.aggregate_point_metrics
        agg_row = {
            "run_id": run_id,
            "model_key": backtest.model_key,
            "window": "AGGREGATE",
            "train_end_date": "",
            "MAE": agg.mae,
            "RMSE": agg.rmse,
            "MAPE": agg.mape,
            "Bias": agg.bias,
            "MASE": agg.mase,
        }
        if backtest.aggregate_probabilistic_metrics:
            pr = backtest.aggregate_probabilistic_metrics
            for cov, val in pr.interval_coverage.items():
                agg_row[f"Coverage_{int(cov*100)}"] = val
            for cov, val in pr.interval_width.items():
                agg_row[f"Width_{int(cov*100)}"] = val
            agg_row["Mean_Pinball"] = pr.mean_pinball_loss
            agg_row["CRPS"] = pr.crps
        rows.append(agg_row)

    return pd.DataFrame(rows)


def infer_frequency_str(index: pd.DatetimeIndex) -> str:
    """Infer frequency string from datetime index."""
    if len(index) < 3:
        return "Unknown"
    try:
        freq = pd.infer_freq(index)
        if freq:
            return freq
    except Exception:
        pass
    return "Irregular"


def generate_run_id() -> str:
    """Generate unique run ID."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]


def save_json(path: str, data: dict[str, Any]) -> None:
    """Save JSON with proper encoding."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)