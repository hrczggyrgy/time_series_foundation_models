"""High-level forecasting service orchestrating data, models, and evaluation."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecasting.contracts import ForecastRequest, ForecastResult, RuntimeConfig
from forecasting.data.parsing import (
    ParsedData,
    ValidationReport,
    build_future_dates,
    infer_frequency_safe,
    prepare_dataframe,
    validate_timeseries,
)
from forecasting.evaluation.backtesting import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from forecasting.evaluation.metrics import ProbabilisticMetrics
from forecasting.exceptions import ForecastingError
from forecasting.registry import ModelRegistry, get_registry, LicensePolicy
from forecasting.reporting.exports import (
    export_backtest_long,
    export_backtest_summary,
    export_forecast_long,
    export_forecast_wide,
    export_metadata_json,
    generate_run_id,
)


@dataclass(frozen=False)
class ForecastJob:
    """Complete forecast job result."""

    run_id: str
    request: ForecastRequest
    result: ForecastResult
    actuals: np.ndarray | None = None
    backtest: BacktestResult | None = None
    timing: dict[str, float] = field(default_factory=dict)
    data_fingerprint: str = ""
    preprocessing_info: dict[str, Any] = field(default_factory=dict)
    # For UI display compatibility
    forecast_df: pd.DataFrame | None = None


class ForecastingService:
    """Main service for running forecasts and backtests.

    Policy and acknowledgments are passed per-call for session isolation.
    """

    def __init__(
        self,
        default_runtime: RuntimeConfig | None = None,
    ):
        self.registry = get_registry()
        self.default_runtime = default_runtime or RuntimeConfig()

    def list_models(self, policy: LicensePolicy, acknowledged: set[str] | None = None) -> list[dict[str, Any]]:
        """List available models with their capabilities."""
        acknowledged = acknowledged or set()
        entries = self.registry.list_available(policy)
        return [
            {
                "model_key": e.model_key,
                "display_name": e.display_name,
                "license_policy": e.license_policy.value,
                "license_notice": e.capabilities.license_notice,
                "supports_multivariate": e.capabilities.supports_multivariate,
                "supports_covariates": e.capabilities.supports_past_covariates or e.capabilities.supports_known_future_covariates,
                "max_context": e.capabilities.max_context_length,
                "max_horizon": e.capabilities.max_horizon,
                "native_quantiles": list(e.capabilities.native_quantile_levels),
                "is_acknowledged": e.model_key in acknowledged,
            }
            for e in entries
        ]

    def get_adapter(self, model_key: str, runtime_config: RuntimeConfig, policy: LicensePolicy, acknowledged: set[str] | None = None):
        """Get adapter for a model with session-specific policy/acknowledgment."""
        return self.registry.get_adapter(model_key, runtime_config, policy, acknowledged)

    def prepare_data(
        self,
        raw_df: pd.DataFrame,
        date_col: str,
        target_cols: list[str],
        past_covariate_cols: list[str] | None = None,
        known_covariate_cols: list[str] | None = None,
        auto_fix: bool = True,
        aggregation: str = "mean",
    ) -> tuple[ParsedData, ValidationReport]:
        """Prepare and validate data. Returns (ParsedData, ValidationReport)."""
        # Compute fingerprint for cache invalidation
        raw_bytes = raw_df.to_csv(index=False).encode()
        fingerprint = hashlib.md5(raw_bytes).hexdigest()[:16]

        prepared, prep_info = prepare_dataframe(
            raw_df=raw_df,
            date_col=date_col,
            target_cols=target_cols,
            past_covariate_cols=past_covariate_cols,
            known_covariate_cols=known_covariate_cols,
            auto_fix=auto_fix,
            aggregation=aggregation,
        )

        validation = validate_timeseries(prepared, target_cols, prep_info)

        parsed = ParsedData(
            df=prepared,
            date_column=date_col,
            target_columns=target_cols,
            past_covariate_columns=past_covariate_cols or [],
            known_covariate_columns=known_covariate_cols or [],
            prep_report=prep_info,
            raw_signature=fingerprint,
        )
        return parsed, validation

    def run_forecast(
        self,
        model_key: str,
        parsed_data: ParsedData,
        horizon: int,
        policy: LicensePolicy = LicensePolicy.PERMISSIVE_ONLY,
        acknowledged: set[str] | None = None,
        runtime_config: RuntimeConfig | None = None,
        business_days_only: bool = False,
        quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ) -> ForecastJob:
        """Run a single forecast job."""
        start_time = time.time()
        run_id = generate_run_id()
        timing = {}

        # Get adapter
        runtime = runtime_config or self.default_runtime
        adapter = self.registry.get_adapter(model_key, runtime, policy, acknowledged)
        load_time = time.time()
        timing["model_load_sec"] = load_time - start_time

        # Build request
        df = parsed_data.df
        # Map original target column names to actual dataframe column names
        # First target is normalized to 'y'
        actual_target_columns = []
        for i, col in enumerate(parsed_data.target_columns):
            if i == 0 and "y" in df.columns:
                actual_target_columns.append("y")
            elif col in df.columns:
                actual_target_columns.append(col)
            else:
                actual_target_columns.append(col)

        target_history = df[actual_target_columns].T.values  # [targets, history]
        history_index = df["ds"]

        freq = infer_frequency_safe(history_index)
        future_dates = build_future_dates(
            last_date=history_index.iloc[-1],
            horizon=horizon,
            business_days_only=business_days_only,
            freq_code=freq,
        )

        # Covariates
        past_cov = None
        if parsed_data.past_covariate_columns:
            past_cov = df[parsed_data.past_covariate_columns].T.values

        known_hist = None
        known_future = None
        if parsed_data.known_covariate_columns:
            known_hist = df[parsed_data.known_covariate_columns].T.values
            # For future known covariates, user would need to provide separately
            # For now, we don't support this in the simple interface

        request = ForecastRequest(
            target_history=target_history,
            history_index=history_index,
            target_names=tuple(parsed_data.target_columns),
            horizon=horizon,
            future_index=future_dates,
            past_covariates=past_cov,
            known_covariates_history=known_hist,
            known_covariates_future=known_future,
            quantile_levels=quantile_levels,
        )

        # Run forecast
        result = adapter.predict(request)
        forecast_time = time.time()
        timing["forecast_sec"] = forecast_time - load_time

        # Build job
        job = ForecastJob(
            run_id=run_id,
            request=request,
            result=result,
            actuals=None,
            backtest=None,
            timing=timing,
            data_fingerprint=parsed_data.raw_signature,
            preprocessing_info=parsed_data.prep_report,
        )

        total_time = time.time()
        timing["total_sec"] = total_time - start_time

        return job

    def run_backtest(
        self,
        model_key: str,
        parsed_data: ParsedData,
        horizon: int,
        n_windows: int = 3,
        step_size: int | None = None,
        expanding_window: bool = True,
        runtime_config: RuntimeConfig | None = None,
        quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        policy: LicensePolicy = LicensePolicy.PERMISSIVE_ONLY,
        acknowledged: set[str] | None = None,
    ) -> BacktestResult:
        """Run rolling-origin backtest."""
        runtime = runtime_config or self.default_runtime
        adapter = self.registry.get_adapter(model_key, runtime, policy, acknowledged)

        df = parsed_data.df
        # Map original target column names to actual dataframe column names
        # First target is normalized to 'y'
        actual_target_columns = []
        for i, col in enumerate(parsed_data.target_columns):
            if i == 0 and "y" in df.columns:
                actual_target_columns.append("y")
            elif col in df.columns:
                actual_target_columns.append(col)
            else:
                actual_target_columns.append(col)

        target_history = df[actual_target_columns].T.values
        history_index = df["ds"]

        past_cov = None
        if parsed_data.past_covariate_columns:
            past_cov = df[parsed_data.past_covariate_columns].T.values

        known_hist = None
        if parsed_data.known_covariate_columns:
            known_hist = df[parsed_data.known_covariate_columns].T.values

        config = BacktestConfig(
            horizon=horizon,
            n_windows=n_windows,
            step_size=step_size,
            expanding_window=expanding_window,
            min_training_length=32,
        )

        return run_backtest(
            adapter=adapter,
            target_history=target_history,
            history_index=history_index,
            target_names=tuple(parsed_data.target_columns),
            config=config,
            past_covariates=past_cov,
            known_covariates_history=known_hist,
            known_covariates_future=None,  # Would need separate future covariate data
            quantile_levels=quantile_levels,
        )

    def run_forecast_with_backtest(
        self,
        model_key: str,
        parsed_data: ParsedData,
        horizon: int,
        n_windows: int = 3,
        runtime_config: RuntimeConfig | None = None,
        business_days_only: bool = False,
        quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        policy: LicensePolicy = LicensePolicy.PERMISSIVE_ONLY,
        acknowledged: set[str] | None = None,
    ) -> ForecastJob:
        """Run forecast and backtest in one call."""
        # Run forecast (uses all data)
        job = self.run_forecast(
            model_key=model_key,
            parsed_data=parsed_data,
            horizon=horizon,
            runtime_config=runtime_config,
            business_days_only=business_days_only,
            quantile_levels=quantile_levels,
            policy=policy,
            acknowledged=acknowledged,
        )

        # Run backtest (uses rolling windows)
        bt_start = time.time()
        try:
            backtest = self.run_backtest(
                model_key=model_key,
                parsed_data=parsed_data,
                horizon=horizon,
                n_windows=n_windows,
                runtime_config=runtime_config,
                quantile_levels=quantile_levels,
                policy=policy,
                acknowledged=acknowledged,
            )
            job.backtest = backtest
            job.timing["backtest_sec"] = time.time() - bt_start
        except Exception as e:
            # Backtest failed but forecast succeeded
            job.timing["backtest_error"] = str(e)

        return job

    def export_forecast(
        self,
        job: ForecastJob,
        format: str = "long",
        coverage_levels: tuple[float, ...] = (0.5, 0.6, 0.8),
    ) -> pd.DataFrame:
        """Export forecast results."""
        if format == "long":
            return export_forecast_long(
                job.result, job.request, job.run_id, job.actuals
            )
        elif format == "wide":
            return export_forecast_wide(
                job.result, job.request, job.run_id, job.actuals, coverage_levels
            )
        else:
            raise ValueError(f"Unknown format: {format}")

    def export_backtest(
        self,
        backtest: BacktestResult,
        run_id: str,
        format: str = "long",
    ) -> pd.DataFrame:
        """Export backtest results."""
        if format == "long":
            return export_backtest_long(backtest, run_id)
        elif format == "summary":
            return export_backtest_summary(backtest, run_id)
        else:
            raise ValueError(f"Unknown format: {format}")

    def export_metadata(self, job: ForecastJob, git_commit: str | None = None) -> dict[str, Any]:
        """Export reproducibility metadata."""
        return export_metadata_json(
            job.result,
            job.request,
            job.run_id,
            job.data_fingerprint,
            git_commit=git_commit,
            preprocessing_info=job.preprocessing_info,
            backtest_config=(
                {
                    "horizon": job.backtest.config.horizon,
                    "n_windows": job.backtest.config.n_windows,
                    "step_size": job.backtest.config.step_size,
                    "expanding": job.backtest.config.expanding_window,
                }
                if job.backtest
                else None
            ),
            timing=job.timing,
        )


# Convenience function for demo data
def generate_demo_data(periods: int = 540, freq: str = "D") -> pd.DataFrame:
    """Generate synthetic demo data with trend and seasonality."""
    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(), periods=periods, freq=freq
    )
    trend = np.linspace(120, 230, periods)
    weekly = 8 * np.sin(np.arange(periods) * 2 * np.pi / 7)
    monthly = 15 * np.sin(np.arange(periods) * 2 * np.pi / 30)
    annual = 10 * np.sin(np.arange(periods) * 2 * np.pi / 365)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 4.0, periods)
    values = trend + weekly + monthly + annual + noise
    return pd.DataFrame({"date": dates, "value": np.round(values, 2)})