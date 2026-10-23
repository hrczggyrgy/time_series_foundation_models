"""Autonomous two-stage forecasting with leakage-free residual training."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecasting.contracts import ForecastRequest, ForecastResult, RuntimeConfig
from forecasting.data.parsing import ParsedData, build_future_dates, infer_frequency_safe
from forecasting.evaluation.backtesting import generate_backtest_splits
from forecasting.evaluation.metrics import PointMetrics, compute_point_metrics

ProgressCallback = Callable[[str, str, float | None], None]

# These adapters' inference implementations concatenate historical and future
# known covariates and pass them into the underlying forecasting model.
_VERIFIED_COVARIATE_MODELS = frozenset({"timesfm_3", "tirex_2", "t0_gamma"})
_DEFAULT_NATIVE_RESIDUAL_MODELS = ("native_exog", "native_arx")


@dataclass(frozen=True)
class CandidateScore:
    """Validation score or explicit failure for one candidate."""

    model_key: str
    mae: float | None
    rmse: float | None
    folds: int
    elapsed_seconds: float
    failure: str | None = None


@dataclass(frozen=True)
class ResidualTrainingData:
    """Aligned out-of-sample residuals and corresponding base predictions."""

    residuals: np.ndarray
    base_predictions: np.ndarray
    index: pd.DatetimeIndex

    @property
    def n_observations(self) -> int:
        return self.residuals.shape[1]


@dataclass(frozen=True)
class HybridRunResult:
    """Complete result and selection report for an autonomous hybrid run."""

    result: ForecastResult
    base_forecast: np.ndarray
    residual_forecast: np.ndarray
    base_model_key: str
    residual_model_key: str | None
    base_validation: PointMetrics
    residual_validation: PointMetrics | None
    holdout_metrics: dict[str, float | None] | None
    independent_holdout: bool
    residual_observations: int
    base_candidates: tuple[CandidateScore, ...]
    residual_candidates: tuple[CandidateScore, ...]
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    status: str
    timing: dict[str, float] = field(default_factory=dict)


def select_best_candidate(scores: list[CandidateScore]) -> CandidateScore | None:
    """Rank valid candidates by MAE, then RMSE, then stable model key."""
    valid = [score for score in scores if score.mae is not None and score.rmse is not None]
    return min(valid, key=lambda score: (score.mae, score.rmse, score.model_key)) if valid else None


def generate_oof_residuals(
    adapter: Any,
    target_history: np.ndarray,
    history_index: pd.DatetimeIndex,
    target_names: tuple[str, ...],
    observation_mask: np.ndarray,
    horizon: int,
    min_training_history: int = 32,
    max_blocks: int = 20,
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    progress_callback: Callable[[int, int], None] | None = None,
) -> ResidualTrainingData:
    """Generate blocked expanding-window errors; never returns in-sample residuals."""
    values = _as_2d(target_history)
    mask = np.array(observation_mask, dtype=bool, copy=True)
    if mask.shape != (values.shape[1],):
        raise ValueError("observation_mask must have one value per timestamp.")
    if len(history_index) != values.shape[1]:
        raise ValueError("history_index and target_history lengths must match.")
    if max_blocks < 1 or horizon < 1:
        raise ValueError("max_blocks and horizon must be positive.")
    if values.shape[1] <= min_training_history:
        raise ValueError("Insufficient history to generate out-of-sample residuals.")

    block_size = max(1, horizon)
    possible_origins = list(range(min_training_history, values.shape[1], block_size))
    origins = possible_origins[-max_blocks:]
    residual_chunks: list[np.ndarray] = []
    base_chunks: list[np.ndarray] = []
    date_chunks: list[pd.DatetimeIndex] = []

    for block_number, origin in enumerate(origins, start=1):
        end = min(origin + block_size, values.shape[1])
        if end <= origin:
            continue
        forecast = _predict_base(
            adapter,
            values[:, :origin],
            history_index[:origin],
            target_names,
            end - origin,
            history_index[origin:end],
            quantile_levels,
        )
        if progress_callback:
            progress_callback(block_number, len(origins))
        prediction = np.asarray(forecast.point, dtype=float)
        actual = values[:, origin:end]
        valid = mask[origin:end] & np.all(np.isfinite(actual), axis=0)
        valid &= np.all(np.isfinite(prediction), axis=0)
        if not np.any(valid):
            continue
        # Sign convention is actual minus the out-of-sample base prediction.
        residual_chunks.append(actual[:, valid] - prediction[:, valid])
        base_chunks.append(prediction[:, valid])
        date_chunks.append(pd.DatetimeIndex(history_index[origin:end][valid]))

    if not residual_chunks:
        raise ValueError("No observed, finite out-of-sample residuals could be generated.")
    return ResidualTrainingData(
        residuals=np.concatenate(residual_chunks, axis=1),
        base_predictions=np.concatenate(base_chunks, axis=1),
        index=pd.DatetimeIndex([date for chunk in date_chunks for date in chunk]),
    )


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values.reshape(1, -1) if values.ndim == 1 else values


def _check_request(adapter: Any, request: ForecastRequest) -> None:
    can_handle, reasons = adapter.capabilities.can_handle_request(request)
    if not can_handle:
        raise ValueError("; ".join(reasons))


def _fit_predict_native(
    data: ResidualTrainingData,
    future_base: np.ndarray,
    lag_count: int,
) -> np.ndarray:
    """Ridge-regularized residual regression with the base forecast as an exog."""
    residuals = _as_2d(data.residuals)
    base = _as_2d(data.base_predictions)
    future_base = _as_2d(future_base)
    if residuals.shape != base.shape or residuals.shape[0] != future_base.shape[0]:
        raise ValueError("Residual targets and base covariates must have compatible shapes.")
    if residuals.shape[1] < max(8, lag_count + 4):
        raise ValueError("Insufficient residual observations for the selected ARX model.")
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(base)):
        raise ValueError("Residual training data must be finite.")
    if not np.all(np.isfinite(future_base)):
        raise ValueError("Future base forecast must be finite.")

    output = np.empty_like(future_base, dtype=float)
    for target in range(residuals.shape[0]):
        y = residuals[target]
        x_base = base[target]
        lags = min(lag_count, max(0, (len(y) - 4) // 3))
        start = lags
        design = np.column_stack(
            [
                np.ones(len(y) - start),
                x_base[start:],
                *[y[start - lag : len(y) - lag] for lag in range(1, lags + 1)],
            ]
        )
        target_values = y[start:]
        penalty = np.eye(design.shape[1]) * 1e-3
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target_values)
        history = list(y[-lags:]) if lags else []
        for step, base_value in enumerate(future_base[target]):
            features = [1.0, base_value]
            features.extend(reversed(history))
            prediction = float(np.dot(coefficients, features))
            if not math.isfinite(prediction):
                raise ValueError("Native residual model produced a non-finite forecast.")
            output[target, step] = prediction
            if lags:
                history = (history + [prediction])[-lags:]
    return output


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> PointMetrics:
    return compute_point_metrics(actual, predicted)


def _observed_values(
    actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.array(mask, dtype=bool, copy=True)
    valid &= np.all(np.isfinite(actual), axis=0)
    valid &= np.all(np.isfinite(predicted), axis=0)
    return actual[:, valid], predicted[:, valid]


def _predict_base(
    adapter: Any,
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    target_names: tuple[str, ...],
    horizon: int,
    future_dates: pd.DatetimeIndex,
    quantile_levels: tuple[float, ...],
) -> ForecastResult:
    capabilities = adapter.capabilities
    max_context = capabilities.max_context_length
    if max_context and values.shape[1] > max_context:
        values = values[:, -max_context:]
        dates = dates[-max_context:]
    request = ForecastRequest(
        target_history=values,
        history_index=dates,
        target_names=target_names,
        horizon=horizon,
        future_index=future_dates,
        quantile_levels=quantile_levels,
    )
    _check_request(adapter, request)
    result = adapter.predict(request)
    if result.future_index.equals(future_dates) is False:
        raise ValueError("Base model returned timestamps that do not match the request.")
    if not np.all(np.isfinite(result.point)):
        raise ValueError("Base model returned non-finite point forecasts.")
    return result


def _foundation_residual_candidates(service: Any) -> list[str]:
    return [
        entry.model_key
        for entry in service.registry.list_available(include_experimental=False)
        if entry.model_key in _VERIFIED_COVARIATE_MODELS
        and entry.capabilities.supports_known_future_covariates
    ]


def _predict_residual_model(
    service: Any,
    key: str,
    data: ResidualTrainingData,
    future_base: np.ndarray,
    future_dates: pd.DatetimeIndex,
    target_names: tuple[str, ...],
    runtime: RuntimeConfig,
    quantile_levels: tuple[float, ...],
) -> np.ndarray:
    if key == "native_exog":
        return _fit_predict_native(data, future_base, 0)
    if key == "native_arx":
        return _fit_predict_native(data, future_base, 4)
    if key not in _VERIFIED_COVARIATE_MODELS:
        raise ValueError(f"Residual model {key} has not passed covariate-use verification.")

    adapter = service.registry.get_adapter(key, runtime)
    max_context = adapter.capabilities.max_context_length
    start = max(0, data.n_observations - max_context) if max_context else 0
    residual_history = data.residuals[:, start:]
    base_history = data.base_predictions[:, start:]
    residual_index = data.index[start:]
    request = ForecastRequest(
        target_history=residual_history,
        history_index=residual_index,
        target_names=target_names,
        horizon=future_base.shape[1],
        future_index=future_dates,
        known_covariates_history=base_history,
        known_covariates_future=future_base,
        quantile_levels=quantile_levels,
    )
    _check_request(adapter, request)
    result = adapter.predict(request)
    if not result.future_index.equals(future_dates):
        raise ValueError("Residual model returned timestamps that do not match the base forecast.")
    if result.point.shape != future_base.shape or not np.all(np.isfinite(result.point)):
        raise ValueError("Residual model returned invalid or non-finite forecasts.")
    return np.asarray(result.point, dtype=float)


def _close_model(service: Any, key: str) -> None:
    matching_keys = [
        cache_key
        for cache_key in service.registry._instances
        if cache_key.startswith(f"{key}:")
    ]
    for cache_key in matching_keys:
        adapter = service.registry._instances[cache_key]
        adapter.close()
        del service.registry._instances[cache_key]


def run_autonomous_hybrid(
    service: Any,
    parsed_data: ParsedData,
    horizon: int,
    runtime_config: RuntimeConfig | None = None,
    business_days_only: bool = False,
    n_windows: int = 3,
    min_training_history: int = 32,
    max_oof_blocks: int = 20,
    quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    progress_callback: ProgressCallback | None = None,
) -> HybridRunResult:
    """Select and evaluate a base/residual model pair, then forecast the future."""
    started = time.perf_counter()
    runtime = runtime_config or service.default_runtime
    df = parsed_data.df
    target_columns = [
        "y" if i == 0 and "y" in df.columns else column
        for i, column in enumerate(parsed_data.target_columns)
    ]
    values = _as_2d(df[target_columns].to_numpy(dtype=float).T)
    dates = pd.DatetimeIndex(df["ds"])
    observed_mask = np.asarray(parsed_data.observation_mask, dtype=bool)
    target_names = tuple(parsed_data.target_columns)
    if horizon < 1 or values.shape[1] != len(dates):
        raise ValueError("Invalid horizon or mismatched input time series.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Prepared target history contains non-finite values.")

    independent_holdout = values.shape[1] >= min_training_history + (2 * horizon)
    if independent_holdout:
        selection_end = values.shape[1] - horizon
        holdout_start = selection_end
        selection_values, selection_dates = values[:, :selection_end], dates[:selection_end]
        selection_mask = observed_mask[:selection_end]
    else:
        selection_end = values.shape[1]
        holdout_start = selection_end
        selection_values, selection_dates = values, dates
        selection_mask = observed_mask
    splits = generate_backtest_splits(
        n_obs=selection_end,
        horizon=horizon,
        n_windows=n_windows,
        expanding=True,
        min_train=min_training_history,
    )
    if not splits:
        raise ValueError(
            "Insufficient history for rolling-origin hybrid selection; "
            f"need at least {min_training_history + horizon} observations."
        )
    if progress_callback:
        progress_callback("base_search", "Evaluating available base models.", 0.05)

    stage_timings: dict[str, float] = {}
    base_search_started = time.perf_counter()
    entries = service.registry.list_available(include_experimental=False)
    candidate_keys = []
    candidate_failures: list[str] = []
    for entry in entries:
        caps = entry.capabilities
        if caps.min_context_length > min(train_end for train_end, _ in splits):
            candidate_failures.append(
                f"{entry.model_key}: minimum context exceeds the shared validation history."
            )
        elif caps.max_horizon is not None and horizon > caps.max_horizon:
            candidate_failures.append(
                f"{entry.model_key}: horizon {horizon} exceeds maximum {caps.max_horizon}."
            )
        elif values.shape[0] > 1 and not caps.supports_multivariate:
            candidate_failures.append(f"{entry.model_key}: multivariate targets are unsupported.")
        else:
            candidate_keys.append(entry.model_key)
    if not candidate_keys:
        raise ValueError(
            "No available models satisfy the requested target and horizon constraints."
        )

    base_scores: list[CandidateScore] = []
    for candidate_index, key in enumerate(candidate_keys):
        model_started = time.perf_counter()
        predicted_values: list[np.ndarray] = []
        actual_values: list[np.ndarray] = []
        try:
            adapter = service.registry.get_adapter(key, runtime)
            for train_end, test_end in splits:
                future_dates = selection_dates[train_end:test_end]
                result = _predict_base(
                    adapter,
                    selection_values[:, :train_end],
                    selection_dates[:train_end],
                    target_names,
                    test_end - train_end,
                    future_dates,
                    quantile_levels,
                )
                actual, predicted = _observed_values(
                    selection_values[:, train_end:test_end],
                    result.point,
                    selection_mask[train_end:test_end],
                )
                if predicted.size == 0:
                    raise ValueError("No genuine observed values were available to score a fold.")
                predicted_values.append(predicted.ravel())
                actual_values.append(actual.ravel())
            aggregate = _metrics(np.concatenate(actual_values), np.concatenate(predicted_values))
            if not math.isfinite(aggregate.mae) or not math.isfinite(aggregate.rmse):
                raise ValueError("Candidate produced invalid validation metrics.")
            base_scores.append(
                CandidateScore(
                    key,
                    aggregate.mae,
                    aggregate.rmse,
                    len(splits),
                    time.perf_counter() - model_started,
                )
            )
        except Exception as exc:
            reason = f"{key}: {exc}"
            candidate_failures.append(reason)
            base_scores.append(
                CandidateScore(key, None, None, 0, time.perf_counter() - model_started, str(exc))
            )
        finally:
            _close_model(service, key)
        if progress_callback:
            progress_callback(
                "base_search",
                f"Scored {key} ({candidate_index + 1}/{len(candidate_keys)}).",
                0.05 + 0.25 * (candidate_index + 1) / len(candidate_keys),
            )

    best_base = select_best_candidate(base_scores)
    if best_base is None:
        raise ValueError(
            "All available base models failed validation. " + " | ".join(candidate_failures)
        )
    if best_base.mae is None or best_base.rmse is None:
        raise ValueError("Selected base model has no valid validation metrics.")
    stage_timings["base_search_seconds"] = time.perf_counter() - base_search_started
    base_validation = PointMetrics(
        mae=float(best_base.mae),
        rmse=float(best_base.rmse),
        mape=None,
        bias=math.nan,
        mase=None,
        mase_denominator=None,
    )
    if progress_callback:
        progress_callback("oof_residuals", f"Selected base model: {best_base.model_key}.", 0.33)

    residual_search_started = time.perf_counter()
    residual_keys = [*_DEFAULT_NATIVE_RESIDUAL_MODELS, *_foundation_residual_candidates(service)]
    residual_fold_inputs: list[
        tuple[
            ResidualTrainingData,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            pd.DatetimeIndex,
        ]
    ] = []
    base_runtime = runtime
    base_adapter = None
    residual_preparation_error: Exception | None = None
    try:
        base_adapter = service.registry.get_adapter(best_base.model_key, base_runtime)
        for fold_number, (train_end, test_end) in enumerate(splits, start=1):
            fold_base = _predict_base(
                base_adapter,
                selection_values[:, :train_end],
                selection_dates[:train_end],
                target_names,
                test_end - train_end,
                selection_dates[train_end:test_end],
                quantile_levels,
            )
            def report_oof_progress(
                completed: int,
                total: int,
                fold_number: int = fold_number,
                fold_count: int = len(splits),
            ) -> None:
                if progress_callback:
                    fraction = 0.33 + 0.07 * (
                        (fold_number - 1) + completed / max(1, total)
                    ) / fold_count
                    progress_callback(
                        "oof_residuals",
                        f"Validation fold {fold_number}/{fold_count}: "
                        f"generated {completed}/{total} residual blocks.",
                        fraction,
                    )

            oof = generate_oof_residuals(
                base_adapter,
                selection_values[:, :train_end],
                selection_dates[:train_end],
                target_names,
                selection_mask[:train_end],
                horizon,
                min_training_history,
                max_oof_blocks,
                quantile_levels,
                report_oof_progress,
            )
            residual_fold_inputs.append(
                (
                    oof,
                    fold_base.point,
                    selection_values[:, train_end:test_end],
                    selection_mask[train_end:test_end],
                    selection_dates[train_end:test_end],
                )
            )
    except Exception as exc:
        residual_preparation_error = exc
        residual_keys = []
        candidate_failures.append(f"residual OOF preparation: {exc}")
    finally:
        if base_adapter is not None:
            _close_model(service, best_base.model_key)
    if progress_callback:
        progress_callback("residual_search", "Evaluating residual models on hybrid accuracy.", 0.4)

    residual_scores: list[CandidateScore] = []
    residual_validation_by_key: dict[str, PointMetrics] = {}
    for candidate_index, key in enumerate(residual_keys):
        model_started = time.perf_counter()
        all_actual: list[np.ndarray] = []
        all_hybrid: list[np.ndarray] = []
        try:
            for oof, base_fold, actual_fold, fold_mask, future_dates in residual_fold_inputs:
                correction = _predict_residual_model(
                    service,
                    key,
                    oof,
                    base_fold,
                    future_dates,
                    target_names,
                    runtime,
                    quantile_levels,
                )
                hybrid = base_fold + correction
                if hybrid.shape != actual_fold.shape or not np.all(np.isfinite(hybrid)):
                    raise ValueError("Hybrid fold forecast has invalid shape or non-finite values.")
                actual, predicted = _observed_values(actual_fold, hybrid, fold_mask)
                if predicted.size == 0:
                    raise ValueError("No genuine observed values were available to score a fold.")
                all_actual.append(actual.ravel())
                all_hybrid.append(predicted.ravel())
            aggregate = _metrics(np.concatenate(all_actual), np.concatenate(all_hybrid))
            if not math.isfinite(aggregate.mae) or not math.isfinite(aggregate.rmse):
                raise ValueError("Candidate produced invalid hybrid validation metrics.")
            residual_scores.append(
                CandidateScore(
                    key,
                    aggregate.mae,
                    aggregate.rmse,
                    len(splits),
                    time.perf_counter() - model_started,
                )
            )
            residual_validation_by_key[key] = aggregate
        except Exception as exc:
            candidate_failures.append(f"residual {key}: {exc}")
            residual_scores.append(
                CandidateScore(key, None, None, 0, time.perf_counter() - model_started, str(exc))
            )
        finally:
            if key in _VERIFIED_COVARIATE_MODELS:
                _close_model(service, key)
        if progress_callback:
            progress_callback(
                "residual_search",
                f"Scored residual model {key} ({candidate_index + 1}/{len(residual_keys)}).",
                0.4 + 0.2 * (candidate_index + 1) / len(residual_keys),
            )

    best_residual = select_best_candidate(residual_scores)
    stage_timings["residual_search_seconds"] = (
        time.perf_counter() - residual_search_started
    )
    residual_selection_warning: str | None = None
    if (
        best_residual is not None
        and best_residual.mae is not None
        and best_residual.rmse is not None
        and (best_residual.mae, best_residual.rmse)
        >= (best_base.mae, best_base.rmse)
    ):
        residual_selection_warning = (
            f"The best residual correction ({best_residual.model_key}) did not improve "
            "the base model on the rolling selection folds; returning the base-only forecast."
        )
        best_residual = None
    residual_validation = (
        residual_validation_by_key.get(best_residual.model_key) if best_residual else None
    )

    if progress_callback:
        progress_callback(
            "final_fit",
            "Evaluating untouched holdout and fitting final forecast.",
            0.65,
        )
    # Reserve the final horizon from all model-selection decisions when data allows.
    holdout_metrics: dict[str, float | None] | None = None

    warnings: list[str] = []
    if residual_selection_warning is not None:
        warnings.append(residual_selection_warning)
    if residual_preparation_error is not None:
        warnings.append(
            "Out-of-sample residual training data could not be prepared; "
            f"the final forecast will use the base model only: {residual_preparation_error}"
        )
    if not independent_holdout:
        warnings.append(
            "The dataset is too short for a separate holdout; validation scores are "
            "model-selection/tuning scores."
        )
    holdout_train_end = selection_end
    if independent_holdout:
        holdout_adapter = None
        try:
            holdout_adapter = service.registry.get_adapter(best_base.model_key, runtime)
            base_holdout = _predict_base(
                holdout_adapter,
                values[:, :holdout_train_end],
                dates[:holdout_train_end],
                target_names,
                horizon,
                dates[holdout_start:],
                quantile_levels,
            )
            valid_actual, valid_base = _observed_values(
                values[:, holdout_start:], base_holdout.point, observed_mask[holdout_start:]
            )
            if valid_actual.size:
                base_metrics = _metrics(valid_actual, valid_base)
                holdout_metrics = {
                    "base_mae": base_metrics.mae,
                    "base_rmse": base_metrics.rmse,
                    "hybrid_mae": None,
                    "hybrid_rmse": None,
                }
                if best_residual is not None:
                    try:
                        oof_holdout = generate_oof_residuals(
                            holdout_adapter,
                            values[:, :holdout_train_end],
                            dates[:holdout_train_end],
                            target_names,
                            observed_mask[:holdout_train_end],
                            horizon,
                            min_training_history,
                            max_oof_blocks,
                            quantile_levels,
                        )
                        correction_holdout = _predict_residual_model(
                            service,
                            best_residual.model_key,
                            oof_holdout,
                            base_holdout.point,
                            dates[holdout_start:],
                            target_names,
                            runtime,
                            quantile_levels,
                        )
                        valid_actual_h, valid_hybrid = _observed_values(
                            values[:, holdout_start:],
                            base_holdout.point + correction_holdout,
                            observed_mask[holdout_start:],
                        )
                        if valid_actual_h.size:
                            hybrid_metrics = _metrics(valid_actual_h, valid_hybrid)
                            holdout_metrics["hybrid_mae"] = hybrid_metrics.mae
                            holdout_metrics["hybrid_rmse"] = hybrid_metrics.rmse
                    except Exception as exc:
                        warnings.append(f"Independent hybrid holdout evaluation failed: {exc}")
                    finally:
                        if best_residual.model_key in _VERIFIED_COVARIATE_MODELS:
                            _close_model(service, best_residual.model_key)
            else:
                warnings.append("Independent holdout contains no genuine observed target values.")
        except Exception as exc:
            warnings.append(f"Independent base holdout evaluation failed: {exc}")
        finally:
            if holdout_adapter is not None:
                _close_model(service, best_base.model_key)

    # Refit both stages on all prepared history for the requested future forecast.
    if progress_callback:
        progress_callback("final_forecast", "Producing the future hybrid forecast.", 0.82)
    final_future_dates = build_future_dates(
        last_date=dates[-1],
        horizon=horizon,
        business_days_only=business_days_only,
        freq_code=infer_frequency_safe(pd.Series(dates)),
        raw_freq=parsed_data.prep_report.get("raw_frequency"),
    )
    final_base_adapter = service.registry.get_adapter(best_base.model_key, runtime)
    final_forecast_started = time.perf_counter()
    try:
        final_base_result = _predict_base(
            final_base_adapter,
            values,
            dates,
            target_names,
            horizon,
            pd.DatetimeIndex(final_future_dates),
            quantile_levels,
        )
        final_base = final_base_result.point
        residual_count = 0
        correction = np.zeros_like(final_base)
        selected_residual_key: str | None = None
        status = "base_only"
        result_warnings = list(final_base_result.warnings)
        if best_residual is not None:
            try:
                final_oof = generate_oof_residuals(
                    final_base_adapter,
                    values,
                    dates,
                    target_names,
                    observed_mask,
                    horizon,
                    min_training_history,
                    max_oof_blocks,
                    quantile_levels,
                )
                residual_count = final_oof.n_observations
                correction = _predict_residual_model(
                    service,
                    best_residual.model_key,
                    final_oof,
                    final_base,
                    pd.DatetimeIndex(final_future_dates),
                    target_names,
                    runtime,
                    quantile_levels,
                )
                selected_residual_key = best_residual.model_key
                status = "hybrid"
            except Exception as exc:
                candidate_failures.append(f"final residual fit {best_residual.model_key}: {exc}")
                warnings.append(
                    "Residual correction was not completed; returning the base-only "
                    f"forecast: {exc}"
                )
        else:
            warnings.append(
                "No residual model improved on base validation; returning the base-only forecast."
            )
        if status == "hybrid" and final_base_result.quantiles is not None:
            final_quantiles = final_base_result.quantiles + correction[:, :, np.newaxis]
            result_warnings.append(
                "Prediction intervals are an approximation: the point correction was "
                "added to base quantiles; "
                "residual-model uncertainty was not integrated."
            )
        else:
            final_quantiles = final_base_result.quantiles
        result = ForecastResult(
            point=final_base + correction,
            quantiles=final_quantiles,
            quantile_levels=final_base_result.quantile_levels,
            target_names=target_names,
            future_index=pd.DatetimeIndex(final_future_dates),
            model_key="autonomous_hybrid",
            checkpoint_id=None,
            checkpoint_revision=None,
            point_definition=(
                "base_plus_residual_correction"
                if status == "hybrid"
                else "base_only_fallback"
            ),
            context_used=final_base_result.context_used,
            warnings=tuple(result_warnings + warnings),
            metadata={
                "strategy": "autonomous_hybrid",
                "status": status,
                "base_model_key": best_base.model_key,
                "residual_model_key": selected_residual_key,
                "residual_observations": residual_count,
                "independent_holdout": independent_holdout,
            },
        )
        if status == "base_only":
            selected_residual_key = None
            correction = np.zeros_like(final_base)
    finally:
        _close_model(service, best_base.model_key)
    if best_residual and best_residual.model_key in _VERIFIED_COVARIATE_MODELS:
        _close_model(service, best_residual.model_key)
    stage_timings["final_forecast_seconds"] = (
        time.perf_counter() - final_forecast_started
    )

    if holdout_metrics is not None:
        if holdout_metrics["hybrid_mae"] is not None:
            warnings.append(
                "Independent holdout scores are reported for both base-only and hybrid forecasts; "
                "the residual model was selected without using those held-out targets."
            )
        else:
            warnings.append("Independent holdout accuracy is reported for the base-only forecast.")
    residual_validation = residual_validation if selected_residual_key is not None else None
    return HybridRunResult(
        result=result,
        base_forecast=final_base,
        residual_forecast=correction,
        base_model_key=best_base.model_key,
        residual_model_key=selected_residual_key,
        base_validation=base_validation,
        residual_validation=residual_validation,
        holdout_metrics=holdout_metrics,
        independent_holdout=independent_holdout,
        residual_observations=int(result.metadata["residual_observations"]),
        base_candidates=tuple(base_scores),
        residual_candidates=tuple(residual_scores),
        failures=tuple(candidate_failures),
        warnings=tuple(warnings),
        status=status,
        timing={
            **stage_timings,
            "total_seconds": time.perf_counter() - started,
        },
    )
