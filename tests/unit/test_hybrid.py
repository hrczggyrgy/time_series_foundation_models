"""Unit tests for autonomous hybrid selection and leakage controls."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecasting.contracts import (
    ForecastRequest,
    ForecastResult,
    ModelCapabilities,
    RuntimeConfig,
)
from forecasting.data.parsing import ParsedData
from forecasting.hybrid import (
    CandidateScore,
    _foundation_residual_candidates,
    _observed_values,
    _predict_residual_model,
    generate_oof_residuals,
    select_best_candidate,
)
from forecasting.service import ForecastingService


class _ConstantAdapter:
    def __init__(self, model_key: str, forecast_value: float = 8.0):
        self.forecast_value = forecast_value
        self.requests: list[ForecastRequest] = []
        self.capabilities = ModelCapabilities(
            model_key=model_key,
            display_name=model_key,
            checkpoint_id="",
            supports_cpu=True,
            supports_cuda=False,
            supports_multivariate=True,
            min_context_length=4,
            max_horizon=128,
        )

    def predict(self, request: ForecastRequest) -> ForecastResult:
        self.requests.append(request)
        point = np.full((request.n_targets, request.horizon), self.forecast_value)
        return ForecastResult(
            point=point,
            quantiles=None,
            quantile_levels=np.asarray(request.quantile_levels),
            target_names=request.target_names,
            future_index=request.future_index,
            model_key=self.capabilities.model_key,
            checkpoint_id=None,
            checkpoint_revision=None,
            point_definition="test",
            context_used=request.n_history,
        )

    def close(self) -> None:
        pass


class _Registry:
    def __init__(self, models: dict[str, _ConstantAdapter]):
        self.models = models
        self._instances: dict[str, _ConstantAdapter] = {}
        self.entries = [
            SimpleNamespace(model_key=key, capabilities=adapter.capabilities)
            for key, adapter in models.items()
        ]

    def list_available(self, include_experimental: bool = False):
        return self.entries

    def get_adapter(self, model_key: str, runtime: RuntimeConfig):
        adapter = self.models[model_key]
        self._instances[f"{model_key}:fake"] = adapter
        return adapter


class _Service:
    def __init__(self, models: dict[str, _ConstantAdapter]):
        self.registry = _Registry(models)
        self.default_runtime = RuntimeConfig(device="cpu")


def _parsed_data(length: int, value: float = 10.0) -> ParsedData:
    dates = pd.date_range("2024-01-01", periods=length, freq="D")
    return ParsedData(
        df=pd.DataFrame({"ds": dates, "y": np.full(length, value)}),
        date_column="ds",
        target_columns=["y"],
        past_covariate_columns=[],
        known_covariate_columns=[],
        prep_report={},
        raw_signature="test-data",
        observation_mask=pd.Series(np.ones(length, dtype=bool)),
    )


def test_oof_residuals_are_actual_minus_predictions_from_earlier_origins():
    adapter = _ConstantAdapter("base")
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    target = np.full((1, 70), 10.0)

    data = generate_oof_residuals(
        adapter,
        target,
        dates,
        ("y",),
        np.ones(70, dtype=bool),
        horizon=5,
        min_training_history=32,
    )

    assert np.allclose(data.residuals, 2.0)
    assert np.allclose(data.base_predictions, 8.0)
    assert len(data.index) == data.n_observations
    assert all(request.history_index[-1] < request.future_index[0] for request in adapter.requests)
    assert max(request.future_index.max() for request in adapter.requests) <= dates[-1]


def test_oof_generation_does_not_mutate_observation_mask():
    adapter = _ConstantAdapter("base")
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    target = np.full((1, 40), 10.0)
    mask = np.ones(40, dtype=bool)
    mask[35] = False
    original_mask = mask.copy()

    generate_oof_residuals(
        adapter,
        target,
        dates,
        ("y",),
        mask,
        horizon=5,
        min_training_history=32,
    )

    assert np.array_equal(mask, original_mask)


def test_observed_value_filter_does_not_mutate_shared_mask():
    mask = np.array([True, True, True])
    original_mask = mask.copy()

    _observed_values(
        np.array([[1.0, np.nan, 3.0]]),
        np.array([[1.0, 2.0, np.inf]]),
        mask,
    )

    assert np.array_equal(mask, original_mask)


def test_candidate_selection_uses_mae_then_rmse_then_stable_key():
    scores = [
        CandidateScore("worse", 1.0, 3.0, 2, 0.1),
        CandidateScore("best_rmse", 1.0, 1.0, 2, 0.1),
        CandidateScore("best_mae", 0.9, 9.0, 2, 0.1),
        CandidateScore("failed", None, None, 0, 0.1, "load failure"),
    ]

    assert select_best_candidate(scores).model_key == "best_mae"
    assert select_best_candidate(scores[:2]).model_key == "best_rmse"
    assert select_best_candidate([scores[-1]]) is None


def test_residual_request_contains_aligned_history_and_future_base_covariates():
    adapter = _ConstantAdapter("t0_gamma", forecast_value=0.5)
    adapter.capabilities = ModelCapabilities(
        model_key="t0_gamma",
        display_name="t0_gamma",
        checkpoint_id="",
        supports_known_future_covariates=True,
    )
    service = _Service({"t0_gamma": adapter})
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    residual_data = SimpleNamespace(
        residuals=np.ones((1, 12)),
        base_predictions=np.full((1, 12), 8.0),
        index=dates,
        n_observations=12,
    )
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=4, freq="D")
    future_base = np.full((1, 4), 9.0)

    prediction = _predict_residual_model(
        service,
        "t0_gamma",
        residual_data,
        future_base,
        future_dates,
        ("y",),
        RuntimeConfig(device="cpu"),
        (0.1, 0.5, 0.9),
    )

    request = adapter.requests[-1]
    assert request.target_history.shape == (1, 12)
    assert request.known_covariates_history.shape == (1, 12)
    assert request.known_covariates_future.shape == (1, 4)
    assert request.history_index.equals(dates)
    assert request.future_index.equals(future_dates)
    assert np.array_equal(prediction, future_base * 0 + 0.5)


def test_residual_candidates_exclude_unverified_advertised_covariate_support():
    models = {
        "timestral_209m": _ConstantAdapter("timestral_209m"),
        "timesfm_2_5": _ConstantAdapter("timesfm_2_5"),
        "timesfm_3": _ConstantAdapter("timesfm_3"),
        "tirex_2": _ConstantAdapter("tirex_2"),
        "t0_gamma": _ConstantAdapter("t0_gamma"),
    }
    # These fixtures advertise the flag, but only source-verified consumers
    # are allowed into the residual-model candidate pool.
    for adapter in models.values():
        adapter.capabilities = ModelCapabilities(
            model_key=adapter.capabilities.model_key,
            display_name=adapter.capabilities.display_name,
            checkpoint_id="",
            supports_known_future_covariates=True,
        )

    assert _foundation_residual_candidates(_Service(models)) == [
        "timesfm_3",
        "tirex_2",
        "t0_gamma",
    ]


def test_autonomous_hybrid_learns_bias_and_reports_holdout_improvement():
    base = _ConstantAdapter("constant_base")
    service = _Service({"constant_base": base})

    result = ForecastingService.run_autonomous_hybrid(
        service,
        _parsed_data(100),
        horizon=5,
        runtime_config=RuntimeConfig(device="cpu"),
        n_windows=3,
    )

    assert result.status == "hybrid"
    assert result.residual_model_key in {"native_exog", "native_arx"}
    assert np.allclose(result.result.point, result.base_forecast + result.residual_forecast)
    assert result.holdout_metrics["hybrid_mae"] < result.holdout_metrics["base_mae"]
    assert result.result.quantiles is None
    assert result.independent_holdout


def test_insufficient_oof_residuals_return_explicit_base_only_fallback():
    base = _ConstantAdapter("constant_base")
    service = _Service({"constant_base": base})

    result = ForecastingService.run_autonomous_hybrid(
        service,
        _parsed_data(40),
        horizon=1,
        runtime_config=RuntimeConfig(device="cpu"),
        n_windows=3,
    )

    assert result.status == "base_only"
    assert result.residual_model_key is None
    assert np.all(result.residual_forecast == 0)
    assert result.result.metadata["status"] == "base_only"
    assert any("base-only" in warning for warning in result.warnings)


def test_oof_preparation_failure_falls_back_and_keeps_base_holdout_score():
    base = _ConstantAdapter("constant_base")
    service = _Service({"constant_base": base})

    with patch(
        "forecasting.hybrid.generate_oof_residuals",
        side_effect=ValueError("simulated OOF preparation failure"),
    ):
        result = ForecastingService.run_autonomous_hybrid(
            service,
            _parsed_data(100),
            horizon=5,
            runtime_config=RuntimeConfig(device="cpu"),
            n_windows=3,
        )

    assert result.status == "base_only"
    assert result.residual_model_key is None
    assert result.holdout_metrics is not None
    assert result.holdout_metrics["base_mae"] == 2.0
    assert result.holdout_metrics["hybrid_mae"] is None
    assert any("simulated OOF preparation failure" in warning for warning in result.warnings)


def test_worse_residual_candidate_is_rejected_in_favor_of_base_forecast():
    base = _ConstantAdapter("constant_base")
    service = _Service({"constant_base": base})

    with patch(
        "forecasting.hybrid._predict_residual_model",
        return_value=np.full((1, 5), -10.0),
    ):
        result = ForecastingService.run_autonomous_hybrid(
            service,
            _parsed_data(100),
            horizon=5,
            runtime_config=RuntimeConfig(device="cpu"),
            n_windows=3,
        )

    assert result.status == "base_only"
    assert result.residual_model_key is None
    assert result.residual_validation is None
    assert np.array_equal(result.result.point, result.base_forecast)
    assert result.holdout_metrics is not None
    assert result.holdout_metrics["hybrid_mae"] is None
    assert any("did not improve the base model" in warning for warning in result.warnings)
