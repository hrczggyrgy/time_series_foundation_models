"""Tests for application-level chart builders."""

import math

import pytest

import app
from app import build_metric_scatter


def test_metric_scatter_plots_available_model_pairs_and_labels_axes():
    metrics = {
        "timesfm_2_5": {"MAE": 2.0, "RMSE": 3.0},
        "chronos": {"MAE": 1.5, "RMSE": 2.5},
        "naive": {"MAE": 4.0, "RMSE": math.nan},
    }

    fig = build_metric_scatter(
        metrics,
        "MAE",
        "RMSE",
        {"timesfm_2_5": "TimesFM 2.5", "chronos": "Chronos"},
    )

    assert len(fig.data) == 2
    assert list(fig.data[0].x) == [2.0]
    assert list(fig.data[0].y) == [3.0]
    assert fig.data[0].name == "TimesFM 2.5"
    assert fig.data[1].name == "Chronos"
    assert fig.layout.xaxis.title.text == "MAE · lower is better"
    assert fig.layout.yaxis.title.text == "RMSE · lower is better"


def test_metric_scatter_rejects_same_metric_on_both_axes():
    with pytest.raises(ValueError, match="different accuracy metric"):
        build_metric_scatter({"model": {"MAE": 1.0}}, "MAE", "MAE")


def test_clear_forecast_results_removes_outputs_from_previous_dataset(monkeypatch):
    state = {
        "forecast_jobs": {"old_model": object()},
        "backtest_results": {"old_model": object()},
        "forecast_run_errors": {"old_model": "old failure"},
        "run_metadata": {"horizon": 12},
    }
    monkeypatch.setattr(app.st, "session_state", state)

    app.clear_forecast_results()

    assert state["forecast_jobs"] == {}
    assert state["backtest_results"] == {}
    assert state["forecast_run_errors"] == {}
    assert state["run_metadata"] is None
