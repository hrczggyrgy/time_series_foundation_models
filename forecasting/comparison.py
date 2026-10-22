"""Model comparison orchestration and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from forecasting.contracts import ForecastRequest
from forecasting.evaluation.backtesting import BacktestResult

if TYPE_CHECKING:
    from forecasting.service import ForecastJob


@dataclass(frozen=True)
class ModelComparisonItem:
    """Result for a single model in a comparison."""

    model_key: str
    display_name: str
    forecast: ForecastJob | None
    backtest: BacktestResult | None
    error: str | None
    elapsed_seconds: float
    context_used: int | None  # Actual context length used by the model
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ComparisonReport:
    """Complete comparison report across multiple models."""

    comparison_id: str
    model_results: dict[str, ModelComparisonItem]  # model_key -> ModelComparisonItem
    horizon: int
    evaluation_origins: list[pd.Timestamp]  # Train end dates for each backtest window
    settings: dict[str, Any]  # Comparison configuration
    data_fingerprint: str  # Dataset fingerprint for cache invalidation
    timestamp: str  # ISO timestamp of comparison run
