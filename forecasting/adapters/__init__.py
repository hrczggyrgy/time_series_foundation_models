"""Adapters package init."""
from forecasting.adapters.timesfm25 import TimesFM25Adapter
from forecasting.adapters.timesfm3 import TimesFM3Adapter
from forecasting.adapters.naive import NaiveAdapter, SeasonalNaiveAdapter
from forecasting.adapters.t0 import T0GammaAdapter
from forecasting.adapters.timestral import TimestralAdapter

__all__ = [
    "TimesFM25Adapter",
    "TimesFM3Adapter",
    "NaiveAdapter",
    "SeasonalNaiveAdapter",
    "T0GammaAdapter",
    "TimestralAdapter",
]