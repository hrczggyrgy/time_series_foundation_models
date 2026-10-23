"""Adapters package init."""
from forecasting.adapters.chronos import ChronosAdapter
from forecasting.adapters.moirai import MoiraiAdapter
from forecasting.adapters.stl import STLAdapter
from forecasting.adapters.t0 import T0GammaAdapter
from forecasting.adapters.timesfm3 import TimesFM3Adapter
from forecasting.adapters.timesfm25 import TimesFM25Adapter
from forecasting.adapters.timestral import TimestralAdapter
from forecasting.adapters.tirex2 import TiRex2Adapter

__all__ = [
    "ChronosAdapter",
    "MoiraiAdapter",
    "TimesFM25Adapter",
    "TimesFM3Adapter",
    "STLAdapter",
    "T0GammaAdapter",
    "TimestralAdapter",
    "TiRex2Adapter",
]