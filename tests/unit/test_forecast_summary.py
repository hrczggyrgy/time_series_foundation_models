"""Unit tests for forecast summary text."""

import pandas as pd

from app import explain_forecast_results


def test_forecast_summary_uses_profile_context_when_metadata_omits_it():
    profile = {
        "basic_stats": {
            "context_used": 120,
            "all_non_negative": False,
        }
    }
    forecast = pd.DataFrame({"Forecast": [10.0, 12.0]})
    metadata = {"horizon": 2}

    summary = explain_forecast_results(profile, forecast, metadata)

    assert any("used 120 recent data point(s)" in line for line in summary)
