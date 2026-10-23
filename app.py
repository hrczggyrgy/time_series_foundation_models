# ============================================================
# TimesFM 2.5 Forecasting Dashboard — Modular Architecture
# Multi-model forecasting workbench with clean separation of concerns
# ============================================================

import hashlib
import io
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import linregress
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)

# Import our modular forecasting components
from forecasting.service import ForecastingService, generate_demo_data
from forecasting.registry import ModelRegistry, get_registry
from forecasting.contracts import RuntimeConfig
from forecasting.reporting.exports import generate_run_id
from forecasting.data.parsing import ValidationReport

# Constants
MIN_CONTEXT_LENGTH = 32
MODEL_CONTEXT_LIMIT = 16384
DEFAULT_MAX_CONTEXT = 2048
DEFAULT_HORIZON = 30
MAX_UI_HORIZON = 512
TORCH_THREADS = 4

FREQ_LABEL_MAP = {
    "T": "Minutely", "MIN": "Minutely", "H": "Hourly",
    "D": "Daily", "B": "Business Daily", "W": "Weekly",
    "M": "Monthly", "Q": "Quarterly", "A": "Annual", "Y": "Annual",
    "Irregular": "Irregular", "Unknown": "Unknown",
}

QUANTILE_PRESETS = {
    "50 % (P25–P75)": 0.50,
    "60 % (P20–P80)": 0.60,
    "80 % (P10–P90)": 0.80,
    "90 % (P05–P95)": 0.90,
}

DEFAULT_QUANTILE_LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    color: #172033;
}
[data-testid="stAppViewContainer"] { background: #f5f7fa; }
[data-testid="stSidebar"] {
    background: #fff;
    border-right: 1px solid #e7ebf1;
}
.block-container { max-width: 1480px; padding-top: 1.6rem; padding-bottom: 2.5rem; }
.hero-title { color: #172033; font-size: 1.85rem; font-weight: 720; letter-spacing: -0.04em; line-height: 1.2; }
.hero-subtitle { color: #687386; font-size: 0.98rem; margin-top: 0.35rem; }
[data-testid="metric-container"] {
    background: #fff;
    border: 1px solid #e7ebf1;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 2px 8px rgba(24, 39, 75, 0.035);
}
.callout, .callout-warn, .callout-good {
    border-radius: 10px;
    padding: 12px 16px;
    margin: 0 0 1rem;
}
.callout { background: #fff; border: 1px solid #dce7f3; color: #24364b; }
.callout-warn { background: #fffaf2; border: 1px solid #f2dfbe; color: #7c4a12; }
.callout-good { background: #f4faf7; border: 1px solid #d3e9dc; color: #20583d; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.03em; }
.badge-green { background: #d1fae5; color: #065f46; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-red { background: #fee2e2; color: #991b1b; }
.section-divider { border: none; border-top: 1px solid #e4e9f0; margin: 1.15rem 0 1rem; }
.small-note { color: #687386; font-size: 0.9rem; }
[data-testid="stRadio"] [role="radiogroup"] { gap: 0.45rem; }
[data-testid="stRadio"] [role="radiogroup"] label {
    padding: 0.35rem 0.8rem;
    border: 1px solid #e1e6ed;
    border-radius: 8px;
    background: #fff;
}
.stButton button, .stDownloadButton button { border-radius: 8px; font-weight: 600; }
.stButton button[kind="primary"] { background: #1d3557; border-color: #1d3557; }
div[data-testid="stDataFrame"] { border: 1px solid #e7ebf1; border-radius: 10px; overflow: hidden; }
</style>
"""

COMPARISON_METRICS = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "MAPE": "MAPE",
    "MASE": "MASE",
    "CRPS": "CRPS",
    "Pinball loss": "Pinball",
}


@dataclass(frozen=True)
class ForecastConfigUI:
    horizon: int
    interval_coverage: float
    business_days_only: bool
    backtest_mode: bool
    n_backtest_windows: int
    autopilot: bool
    auto_fix_data: bool
    model_key: str


def init_session_state() -> None:
    defaults = {
        "raw_signature": None,
        "df": None,
        "date_col": None,
        "target_cols": None,
        "past_covariate_cols": [],
        "known_covariate_cols": [],
        "validation_report": None,
        "profile_results": None,
        "forecast_job": None,
        "forecast_run_errors": {},
        "backtest_result": None,
        "run_metadata": None,
        "analysis_ready": False,
        "service": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Initialize service
    if st.session_state.service is None:
        st.session_state.service = ForecastingService()


def clear_results(keep_signature: bool = True) -> None:
    signature = st.session_state.get("raw_signature") if keep_signature else None
    for key in [
        "df", "date_col", "target_cols", "past_covariate_cols",
        "known_covariate_cols", "validation_report", "profile_results",
        "forecast_job", "backtest_result", "run_metadata", "analysis_ready",
        "forecast_run_errors",
    ]:
        st.session_state[key] = None if key != "analysis_ready" else False
    if keep_signature:
        st.session_state["raw_signature"] = signature


@st.cache_data(ttl=3600, show_spinner=False)
def sample_csv_bytes() -> bytes:
    return generate_demo_data().head(90).to_csv(index=False).encode()


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def infer_column_candidates(df: pd.DataFrame) -> tuple[int, int]:
    date_idx = 0
    target_idx = min(1, len(df.columns) - 1)
    cols = list(df.columns)
    for i, col in enumerate(cols):
        name = str(col).lower()
        if any(token in name for token in ["date", "time", "timestamp", "ds"]):
            date_idx = i
            break
    for i, col in enumerate(cols):
        name = str(col).lower()
        if any(token in name for token in ["value", "target", "y", "sales", "price", "demand", "volume"]):
            target_idx = i
            break
    if len(cols) >= 2 and date_idx == target_idx:
        for i in range(len(cols)):
            if i != date_idx:
                target_idx = i
                break
    return date_idx, target_idx


def canonical_freq(freq: str) -> str:
    if not freq:
        return "Unknown"
    freq = freq.upper().split("-")[0]
    mapping = {"MS": "M", "ME": "M", "BMS": "M", "BME": "M", "QS": "Q", "QE": "Q", "YS": "A", "YE": "A", "AS": "A", "AE": "A"}
    return mapping.get(freq, freq)


def infer_frequency_safe(series: pd.Series) -> str:
    if len(series) < 3:
        return "Unknown"
    try:
        inferred = pd.infer_freq(pd.DatetimeIndex(series))
        if inferred:
            return canonical_freq(inferred)
    except Exception:
        pass
    deltas = pd.Series(pd.DatetimeIndex(series).sort_values()).diff().dropna()
    if deltas.empty:
        return "Unknown"
    mode_delta = deltas.mode().iloc[0]
    thresholds = [
        (pd.Timedelta(minutes=1), "T"), (pd.Timedelta(hours=1), "H"),
        (pd.Timedelta(days=1), "D"), (pd.Timedelta(weeks=1), "W"),
        (pd.Timedelta(days=28), "M"), (pd.Timedelta(days=90), "Q"),
        (pd.Timedelta(days=365), "A"),
    ]
    for td, code in thresholds:
        if abs(mode_delta - td) <= td * 0.12:
            return code
    return "Irregular"


def dominant_resample_rule(ds: pd.Series) -> str | None:
    if len(ds) < 3:
        return None
    deltas = pd.Series(pd.DatetimeIndex(ds).sort_values()).diff().dropna()
    if deltas.empty:
        return None
    mode_delta = deltas.mode().iloc[0]
    mapping = [
        (pd.Timedelta(minutes=1), "T"), (pd.Timedelta(hours=1), "H"),
        (pd.Timedelta(days=1), "D"), (pd.Timedelta(weeks=1), "W"),
        (pd.Timedelta(days=30), "M"), (pd.Timedelta(days=90), "Q"),
    ]
    return min(mapping, key=lambda x: abs(x[0] - mode_delta))[1]


def seasonal_candidates(freq: str, n_obs: int) -> list[int]:
    freq = canonical_freq(freq)
    if freq in ("T", "MIN"): candidates = [60, 1440]
    elif freq == "H": candidates = [24, 168, 720]
    elif freq in ("D", "B", "IRREGULAR"): candidates = [7, 30, 365]
    elif freq == "W": candidates = [52, 26]
    elif freq == "M": candidates = [12, 6]
    elif freq == "Q": candidates = [4, 8]
    else: candidates = [7, 30]
    return [p for p in candidates if n_obs >= 2 * p]


def recommended_horizon(freq: str, n_obs: int) -> tuple[int, str]:
    freq = canonical_freq(freq)
    if freq == "T": h, label = 60, "60 minutes"
    elif freq == "H": h, label = 48, "48 hours"
    elif freq in ("D", "B"): h, label = 30, "30 days"
    elif freq == "W": h, label = 26, "26 weeks"
    elif freq == "M": h, label = 12, "12 months"
    elif freq == "Q": h, label = 8, "8 quarters"
    else: h, label = 30, "30 steps"
    h = min(h, max(7, min(MAX_UI_HORIZON, max(7, n_obs // 3))))
    return h, label


def analyze_trend(values: np.ndarray) -> dict[str, Any]:
    x = np.arange(len(values))
    slope, intercept, r_value, p_value, std_err = linregress(x, values)
    direction = "Flat"
    if p_value < 0.05:
        direction = "Increasing" if slope > 0 else "Decreasing"
    return {
        "slope": float(slope), "intercept": float(intercept),
        "p_value": float(p_value), "r_squared": float(r_value**2),
        "std_err": float(std_err), "direction": direction,
        "significant": bool(p_value < 0.05),
    }


def test_stationarity(values: np.ndarray) -> dict[str, Any]:
    try:
        result = adfuller(values, maxlag=min(12, max(1, len(values) // 4)))
        p_value = float(result[1])
        is_stationary = p_value < 0.05
        return {
            "adf_statistic": float(result[0]), "p_value": p_value,
            "is_stationary": is_stationary,
            "interpretation": "Stationary" if is_stationary else "Non-stationary",
        }
    except Exception as exc:
        return {"adf_statistic": math.nan, "p_value": math.nan, "is_stationary": None, "interpretation": f"ADF failed: {exc}"}


def analyze_seasonality(values: np.ndarray, freq: str) -> dict[str, Any]:
    candidates = seasonal_candidates(freq, len(values))
    if not candidates:
        return {"seasonal_strength": None, "period": None, "decomposition": None, "additional_periods": [], "interpretation": "Not enough data for seasonality analysis.", "strength_interpretation": "N/A"}
    results = []
    for period in candidates:
        try:
            decomp = seasonal_decompose(values, period=period, model="additive")
            seasonal = np.asarray(decomp.seasonal, dtype=float)
            resid = np.asarray(decomp.resid, dtype=float)
            s_var = float(np.nanvar(seasonal))
            r_var = float(np.nanvar(resid[~np.isnan(resid)])) if np.any(~np.isnan(resid)) else 0.0
            strength = s_var / (s_var + r_var) if (s_var + r_var) > 0 else 0.0
            label = "Strong" if strength > 0.6 else ("Moderate" if strength > 0.3 else "Weak" if strength > 0.1 else "Negligible")
            results.append({"period": int(period), "strength": float(strength), "label": label, "decomposition": {"trend": np.asarray(decomp.trend, dtype=float), "seasonal": seasonal, "resid": resid}})
        except Exception:
            continue
    if not results:
        return {"seasonal_strength": None, "period": None, "decomposition": None, "additional_periods": [], "interpretation": "Seasonality analysis failed for all candidate periods.", "strength_interpretation": "N/A"}
    results.sort(key=lambda r: r["strength"], reverse=True)
    primary = results[0]
    extras = [{"period": r["period"], "strength": r["strength"], "label": r["label"]} for r in results[1:]]
    return {"seasonal_strength": primary["strength"], "period": primary["period"], "decomposition": primary["decomposition"], "additional_periods": extras, "interpretation": f"{primary['label']} seasonality detected with period {primary['period']}.", "strength_interpretation": primary["label"]}


def profile_timeseries(df: pd.DataFrame, target_cols: list[str]) -> dict[str, Any]:
    """Profile all target columns."""
    profiles = {}
    for i, col in enumerate(target_cols):
        # Use normalized 'y' column if available (after prepare_dataframe normalization)
        actual_col = "y" if "y" in df.columns and i == 0 else col
        values = df[actual_col].to_numpy(dtype=float)
        freq = infer_frequency_safe(df["ds"])
        gaps, gap_share = detect_irregular_gaps(df["ds"])
        outliers = detect_outliers(df[actual_col])
        ctx_used = min(len(df), MODEL_CONTEXT_LIMIT)
        # Store with normalized key for internal consistency
        normalized_key = "y" if i == 0 and "y" in df.columns else col
        profiles[normalized_key] = {
            "basic_stats": {
                "n_observations": int(len(df)),
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "min": float(np.min(values)), "max": float(np.max(values)),
                "all_non_negative": bool(np.all(values >= 0)),
                "date_range": f"{df['ds'].min().date()} → {df['ds'].max().date()}",
                "inferred_frequency": freq,
                "freq_label": FREQ_LABEL_MAP.get(freq, freq),
                "irregular_gap_count": gaps,
                "irregular_gap_share": gap_share,
                "outlier_count": outliers,
                "context_used": ctx_used,
                "context_utilization": ctx_used / MODEL_CONTEXT_LIMIT,
            },
            "trend": analyze_trend(values),
            "stationarity": test_stationarity(values),
            "seasonality": analyze_seasonality(values, freq),
        }
    # primary_target uses normalized column name for internal use
    primary_internal = "y" if "y" in df.columns and target_cols else (target_cols[0] if target_cols else None)
    return {"targets": profiles, "primary_target": primary_internal}


def detect_irregular_gaps(ds: pd.Series) -> tuple[int, float]:
    if len(ds) < 3: return 0, 0.0
    deltas = pd.Series(pd.DatetimeIndex(ds).sort_values()).diff().dropna()
    if deltas.empty: return 0, 0.0
    mode_delta = deltas.mode().iloc[0]
    if mode_delta.total_seconds() == 0: return 0, 0.0
    irregular = int((deltas != mode_delta).sum())
    return irregular, irregular / max(1, len(deltas))


def detect_outliers(values: pd.Series, z_thresh: float = 3.5) -> int:
    clean = pd.Series(values).dropna()
    if len(clean) < 10: return 0
    median = clean.median()
    mad = np.median(np.abs(clean - median))
    if mad == 0: return 0
    modified_z = 0.6745 * (clean - median) / mad
    return int((np.abs(modified_z) > z_thresh).sum())


def readiness_badge(report: ValidationReport) -> str:
    errors = len(report.errors)
    warnings_n = len(report.warnings)
    if errors: return f'<span class="badge badge-red"> {errors} error(s)</span>'
    if warnings_n: return f'<span class="badge badge-yellow">⚠ {warnings_n} warning(s)</span>'
    return '<span class="badge badge-green">✓ Ready</span>'


def explain_series(profile: dict[str, Any], report: ValidationReport) -> list[str]:
    stats = profile["basic_stats"]
    trend = profile["trend"]
    seas = profile["seasonality"]
    notes = []
    notes.append(f"The series contains {stats['n_observations']:,} observations with inferred frequency {stats['freq_label']}.")
    notes.append(f"Trend is {trend['direction'].lower()} and the strongest detected seasonal period is {seas['period'] or 'not available'}.")
    if stats["outlier_count"]:
        notes.append(f"About {stats['outlier_count']} potential outlier(s) were detected, so sharp jumps may widen uncertainty bands.")
    if report.warnings:
        notes.append("Warnings do not block forecasting, but they reduce trust and should be reviewed before production use.")
    return notes


def explain_forecast_results(profile: dict[str, Any], forecast_df: pd.DataFrame, meta: dict[str, Any]) -> list[str]:
    first_val = float(forecast_df["Forecast"].iloc[0])
    last_val = float(forecast_df["Forecast"].iloc[-1])
    direction = "upward" if last_val > first_val else "downward" if last_val < first_val else "flat"
    lo_col = [c for c in forecast_df.columns if c.startswith("Lower_")][0] if any(c.startswith("Lower_") for c in forecast_df.columns) else None
    hi_col = [c for c in forecast_df.columns if c.startswith("Upper_")][0] if any(c.startswith("Upper_") for c in forecast_df.columns) else None
    text = [f"The forecast points {direction} over the selected horizon of {meta['horizon']} step(s)."]
    if lo_col and hi_col:
        avg_band = float((forecast_df[hi_col] - forecast_df[lo_col]).mean())
        text.append(f"Average uncertainty band width is {avg_band:.2f}; larger numbers indicate less certainty.")
    context_used = meta.get("ctx_used", profile["basic_stats"]["context_used"])
    text.append(f"The model used {context_used:,} recent data point(s) out of the {MODEL_CONTEXT_LIMIT:,}-point maximum context.")
    if profile["basic_stats"]["all_non_negative"]:
        text.append("Because the historical series is non-negative, the model was configured to keep future values non-negative as well.")
    return text


def explain_backtest_results(metrics: dict[str, float], meta: dict[str, Any]) -> list[str]:
    text = [f"Backtest MAE is {metrics['MAE']:.3f} and RMSE is {metrics['RMSE']:.3f}; lower values are better."]
    text.append(f"Interval coverage is {metrics['Coverage']:.1f}% against a nominal {meta['interval_pct']}% interval.")
    if not math.isnan(metrics.get("MAPE", math.nan)):
        text.append(f"MAPE is {metrics['MAPE']:.2f}%, which is easier to interpret across scales.")
    if metrics["Coverage"] < max(35, meta["interval_pct"] - 20):
        text.append("Coverage is meaningfully below the nominal interval, so the forecast intervals may be too narrow.")
    elif metrics["Coverage"] > min(99, meta["interval_pct"] + 15):
        text.append("Coverage is far above the nominal interval, so the forecast intervals may be conservative.")
    return text


def style_figure(fig: go.Figure, title: str, height: int, yaxis_title: str = "Value") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=16, color="#172033")),
        height=height,
        margin=dict(l=18, r=18, t=54, b=18),
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#334155", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#edf1f5",
        zeroline=False,
        linecolor="#d9e0e8",
        ticks="outside",
        tickcolor="#cbd5e1",
    )
    fig.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor="#edf1f5",
        zeroline=False,
        linecolor="#d9e0e8",
        ticks="outside",
        tickcolor="#cbd5e1",
    )
    return fig


def build_metric_scatter(
    metrics: Mapping[str, Mapping[str, float | None]],
    x_metric: str,
    y_metric: str,
    model_labels: Mapping[str, str] | None = None,
) -> go.Figure:
    """Build a two-metric scatterplot, omitting unavailable metric values."""
    if x_metric == y_metric:
        raise ValueError("Choose a different accuracy metric for each axis.")

    colors = [
        "#315a8a", "#d17b49", "#3f806d", "#8568a8", "#bc5864",
        "#548c9d", "#9b8442", "#56677c", "#5e9278", "#a65f83",
    ]
    points = []
    for model, values in metrics.items():
        x_value = values.get(x_metric)
        y_value = values.get(y_metric)
        if (
            x_value is not None
            and y_value is not None
            and math.isfinite(x_value)
            and math.isfinite(y_value)
        ):
            points.append((model, x_value, y_value))
    labels = model_labels or {}
    metric_names = {value: name for name, value in COMPARISON_METRICS.items()}
    fig = go.Figure()
    for index, (model, x_value, y_value) in enumerate(points):
        label = labels.get(model, model)
        fig.add_trace(
            go.Scatter(
                x=[x_value],
                y=[y_value],
                name=label,
                mode="markers",
                marker=dict(
                    size=12,
                    color=colors[index % len(colors)],
                    line=dict(color="#ffffff", width=1.5),
                    opacity=0.92,
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>{x_metric}: %{{x:.4g}}"
                    f"<br>{y_metric}: %{{y:.4g}}"
                    "<extra></extra>"
                ),
            ),
        )
    fig.update_layout(
        template="plotly_white",
        height=470,
        margin=dict(l=24, r=24, t=54, b=24),
        hovermode="closest",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#334155", size=12),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
    )
    fig.update_xaxes(
        title=f"{metric_names.get(x_metric, x_metric)} · lower is better",
        showgrid=True,
        gridcolor="#edf1f5",
        zeroline=False,
        linecolor="#d9e0e8",
    )
    fig.update_yaxes(
        title=f"{metric_names.get(y_metric, y_metric)} · lower is better",
        showgrid=True,
        gridcolor="#edf1f5",
        zeroline=False,
        linecolor="#d9e0e8",
    )
    return fig


def plot_forecast(hist_df: pd.DataFrame, future_dates: pd.DatetimeIndex, fc: dict[str, Any], interval_label: str, actuals: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["ds"], y=hist_df["y"], mode="lines", name="Historical", line=dict(color="#6366f1", width=2.2)))
    if actuals is not None:
        fig.add_trace(go.Scatter(x=future_dates, y=actuals, mode="lines+markers", name="Actual", line=dict(color="#059669", width=2, dash="dot"), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=future_dates, y=fc["q_high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_dates, y=fc["q_low"], mode="lines", fill="tonexty", name=f"Interval ({interval_label.split('(')[0].strip()})", line=dict(width=0), fillcolor="rgba(99,102,241,0.12)"))
    fig.add_trace(go.Scatter(x=future_dates, y=fc["point"], mode="lines", name="Forecast", line=dict(color="#111827", width=3)))
    if len(future_dates) > 0:
        fig.add_vline(x=future_dates[0], line_dash="dot", line_color="#94a3b8")
    return style_figure(fig, "Forecast", 520, "Value")


def plot_decomposition(dates: pd.DatetimeIndex, decomp: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for (name, vals), color in zip([("Trend", decomp["trend"]), ("Seasonal", decomp["seasonal"]), ("Residual", decomp["resid"])], ["#4f46e5", "#f59e0b", "#10b981"]):
        fig.add_trace(go.Scatter(x=dates, y=vals, mode="lines", name=name, line=dict(color=color, width=1.7)))
    return style_figure(fig, "Seasonal decomposition", 340, "Component value")


def plot_context_window(df: pd.DataFrame, ctx_used: int) -> go.Figure:
    fig = go.Figure()
    if len(df) > ctx_used:
        fig.add_trace(go.Scatter(x=df["ds"].iloc[:-ctx_used], y=df["y"].iloc[:-ctx_used], mode="lines", name="Outside context", line=dict(color="#d1d5db", width=1.2)))
    fig.add_trace(go.Scatter(x=df["ds"].iloc[-ctx_used:], y=df["y"].iloc[-ctx_used:], mode="lines", name=f"Context window ({ctx_used:,} pts)", line=dict(color="#4f46e5", width=2)))
    return style_figure(fig, "Context used by model", 280, "Value")


def plot_error_distribution(fdf: pd.DataFrame) -> go.Figure:
    errors = fdf["Actual"] - fdf["Forecast"]
    fig = go.Figure(go.Histogram(x=errors, nbinsx=24, marker_color="#4f46e5", marker_line_color="#312e81", marker_line_width=0.6))
    fig.add_vline(x=0, line_dash="dot", line_color="#dc2626")
    fig = style_figure(fig, "Error distribution", 320, "Count")
    fig.update_xaxes(title="Error (actual - forecast)")
    return fig


def plot_rolling_backtest(windows: list[dict[str, Any]], hist_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    ctx = hist_df.tail(min(180, len(hist_df)))
    fig.add_trace(go.Scatter(x=ctx["ds"], y=ctx["y"], mode="lines", name="Historical", line=dict(color="#6366f1", width=1.5)))
    colors = ["#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
    for w in windows:
        c = colors[(w["window"] - 1) % len(colors)]
        fig.add_trace(go.Scatter(x=w["test_dates"], y=w["actuals"], mode="lines", name=f"W{w['window']} Actual", line=dict(color=c, width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=w["test_dates"], y=w["point"], mode="lines", name=f"W{w['window']} Forecast", line=dict(color=c, width=2)))
    return style_figure(fig, "Rolling backtest", 420, "Value")


def render_sidebar(profile_results: dict[str, Any] | None) -> tuple[ForecastConfigUI, str, list[str]]:
    st.sidebar.markdown("## ⚙️ Settings")

    service: ForecastingService = st.session_state.service

    # Get all available models (exclude experimental by default)
    all_models = service.list_models(include_experimental=False)
    model_options = {m["model_key"]: m["display_name"] for m in all_models}
    model_keys = list(model_options.keys())

    # Multi-model selector for comparison
    default_models = ["timesfm_2_5"] if "timesfm_2_5" in model_keys else [model_keys[0]]
    # Filter session state to only include currently available models
    stored_models = st.session_state.get("selected_models", default_models)
    stored_models = [m for m in stored_models if m in model_keys]
    if not stored_models:
        stored_models = default_models

    selected_models = st.sidebar.multiselect(
        "Models to compare",
        options=model_keys,
        format_func=lambda k: model_options[k],
        default=stored_models,
        help="Select one or more models to run and compare. Experimental models are hidden by default.",
    )

    if not selected_models:
        st.sidebar.warning("Select at least one model.")
        st.stop()

    # Store in session state for other tabs
    st.session_state.selected_models = selected_models

    # Primary model (first selected) for single-model operations
    selected_model = selected_models[0]
    model_info = next((m for m in all_models if m["model_key"] == selected_model), None)

    # Data source
    data_source = st.sidebar.radio("Data source", ["Synthetic demo data", "Upload CSV"], label_visibility="collapsed")
    st.sidebar.markdown("---")

    # Horizon
    suggested = DEFAULT_HORIZON
    suggestion_label = "30 steps"
    if profile_results is not None:
        primary = profile_results.get("primary_target")
        if primary and primary in profile_results.get("targets", {}):
            stats = profile_results["targets"][primary]["basic_stats"]
            suggested, suggestion_label = recommended_horizon(stats["inferred_frequency"], stats["n_observations"])

    autopilot = st.sidebar.toggle("Use recommended settings", value=True, help="Applies recommended horizon and safer defaults after analysis.")
    horizon = st.sidebar.slider("Horizon (steps ahead)", 7, MAX_UI_HORIZON, suggested if autopilot else DEFAULT_HORIZON, help=f"Recommended: {suggestion_label}.", disabled=autopilot)

    # Interval
    interval_label = st.sidebar.selectbox("Prediction interval", list(QUANTILE_PRESETS.keys()), index=2, help="Wider intervals are more conservative.")
    interval_coverage = QUANTILE_PRESETS[interval_label]

    # Advanced options
    business_days_only = False
    backtest_mode = True
    n_backtest_windows = 3
    use_full_context = False
    auto_fix_data = True

    with st.sidebar.expander("Advanced options", expanded=False):
        auto_fix_data = st.checkbox("Auto-fix duplicates, gaps, irregular spacing", value=True)
        business_days_only = st.checkbox("Business days only", value=False, help="Skip weekends for daily data.")
        backtest_mode = st.checkbox("Backtesting mode", value=True, help="Uses last horizon as holdout for accuracy metrics.")
        n_backtest_windows = st.slider("Rolling windows", 1, 5, 3, help="More windows = more robust but slower.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Reset session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.download_button("Download sample CSV", data=sample_csv_bytes(), file_name="sample_timeseries.csv", mime="text/csv", use_container_width=True, help="Two-column example with enough rows for diagnostics and backtesting.")

    # Model info caption
    if model_info:
        caps = model_info
        max_ctx = f"{caps['max_context']:,}" if caps['max_context'] else 'N/A'
        max_hz = f"{caps['max_horizon']:,}" if caps['max_horizon'] else 'N/A'
        exp_str = " ⚠️ EXPERIMENTAL" if caps.get('experimental', False) else ""
        st.sidebar.caption(f"""Primary model: `{model_options[selected_model]}`
License notice: {caps['license_notice']}
Max context: {max_ctx}
Max horizon: {max_hz}
Quantiles: {len(caps['native_quantiles'])} native{exp_str}
Compare with: {', '.join([model_options[m] for m in selected_models[1:]]) if len(selected_models) > 1 else 'none'}""")

    return ForecastConfigUI(
        horizon=horizon, interval_coverage=interval_coverage,
        business_days_only=business_days_only, backtest_mode=backtest_mode,
        n_backtest_windows=n_backtest_windows,
        autopilot=autopilot, auto_fix_data=auto_fix_data,
        model_key=selected_model,
    ), data_source, selected_models


def render_tab_data(data_source: str, ui_config: ForecastConfigUI) -> None:
    st.markdown("### 1. Data")
    service: ForecastingService = st.session_state.service

    if data_source == "Synthetic demo data":
        raw_df = generate_demo_data()
        signature = "demo-data-v2"
        st.markdown('<div class="callout">Using built-in demo data with trend, seasonality, and noise. Switch to <b>Upload CSV</b> to use your own series.</div>', unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader("Upload CSV with date column and numeric target(s)", type=["csv"], help="CSV should contain at least 32 rows for forecasting.")
        if uploaded is None:
            st.markdown('<div class="callout">Upload a CSV or use the sample file from the sidebar.</div>', unsafe_allow_html=True)
            return
        raw_df = load_csv(uploaded.getvalue())
        signature = hashlib.md5(uploaded.getvalue()).hexdigest()

    if signature != st.session_state.raw_signature:
        clear_results(keep_signature=False)
        st.session_state.raw_signature = signature

    cols = list(raw_df.columns)
    date_idx, target_idx = infer_column_candidates(raw_df)

    c1, c2 = st.columns(2)
    date_col = c1.selectbox("Date column", cols, index=date_idx, key=f"date_col_{signature}")
    target_col = c2.selectbox("Target column", cols, index=target_idx, key=f"target_col_{signature}")

    # Support multiple targets in future - for now single target
    target_cols = [target_col]

    with st.expander("Preview raw data", expanded=False):
        st.dataframe(raw_df.head(25), use_container_width=True, height=240)
        st.caption(f"Rows: {len(raw_df):,} · Columns: {len(raw_df.columns)}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    btn1, btn2 = st.columns([1, 1])
    prepare_clicked = btn1.button("Prepare data", type="primary", use_container_width=True)
    auto_prepare_clicked = btn2.button("Prepare with recommended settings", use_container_width=True)

    if prepare_clicked or auto_prepare_clicked:
        with st.spinner("Preparing, validating, and profiling the series…"):
            try:
                parsed, validation = service.prepare_data(
                    raw_df=raw_df,
                    date_col=date_col,
                    target_cols=target_cols,
                    past_covariate_cols=[],
                    known_covariate_cols=[],
                    auto_fix=ui_config.auto_fix_data,
                )
            except Exception as exc:
                st.error(str(exc))
                return

        st.session_state.df = parsed.df
        st.session_state.parsed_data = parsed  # Store full ParsedData for forecasting
        st.session_state.date_col = date_col
        st.session_state.target_cols = parsed.target_columns
        st.session_state.past_covariate_cols = parsed.past_covariate_columns
        st.session_state.known_covariate_cols = parsed.known_covariate_columns
        st.session_state.validation_report = validation
        st.session_state.profile_results = profile_timeseries(parsed.df, parsed.target_columns)
        st.session_state.forecast_job = None
        st.session_state.backtest_result = None
        st.session_state.run_metadata = None
        st.session_state.analysis_ready = True

    if st.session_state.validation_report is None or st.session_state.profile_results is None:
        return

    val = st.session_state.validation_report
    prof = st.session_state.profile_results
    primary = prof.get("primary_target")
    stats = prof["targets"][primary]["basic_stats"] if primary else {}

    st.markdown(f"**Data readiness:** {readiness_badge(val)}", unsafe_allow_html=True)
    for msg in val.errors: st.error(msg)
    for msg in val.warnings: st.warning(msg)
    for msg in val.notes: st.info(msg)
    if val.valid and not val.warnings:
        st.success("All checks passed.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{stats.get('n_observations', 0):,}")
    c2.metric("Frequency", stats.get("freq_label", "N/A"))
    c3.metric("Outliers ~", str(stats.get("outlier_count", 0)))
    c4.metric("Context used", f"{stats.get('context_used', 0):,}")
    c5.metric("Recommended horizon", str(recommended_horizon(stats.get("inferred_frequency", ""), stats.get("n_observations", 0))[0]))

    st.markdown("#### What this means")
    if primary and primary in prof["targets"]:
        for note in explain_series(prof["targets"][primary], val):
            st.markdown(f"- {note}")


def render_tab_diagnostics() -> None:
    st.markdown("### 2. Diagnostics")
    if st.session_state.profile_results is None:
        st.info("Prepare data first in the Data tab.")
        return

    prof = st.session_state.profile_results
    df = st.session_state.df
    primary = prof.get("primary_target")

    if not primary or primary not in prof["targets"]:
        st.error("Primary target not found in profile.")
        return

    target_profile = prof["targets"][primary]
    stats = target_profile["basic_stats"]
    trend = target_profile["trend"]
    stationarity = target_profile["stationarity"]
    seasonality = target_profile["seasonality"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Global linear trend", trend["direction"])
    c2.metric("Trend R²", f"{trend['r_squared']:.3f}")
    c3.metric("Stationary", "Yes" if stationarity.get("is_stationary") else "No")
    c4.metric("Seasonal period", str(seasonality["period"] or "N/A"))
    c5.metric("Seasonal strength", seasonality.get("strength_interpretation", "N/A"))

    st.caption(f"Date range: {stats['date_range']} · Inferred frequency: {stats['freq_label']} · Irregular gaps: {stats['irregular_gap_count']}")
    st.markdown(f"**Global linear trend:** {trend['direction']} (slope {trend['slope']:.4f}, p-value {trend['p_value']:.4f}, R² {trend['r_squared']:.3f})")
    st.markdown(f"**Stationarity:** {stationarity['interpretation']}")
    st.markdown(f"**Seasonality:** {seasonality['interpretation']}")

    if seasonality.get("additional_periods"):
        extras = ", ".join([f"{x['period']} ({x['label']})" for x in seasonality["additional_periods"]])
        if extras: st.markdown(f"**Additional candidate periods:** {extras}")

    if stats["all_non_negative"]:
        st.markdown('<div class="callout-good">All historical values are non-negative, so the model will keep the forecast non-negative as well.</div>', unsafe_allow_html=True)

    if stats["context_utilization"] < 0.05:
        st.markdown('<div class="callout-warn">This series is short relative to the model maximum context. Results may still be useful, but uncertainty should be interpreted conservatively.</div>', unsafe_allow_html=True)

    ctx = stats["context_used"]
    if st.session_state.run_metadata:
        ctx = st.session_state.run_metadata.get("ctx_used", ctx)
    st.plotly_chart(plot_context_window(df, int(ctx)), use_container_width=True)

    if seasonality.get("decomposition"):
        st.plotly_chart(plot_decomposition(df["ds"], seasonality["decomposition"]), use_container_width=True)

    with st.expander("Show full historical series"):
        fig = go.Figure()
        # Use normalized 'y' column for plotting
        plot_col = "y" if "y" in df.columns else primary
        fig.add_trace(go.Scatter(x=df["ds"], y=df[plot_col], mode="lines", line=dict(color="#4f46e5", width=1.6), name="Series"))
        fig = style_figure(fig, "Historical series", 300, "Value")
        fig.update_xaxes(title="Date")
        st.plotly_chart(fig, use_container_width=True)


def render_tab_forecast(ui_config: ForecastConfigUI, selected_models: list[str]) -> None:
    st.markdown("### 3. Forecast")
    service: ForecastingService = st.session_state.service

    df = st.session_state.df
    val = st.session_state.validation_report
    prof = st.session_state.profile_results

    if df is None or val is None or prof is None:
        st.info("Prepare data first in the Data tab.")
        return
    if not val.valid:
        st.error("Fix the data issues before forecasting.")
        return

    primary = prof.get("primary_target")
    if not primary or primary not in prof["targets"]:
        st.error("Primary target not found.")
        return

    stats = prof["targets"][primary]["basic_stats"]
    inferred_freq = stats["inferred_frequency"]
    rec_horizon, rec_label = recommended_horizon(inferred_freq, len(df))
    effective_horizon = rec_horizon if ui_config.autopilot else ui_config.horizon

    st.markdown(f'<div class="callout">Recommended horizon for <b>{stats["freq_label"]}</b> data is <b>{rec_label}</b>. Autopilot is <b>{"on" if ui_config.autopilot else "off"}</b>, so effective horizon is <b>{effective_horizon}</b> step(s).</div>', unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    run_manual = left.button("▶️ Run forecast", type="primary", use_container_width=True)
    run_autopilot = right.button("Run with recommended settings", use_container_width=True)
    should_run = run_manual or run_autopilot
    autopilot_now = ui_config.autopilot or run_autopilot

    if should_run:
        horizon = rec_horizon if autopilot_now else ui_config.horizon
        with st.spinner("Loading models and generating forecasts…"):
            runtime = RuntimeConfig(
                device="auto",
                dtype="float32",
                compile=False,
            )

            service: ForecastingService = st.session_state.service
            available_models = service.list_models(include_experimental=False)
            available_keys = {m["model_key"] for m in available_models}
            filtered_models = [m for m in selected_models if m in available_keys]
            if not filtered_models:
                st.error("No valid models selected. Please select at least one available model.")
                return

            jobs = {}
            run_errors = {}
            for model_key in filtered_models:
                try:
                    with st.spinner(f"Running {model_key}..."):
                        if ui_config.backtest_mode:
                            job = service.run_forecast_with_backtest(
                                model_key=model_key,
                                parsed_data=st.session_state.parsed_data,
                                horizon=horizon,
                                n_windows=ui_config.n_backtest_windows,
                                runtime_config=runtime,
                                business_days_only=ui_config.business_days_only,
                                quantile_levels=DEFAULT_QUANTILE_LEVELS,
                            )
                        else:
                            job = service.run_forecast(
                                model_key=model_key,
                                parsed_data=st.session_state.parsed_data,
                                horizon=horizon,
                                runtime_config=runtime,
                                business_days_only=ui_config.business_days_only,
                                quantile_levels=DEFAULT_QUANTILE_LEVELS,
                            )
                        jobs[model_key] = job
                except Exception as exc:
                    run_errors[model_key] = str(exc)

            st.session_state.forecast_jobs = jobs
            st.session_state.forecast_run_errors = run_errors
            st.session_state.run_metadata = {
                "horizon": horizon,
                "interval_label": ui_config.interval_coverage,
                "business_days_only": ui_config.business_days_only,
                "backtest_mode": ui_config.backtest_mode,
                "autopilot": autopilot_now,
                "interval_pct": int(ui_config.interval_coverage * 100),
            }
            st.session_state.backtest_results = {k: v.backtest for k, v in jobs.items()}
            if run_errors:
                st.warning(
                    "Some selected models could not produce a forecast. "
                    "Their errors are listed in Evaluation."
                )

    jobs = st.session_state.get("forecast_jobs", {})
    if not jobs:
        return

    meta = st.session_state.run_metadata

    # Display results for each model
    for model_key, job in jobs.items():
        result = job.result
        st.markdown(f"#### Model: {model_key}")

        interval_label = f"{int(ui_config.interval_coverage * 100)}%"
        lo_col = f"Lower_{int(ui_config.interval_coverage * 100)}"
        hi_col = f"Upper_{int(ui_config.interval_coverage * 100)}"

        interval_bounds = result.get_interval(ui_config.interval_coverage)
        if interval_bounds is not None:
            lower, upper = interval_bounds
            fc_display = {
                "point": result.point[0] if result.point.ndim == 2 else result.point,
                "q_low": lower[0] if lower.ndim == 2 else lower,
                "q_high": upper[0] if upper.ndim == 2 else upper,
            }
        else:
            fc_display = {
                "point": result.point[0] if result.point.ndim == 2 else result.point,
                "q_low": result.point[0] if result.point.ndim == 2 else result.point,
                "q_high": result.point[0] if result.point.ndim == 2 else result.point,
            }

        future_dates = result.future_index
        actuals = None
        is_bt = bool(meta.get("backtest_mode", False))
        if is_bt and job.actuals is not None:
            actuals = job.actuals
            hist_df = df.iloc[:-horizon] if len(df) > horizon else df
        else:
            hist_df = df

        st.plotly_chart(plot_forecast(hist_df, future_dates, fc_display, interval_label, actuals=actuals), use_container_width=True)

        st.markdown("##### Forecast summary")
        for line in explain_forecast_results(prof["targets"][primary], pd.DataFrame({"Forecast": fc_display["point"]}), meta):
            st.markdown(f"- {line}")

        # Forecast table and download
        head_col, dl_col = st.columns([3, 1])
        head_col.markdown("##### Forecast table")

        display_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": fc_display["point"],
            lo_col: fc_display["q_low"],
            hi_col: fc_display["q_high"],
            "Model": model_key,
            "Run": meta.get("run_timestamp_utc", "N/A"),
        })
        if actuals is not None:
            display_df["Actual"] = actuals
            display_df["AbsError"] = np.abs(display_df["Actual"] - display_df["Forecast"])
            denom = np.where(np.abs(display_df["Actual"]) < 1e-8, np.nan, np.abs(display_df["Actual"]))
            display_df["APE_%"] = np.abs((display_df["Actual"] - display_df["Forecast"]) / denom) * 100
            display_df["InInterval"] = (display_df["Actual"] >= display_df[lo_col]) & (display_df["Actual"] <= display_df[hi_col])

        dl_col.download_button("Download CSV", data=display_df.to_csv(index=False).encode(), file_name=f"{model_key}_forecast.csv", mime="text/csv", use_container_width=True)
        st.dataframe(display_df, use_container_width=True, height=280)

    st.caption(f"Horizon: {meta['horizon']} · Interval: {interval_label} · Models: {', '.join(selected_models)}")


def render_tab_evaluation(ui_config: ForecastConfigUI, selected_models: list[str]) -> None:
    st.markdown("### 4. Evaluation")
    jobs = st.session_state.get("forecast_jobs", {})
    meta = st.session_state.get("run_metadata", {})
    backtest_results = st.session_state.get("backtest_results", {})
    prof = st.session_state.profile_results

    if not jobs or not meta:
        st.info("Run a forecast first in the Forecast tab.")
        return
    if not meta.get("backtest_mode", False):
        st.info("Enable backtesting mode in the sidebar for accuracy metrics.")
        return

    interval_cov = ui_config.interval_coverage

    # Aggregate metrics for models with a completed backtest.
    all_metrics = {}
    comparison_data = []
    for model_key in selected_models:
        bt = backtest_results.get(model_key)
        if bt is None:
            job = jobs.get(model_key)
            error = st.session_state.get("forecast_run_errors", {}).get(model_key)
            if job is not None:
                error = error or job.timing.get("backtest_error")
            comparison_data.append({
                "Model": model_key,
                "Status": "Unavailable",
                "Details": error or "No backtest result was recorded.",
                "MAE": "—",
                "RMSE": "—",
                "MAPE": "—",
                "MASE": "—",
                "Coverage %": "—",
            })
            continue
        agg_pm = bt.aggregate_point_metrics
        agg_prob = bt.aggregate_probabilistic_metrics
        all_metrics[model_key] = {
            "MAE": agg_pm.mae,
            "RMSE": agg_pm.rmse,
            "MAPE": agg_pm.mape,
            "MASE": agg_pm.mase,
            "Coverage": agg_prob.interval_coverage.get(interval_cov, math.nan) if agg_prob else math.nan,
            "backtest": bt,
        }
        comparison_data.append({
            "Model": model_key,
            "Status": "Complete",
            "Details": "",
            "MAE": round(agg_pm.mae, 3),
            "RMSE": round(agg_pm.rmse, 3),
            "MAPE": "N/A" if math.isnan(agg_pm.mape) else f"{agg_pm.mape:.2f}%",
            "MASE": "N/A" if math.isnan(agg_pm.mase) else f"{agg_pm.mase:.3f}",
            "Coverage %": (
                "N/A" if not agg_prob or math.isnan(agg_prob.interval_coverage.get(interval_cov, math.nan))
                else f"{agg_prob.interval_coverage[interval_cov]:.1f}%"
            ),
        })

    st.markdown("#### Model comparison")
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    if not all_metrics:
        st.warning("No selected model has a completed backtest yet.")
        return

    # Per-window chart for each model
    st.markdown("#### Rolling backtest windows")
    for model_key, metrics in all_metrics.items():
        bt = metrics["backtest"]
        if bt.windows:
            plot_windows = []
            for w in bt.windows:
                plot_windows.append({
                    "window": w.window_id,
                    "test_dates": w.test_dates,
                    "actuals": w.test_actuals[0] if w.test_actuals.ndim == 2 else w.test_actuals,
                    "point": w.point_forecasts[0] if w.point_forecasts.ndim == 2 else w.point_forecasts,
                })
            st.markdown(f"##### {model_key}")
            st.plotly_chart(
                plot_rolling_backtest(plot_windows, st.session_state.df),
                use_container_width=True,
                key=f"rolling-backtest-{model_key}",
            )

            # Per-window metrics table
            wdf = pd.DataFrame([{
                "Window": w.window_id,
                "Train ends": str(pd.Timestamp(w.train_end_date).date()),
                "MAE": round(w.point_metrics.mae, 3),
                "RMSE": round(w.point_metrics.rmse, 3),
                "MAPE": round(w.point_metrics.mape, 2) if not math.isnan(w.point_metrics.mape) else "N/A",
                "MASE": round(w.point_metrics.mase, 3) if not math.isnan(w.point_metrics.mase) else "N/A",
                "Coverage %": round(w.probabilistic_metrics.interval_coverage.get(ui_config.interval_coverage, math.nan), 1) if w.probabilistic_metrics else "N/A",
            } for w in bt.windows])
            st.dataframe(wdf, use_container_width=True, hide_index=True)

    # Forecast vs actual + error distribution for primary model
    primary_model = selected_models[0] if selected_models else None
    if primary_model and primary_model in jobs:
        job = jobs[primary_model]
        primary = prof.get("primary_target") if prof else None
        if primary and job.forecast_df is not None:
            fdf = job.forecast_df
            if "Actual" in fdf.columns:
                lo_col = [c for c in fdf.columns if c.startswith("Lower_")][0]
                hi_col = [c for c in fdf.columns if c.startswith("Upper_")][0]
                left, right = st.columns(2)
                with left:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Forecast"], mode="lines", name="Forecast", line=dict(color="#111827", width=2.5)))
                    fig.add_trace(go.Scatter(x=fdf["Date"], y=fdf[hi_col], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=fdf["Date"], y=fdf[lo_col], mode="lines", fill="tonexty", name="Interval", line=dict(width=0), fillcolor="rgba(99,102,241,0.12)"))
                    fig.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Actual"], mode="lines+markers", name="Actual", line=dict(color="#059669", width=2, dash="dot"), marker=dict(size=5)))
                    fig = style_figure(fig, "Forecast vs actual", 380, "Value")
                    fig.update_xaxes(title="Date")
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    st.plotly_chart(plot_error_distribution(fdf), use_container_width=True)


def render_tab_comparison(selected_models: list[str]) -> None:
    st.markdown("### 5. Model Comparison")
    st.caption(
        "Compare forecast accuracy across models, then inspect forecast paths and run details."
    )
    jobs = st.session_state.get("forecast_jobs", {})
    meta = st.session_state.get("run_metadata", {})
    backtest_results = st.session_state.get("backtest_results", {})

    if not jobs:
        st.info("Run forecasts for the selected models to compare them.")
        return

    horizon = meta.get("horizon", 30)
    available_jobs = {model_key: jobs[model_key] for model_key in selected_models if model_key in jobs}
    if not available_jobs:
        st.warning("No forecast data available for the selected models.")
        return
    interval_coverage = meta.get("interval_label", 0.8)
    interval_pct = int(interval_coverage * 100)
    future_dates = next(iter(available_jobs.values())).result.future_index

    metric_values: dict[str, dict[str, float]] = {}
    for model_key in selected_models:
        backtest = backtest_results.get(model_key)
        if backtest is None:
            continue
        point = backtest.aggregate_point_metrics
        probabilistic = backtest.aggregate_probabilistic_metrics
        metric_values[model_key] = {
            "MAE": point.mae,
            "RMSE": point.rmse,
            "MAPE": point.mape,
            "MASE": point.mase,
            "CRPS": probabilistic.crps if probabilistic else math.nan,
            "Pinball": (
                probabilistic.mean_pinball_loss
                if probabilistic and probabilistic.mean_pinball_loss is not None
                else math.nan
            ),
        }

    display_names = {
        model["model_key"]: model["display_name"]
        for model in st.session_state.service.list_models(include_experimental=True)
    }
    available_metrics = [
        name
        for name, key in COMPARISON_METRICS.items()
        if any(
            (value := values.get(key)) is not None and math.isfinite(value)
            for values in metric_values.values()
        )
    ]
    st.markdown("#### Accuracy landscape")
    if len(available_metrics) >= 2:
        axis_x, axis_y = st.columns(2)
        x_metric = axis_x.selectbox(
            "Horizontal axis",
            available_metrics,
            index=available_metrics.index("MAE") if "MAE" in available_metrics else 0,
            key="comparison_scatter_x",
        )
        y_options = [metric for metric in available_metrics if metric != x_metric]
        default_y = "RMSE" if "RMSE" in y_options else y_options[0]
        y_metric = axis_y.selectbox(
            "Vertical axis",
            y_options,
            index=y_options.index(default_y),
            key="comparison_scatter_y",
        )
        scatter = build_metric_scatter(
            metric_values,
            COMPARISON_METRICS[x_metric],
            COMPARISON_METRICS[y_metric],
            display_names,
        )
        if scatter.data:
            st.plotly_chart(
                scatter,
                use_container_width=True,
                key="comparison-accuracy-scatter",
            )
            st.caption(
                "Each point is a completed model backtest. Lower values are better; "
                "models nearer the lower-left perform better on both selected metrics. "
                "Hover over a point for exact values."
            )
        else:
            st.info("No model has both selected metrics available.")
    else:
        st.info("Run completed backtests for at least two accuracy metrics to plot this chart.")

    st.markdown(f"#### Forecast overlay · {interval_pct}% prediction interval")
    fig = go.Figure()

    # Add historical data
    df = st.session_state.df
    hist_col = "y" if "y" in df.columns else df.columns[1]
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df[hist_col], mode="lines", name="Historical",
        line=dict(color="#6366f1", width=2.2)
    ))

    colors = ["#111827", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    for i, (model_key, job) in enumerate(available_jobs.items()):
        result = job.result
        point = result.point[0] if result.point.ndim == 2 else result.point

        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=future_dates, y=point, mode="lines", name=model_key,
            line=dict(color=color, width=2.5)
        ))
        interval_bounds = result.get_interval(interval_coverage)
        if interval_bounds is not None:
            lower, upper = interval_bounds
            lower = lower[0] if lower.ndim == 2 else lower
            upper = upper[0] if upper.ndim == 2 else upper
            fig.add_trace(go.Scatter(
                x=future_dates, y=upper, mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=lower, mode="lines", fill="tonexty",
                name=f"{model_key} interval", line=dict(width=0),
                fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.1)"
            ))

    if len(future_dates) > 0:
        fig.add_vline(x=future_dates[0], line_dash="dot", line_color="#94a3b8")
    fig = style_figure(fig, f"Model comparison (horizon={horizon})", 520, "Value")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Selected model results")
    comparison_rows = []
    run_errors = st.session_state.get("forecast_run_errors", {})
    for model_key in selected_models:
        job = available_jobs.get(model_key)
        if job is None:
            comparison_rows.append({
                "Model": model_key,
                "Forecast": "Failed",
                "Backtest": "Not run",
                "Details": run_errors.get(model_key, "No forecast result was recorded."),
                "MAE": "—",
                "RMSE": "—",
                "Coverage": "—",
            })
            continue
        backtest = backtest_results.get(model_key)
        if backtest is not None:
            point_metrics = backtest.aggregate_point_metrics
            probabilistic_metrics = backtest.aggregate_probabilistic_metrics
            coverage = (
                probabilistic_metrics.interval_coverage.get(interval_coverage, math.nan)
                if probabilistic_metrics
                else math.nan
            )
            mae = f"{point_metrics.mae:.3f}"
            rmse = f"{point_metrics.rmse:.3f}"
            coverage_text = "N/A" if math.isnan(coverage) else f"{coverage:.1f}%"
        else:
            mae = rmse = coverage_text = "—"
        comparison_rows.append({
            "Model": model_key,
            "Forecast": "Complete",
            "Backtest": "Complete" if backtest is not None else "Unavailable",
            "Details": (
                "" if backtest is not None
                else job.timing.get("backtest_error", "No backtest result was recorded.")
            ),
            "MAE": mae,
            "RMSE": rmse,
            "Coverage": coverage_text,
        })
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Side-by-side forecasts")
    for model_key, job in available_jobs.items():
        result = job.result
        point = result.point[0] if result.point.ndim == 2 else result.point

        display_df = pd.DataFrame({"Date": future_dates, "Forecast": point})
        interval_bounds = result.get_interval(interval_coverage)
        if interval_bounds is not None:
            lower, upper = interval_bounds
            display_df[f"Lower_{interval_pct}"] = lower[0] if lower.ndim == 2 else lower
            display_df[f"Upper_{interval_pct}"] = upper[0] if upper.ndim == 2 else upper
        with st.expander(model_key, expanded=False):
            st.dataframe(display_df, use_container_width=True, height=200)


def main() -> None:
    st.set_page_config(page_title="Multi-Model Forecasting Workbench", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    init_session_state()

    col1, col2 = st.columns([5, 1])
    col1.markdown(
        '<div class="hero-title">Time Series Forecasting Workbench</div>'
        '<div class="hero-subtitle">'
        "Explore zero-shot forecasts, compare model accuracy, and inspect backtest results."
        "</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.validation_report:
        col2.markdown("<br>", unsafe_allow_html=True)
        col2.markdown(readiness_badge(st.session_state.validation_report), unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    ui_config, data_source, selected_models = render_sidebar(st.session_state.profile_results)

    section = st.radio(
        "Workflow section",
        ["Data", "Diagnostics", "Forecast", "Evaluation", "Comparison"],
        horizontal=True,
        label_visibility="collapsed",
        key="active_section",
    )
    if section == "Data":
        render_tab_data(data_source, ui_config)
    elif section == "Diagnostics":
        render_tab_diagnostics()
    elif section == "Forecast":
        render_tab_forecast(ui_config, selected_models)
    elif section == "Evaluation":
        render_tab_evaluation(ui_config, selected_models)
    else:
        render_tab_comparison(selected_models)


if __name__ == "__main__":
    main()