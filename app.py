# ============================================================
# TimesFM 2.5 Forecasting Dashboard — improved app.py
# Focus: robustness, autonomy, explainability, UX
# ============================================================

import hashlib
import io
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import timesfm
import torch
from scipy.stats import linregress
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)

MIN_CONTEXT_LENGTH = 32
MODEL_CONTEXT_LIMIT = 16384
DEFAULT_MAX_CONTEXT = 2048
DEFAULT_HORIZON = 30
MAX_UI_HORIZON = 512
TORCH_THREADS = 4
TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-pytorch"

FREQ_LABEL_MAP = {
    "T": "Minutely",
    "MIN": "Minutely",
    "H": "Hourly",
    "D": "Daily",
    "B": "Business Daily",
    "W": "Weekly",
    "M": "Monthly",
    "Q": "Quarterly",
    "A": "Annual",
    "Y": "Annual",
    "Irregular": "Irregular",
    "Unknown": "Unknown",
}

QUANTILE_INDEX_MAP: dict[str, tuple[int, int, int]] = {
    "50 % (P25–P75)": (3, 7, 50),
    "60 % (P20–P80)": (2, 8, 60),
    "80 % (P10–P90)": (1, 9, 80),
}

CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] { font-family: Inter, Segoe UI, Arial, sans-serif; }
[data-testid="stAppViewContainer"] { background: #f6f5f2; }
[data-testid="stSidebar"] { background: #f8f7f4; border-right: 1px solid #e8e6e2; }
[data-testid="metric-container"] { background: #ffffff; border: 1px solid rgba(17,24,39,0.08); border-radius: 14px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.callout { background: #ffffff; border: 1px solid rgba(14,165,233,0.18); border-radius: 12px; padding: 12px 16px; color: #0f172a; margin: 0 0 1rem 0; }
.callout-warn { background: #fffaf2; border: 1px solid rgba(245,158,11,0.22); border-radius: 12px; padding: 12px 16px; color: #7c2d12; margin: 0 0 1rem 0; }
.callout-good { background: #f5fbf7; border: 1px solid rgba(16,185,129,0.22); border-radius: 12px; padding: 12px 16px; color: #065f46; margin: 0 0 1rem 0; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.03em; }
.badge-green { background: #d1fae5; color: #065f46; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-red { background: #fee2e2; color: #991b1b; }
.section-divider { border: none; border-top: 1px solid #e8e6e2; margin: 1.2rem 0 1rem 0; }
.small-note { color: #6b7280; font-size: 0.92rem; }
.step-chip { display: inline-block; padding: 0.32rem 0.7rem; margin-right: 0.45rem; border-radius: 9999px; background: #ece8ff; color: #4338ca; font-size: 0.8rem; font-weight: 700; }
.stButton button, .stDownloadButton button { border-radius: 10px; font-weight: 600; }
.stButton button[kind="primary"] { background: #111827; border-color: #111827; }
</style>
"""


@dataclass(frozen=True)
class ForecastConfigUI:
    horizon: int
    interval_label: str
    business_days_only: bool
    backtest_mode: bool
    n_backtest_windows: int
    use_full_context: bool
    autopilot: bool
    auto_fix_data: bool


def init_session_state() -> None:
    defaults = {
        "raw_signature": None,
        "df": None,
        "date_col": None,
        "target_col": None,
        "validation_report": None,
        "profile_results": None,
        "forecast_results": None,
        "future_dates": None,
        "forecast_df": None,
        "run_metadata": None,
        "backtest_windows": None,
        "compile_signature": None,
        "analysis_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clear_results(keep_signature: bool = True) -> None:
    signature = st.session_state.get("raw_signature") if keep_signature else None
    for key in [
        "df",
        "date_col",
        "target_col",
        "validation_report",
        "profile_results",
        "forecast_results",
        "future_dates",
        "forecast_df",
        "run_metadata",
        "backtest_windows",
        "analysis_ready",
    ]:
        st.session_state[key] = None if key != "analysis_ready" else False
    if keep_signature:
        st.session_state["raw_signature"] = signature


def generate_demo_data() -> pd.DataFrame:
    periods = 540
    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(), periods=periods, freq="D"
    )
    trend = np.linspace(120, 230, periods)
    weekly = 8 * np.sin(np.arange(periods) * 2 * np.pi / 7)
    monthly = 15 * np.sin(np.arange(periods) * 2 * np.pi / 30)
    annualish = 10 * np.sin(np.arange(periods) * 2 * np.pi / 365)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 4.0, periods)
    values = trend + weekly + monthly + annualish + noise
    return pd.DataFrame({"date": dates, "value": np.round(values, 2)})


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
        if any(
            token in name
            for token in ["value", "target", "y", "sales", "price", "demand"]
        ):
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
    mapping = {
        "MS": "M",
        "ME": "M",
        "BMS": "M",
        "BME": "M",
        "QS": "Q",
        "QE": "Q",
        "YS": "A",
        "YE": "A",
        "AS": "A",
        "AE": "A",
    }
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
        (pd.Timedelta(minutes=1), "T"),
        (pd.Timedelta(hours=1), "H"),
        (pd.Timedelta(days=1), "D"),
        (pd.Timedelta(weeks=1), "W"),
        (pd.Timedelta(days=28), "M"),
        (pd.Timedelta(days=90), "Q"),
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
        (pd.Timedelta(minutes=1), "T"),
        (pd.Timedelta(hours=1), "H"),
        (pd.Timedelta(days=1), "D"),
        (pd.Timedelta(weeks=1), "W"),
        (pd.Timedelta(days=30), "M"),
        (pd.Timedelta(days=90), "Q"),
    ]
    return min(mapping, key=lambda x: abs(x[0] - mode_delta))[1]


def seasonal_candidates(freq: str, n_obs: int) -> list[int]:
    freq = canonical_freq(freq)
    if freq in ("T", "MIN"):
        candidates = [60, 1440]
    elif freq == "H":
        candidates = [24, 168, 720]
    elif freq in ("D", "B", "IRREGULAR"):
        candidates = [7, 30, 365]
    elif freq == "W":
        candidates = [52, 26]
    elif freq == "M":
        candidates = [12, 6]
    elif freq == "Q":
        candidates = [4, 8]
    else:
        candidates = [7, 30]
    return [p for p in candidates if n_obs >= 2 * p]


def recommended_horizon(freq: str, n_obs: int) -> tuple[int, str]:
    freq = canonical_freq(freq)
    if freq == "T":
        h, label = 60, "60 minutes"
    elif freq == "H":
        h, label = 48, "48 hours"
    elif freq in ("D", "B"):
        h, label = 30, "30 days"
    elif freq == "W":
        h, label = 26, "26 weeks"
    elif freq == "M":
        h, label = 12, "12 months"
    elif freq == "Q":
        h, label = 8, "8 quarters"
    else:
        h, label = 30, "30 steps"
    h = min(h, max(7, min(MAX_UI_HORIZON, max(7, n_obs // 3))))
    return h, label


def prepare_dataframe(
    raw_df: pd.DataFrame, date_col: str, target_col: str, auto_fix_data: bool
) -> tuple[pd.DataFrame, dict[str, object]]:
    info: dict[str, object] = {
        "dropped_bad_dates": 0,
        "dropped_bad_values": 0,
        "duplicate_rows_aggregated": 0,
        "auto_resampled_to": None,
        "filled_missing_points": 0,
    }
    data = raw_df[[date_col, target_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    info["dropped_bad_dates"] = int(data[date_col].isna().sum())
    info["dropped_bad_values"] = int(data[target_col].isna().sum())
    data = data.dropna(subset=[date_col, target_col]).rename(
        columns={date_col: "ds", target_col: "y"}
    )
    if data.empty:
        raise ValueError("No valid rows after parsing the date and target columns.")
    duplicate_count = int(data["ds"].duplicated().sum())
    if duplicate_count:
        info["duplicate_rows_aggregated"] = duplicate_count
        data = data.groupby("ds", as_index=False)["y"].mean()
    data = data.sort_values("ds").reset_index(drop=True)
    freq_before = infer_frequency_safe(data["ds"])
    if auto_fix_data and freq_before == "Irregular":
        rule = dominant_resample_rule(data["ds"])
        if rule:
            before_len = len(data)
            rs = data.set_index("ds").resample(rule).mean()
            missing_before_fill = int(rs["y"].isna().sum())
            rs["y"] = rs["y"].interpolate(limit_direction="both").ffill().bfill()
            data = rs.reset_index()
            info["auto_resampled_to"] = rule
            info["filled_missing_points"] = (
                max(0, len(data) - before_len) + missing_before_fill
            )
    if data["y"].isna().any():
        missing_before = int(data["y"].isna().sum())
        data["y"] = data["y"].interpolate(limit_direction="both").ffill().bfill()
        info["filled_missing_points"] = (
            int(info["filled_missing_points"]) + missing_before
        )
    if data["y"].isna().any():
        raise ValueError("Unable to fill all missing values automatically.")
    return data, info


def detect_irregular_gaps(ds: pd.Series) -> tuple[int, float]:
    if len(ds) < 3:
        return 0, 0.0
    deltas = pd.Series(pd.DatetimeIndex(ds).sort_values()).diff().dropna()
    if deltas.empty:
        return 0, 0.0
    mode_delta = deltas.mode().iloc[0]
    if mode_delta.total_seconds() == 0:
        return 0, 0.0
    irregular = int((deltas != mode_delta).sum())
    return irregular, irregular / max(1, len(deltas))


def detect_outliers(values: pd.Series, z_thresh: float = 3.5) -> int:
    clean = pd.Series(values).dropna()
    if len(clean) < 10:
        return 0
    median = clean.median()
    mad = np.median(np.abs(clean - median))
    if mad == 0:
        return 0
    modified_z = 0.6745 * (clean - median) / mad
    return int((np.abs(modified_z) > z_thresh).sum())


def validate_timeseries(
    df: pd.DataFrame, prep_info: dict[str, object]
) -> dict[str, object]:
    report: dict[str, object] = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "notes": [],
    }
    if df["ds"].duplicated().any():
        report["errors"].append("Duplicate timestamps remain after preparation.")
    if df["y"].nunique(dropna=True) < 2:
        report["errors"].append(
            "The target series is constant, so forecasting is not meaningful."
        )
    if len(df) < MIN_CONTEXT_LENGTH:
        report["errors"].append(
            f"Need at least {MIN_CONTEXT_LENGTH} observations; found {len(df)}."
        )
    if prep_info.get("dropped_bad_dates", 0):
        report["warnings"].append(
            f"Dropped {prep_info['dropped_bad_dates']} row(s) with invalid dates."
        )
    if prep_info.get("dropped_bad_values", 0):
        report["warnings"].append(
            f"Dropped {prep_info['dropped_bad_values']} row(s) with invalid numeric values."
        )
    if prep_info.get("duplicate_rows_aggregated", 0):
        report["warnings"].append(
            f"Aggregated {prep_info['duplicate_rows_aggregated']} duplicate timestamp row(s) by mean."
        )
    if prep_info.get("auto_resampled_to"):
        report["warnings"].append(
            f"Auto-resampled irregular data to '{prep_info['auto_resampled_to']}' frequency."
        )
    if prep_info.get("filled_missing_points", 0):
        report["warnings"].append(
            f"Filled {prep_info['filled_missing_points']} missing/interpolated point(s)."
        )
    gaps, gap_share = detect_irregular_gaps(df["ds"])
    if gaps > 0:
        report["warnings"].append(
            f"Irregular spacing remains in {gaps} interval(s) ({gap_share:.0%} of gaps)."
        )
    n_out = detect_outliers(df["y"])
    if n_out > 0:
        report["warnings"].append(f"Detected about {n_out} potential outlier(s).")
    if len(df) > MODEL_CONTEXT_LIMIT:
        report["notes"].append(
            f"TimesFM will use at most the most recent {MODEL_CONTEXT_LIMIT:,} points."
        )
    report["valid"] = len(report["errors"]) == 0
    return report


def analyze_trend(values: np.ndarray) -> dict[str, object]:
    x = np.arange(len(values))
    slope, intercept, r_value, p_value, std_err = linregress(x, values)
    direction = "Flat"
    if p_value < 0.05:
        direction = "Increasing" if slope > 0 else "Decreasing"
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "p_value": float(p_value),
        "r_squared": float(r_value**2),
        "std_err": float(std_err),
        "direction": direction,
        "significant": bool(p_value < 0.05),
    }


def test_stationarity(values: np.ndarray) -> dict[str, object]:
    try:
        result = adfuller(values, maxlag=min(12, max(1, len(values) // 4)))
        p_value = float(result[1])
        is_stationary = p_value < 0.05
        interpretation = "Stationary" if is_stationary else "Non-stationary"
        return {
            "adf_statistic": float(result[0]),
            "p_value": p_value,
            "is_stationary": is_stationary,
            "interpretation": interpretation,
        }
    except Exception as exc:
        return {
            "adf_statistic": math.nan,
            "p_value": math.nan,
            "is_stationary": None,
            "interpretation": f"ADF failed: {exc}",
        }


def analyze_seasonality(values: np.ndarray, freq: str) -> dict[str, object]:
    candidates = seasonal_candidates(freq, len(values))
    if not candidates:
        return {
            "seasonal_strength": None,
            "period": None,
            "decomposition": None,
            "additional_periods": [],
            "interpretation": "Not enough data for seasonality analysis.",
            "strength_interpretation": "N/A",
        }
    results = []
    for period in candidates:
        try:
            decomp = seasonal_decompose(values, period=period, model="additive")
            seasonal = np.asarray(decomp.seasonal, dtype=float)
            resid = np.asarray(decomp.resid, dtype=float)
            s_var = float(np.nanvar(seasonal))
            r_var = (
                float(np.nanvar(resid[~np.isnan(resid)]))
                if np.any(~np.isnan(resid))
                else 0.0
            )
            strength = s_var / (s_var + r_var) if (s_var + r_var) > 0 else 0.0
            label = (
                "Strong"
                if strength > 0.6
                else (
                    "Moderate"
                    if strength > 0.3
                    else "Weak" if strength > 0.1 else "Negligible"
                )
            )
            results.append(
                {
                    "period": int(period),
                    "strength": float(strength),
                    "label": label,
                    "decomposition": {
                        "trend": np.asarray(decomp.trend, dtype=float),
                        "seasonal": np.asarray(decomp.seasonal, dtype=float),
                        "resid": np.asarray(decomp.resid, dtype=float),
                    },
                }
            )
        except Exception:
            continue
    if not results:
        return {
            "seasonal_strength": None,
            "period": None,
            "decomposition": None,
            "additional_periods": [],
            "interpretation": "Seasonality analysis failed for all candidate periods.",
            "strength_interpretation": "N/A",
        }
    results.sort(key=lambda r: r["strength"], reverse=True)
    primary = results[0]
    extras = [
        {"period": r["period"], "strength": r["strength"], "label": r["label"]}
        for r in results[1:]
    ]
    return {
        "seasonal_strength": primary["strength"],
        "period": primary["period"],
        "decomposition": primary["decomposition"],
        "additional_periods": extras,
        "interpretation": f"{primary['label']} seasonality detected with period {primary['period']}.",
        "strength_interpretation": primary["label"],
    }


def profile_timeseries(df: pd.DataFrame) -> dict[str, object]:
    values = df["y"].to_numpy(dtype=float)
    freq = infer_frequency_safe(df["ds"])
    gaps, gap_share = detect_irregular_gaps(df["ds"])
    outliers = detect_outliers(df["y"])
    ctx_used = min(len(df), MODEL_CONTEXT_LIMIT)
    return {
        "basic_stats": {
            "n_observations": int(len(df)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
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


@st.cache_resource(show_spinner=False)
def load_model() -> object:
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        return timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            TIMESFM_MODEL_ID, torch_compile=True
        )
    except TypeError:
        return timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_MODEL_ID)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load TimesFM. Check your internet connection or local cache."
        ) from exc


def compile_signature(
    max_context: int, max_horizon: int, infer_is_positive: bool
) -> tuple[int, int, bool]:
    rounded_context = int(
        min(
            MODEL_CONTEXT_LIMIT,
            max(MIN_CONTEXT_LENGTH, math.ceil(max_context / 32) * 32),
        )
    )
    rounded_horizon = int(max(32, math.ceil(max_horizon / 32) * 32))
    return (rounded_context, rounded_horizon, bool(infer_is_positive))


def ensure_model_compiled(
    model: object, max_context: int, max_horizon: int, infer_is_positive: bool
) -> tuple[int, int, bool]:
    signature = compile_signature(max_context, max_horizon, infer_is_positive)
    if st.session_state.get("compile_signature") == signature:
        return signature
    config = timesfm.ForecastConfig(
        max_context=signature[0],
        max_horizon=signature[1],
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=signature[2],
        fix_quantile_crossing=True,
    )
    if hasattr(model, "compile"):
        model.compile(config)
    st.session_state["compile_signature"] = signature
    return signature


def run_forecast(
    model: object,
    values: np.ndarray,
    horizon: int,
    interval_label: str,
    all_non_negative: bool,
    use_full_context: bool,
) -> dict[str, object]:
    if len(values) < MIN_CONTEXT_LENGTH:
        raise ValueError(
            f"Series too short. Need at least {MIN_CONTEXT_LENGTH} observations."
        )
    ctx = min(
        len(values), MODEL_CONTEXT_LIMIT if use_full_context else DEFAULT_MAX_CONTEXT
    )
    signature = ensure_model_compiled(
        model, ctx, max(horizon, 128), bool(all_non_negative)
    )
    input_values = values[-ctx:].astype(np.float32)
    point, quantiles = model.forecast(horizon=horizon, inputs=[input_values])
    point_arr = np.asarray(point[0], dtype=float)
    q_arr = (
        np.asarray(quantiles[0], dtype=float)
        if quantiles is not None
        else np.column_stack([point_arr] * 10)
    )
    q_low_idx, q_high_idx, interval_pct = QUANTILE_INDEX_MAP[interval_label]
    return {
        "point": point_arr,
        "q_low": q_arr[:, q_low_idx],
        "q_high": q_arr[:, q_high_idx],
        "q_all": q_arr,
        "ctx_used": int(ctx),
        "compiled_context": signature[0],
        "compiled_horizon": signature[1],
        "infer_is_positive": bool(all_non_negative),
        "interval_pct": interval_pct,
    }


def build_future_dates(
    last_date: pd.Timestamp, horizon: int, business_days_only: bool, freq_code: str
) -> pd.DatetimeIndex:
    freq_code = canonical_freq(freq_code)
    if business_days_only and freq_code in ("D", "B"):
        return pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)
    if freq_code in ("T", "MIN"):
        offset = pd.offsets.Minute(1)
    elif freq_code == "H":
        offset = pd.offsets.Hour(1)
    elif freq_code == "D":
        offset = pd.offsets.Day(1)
    elif freq_code == "B":
        offset = pd.offsets.BDay(1)
    elif freq_code == "W":
        offset = pd.offsets.Week(weekday=last_date.weekday())
    elif freq_code == "M":
        offset = pd.offsets.MonthEnd(1)
    elif freq_code == "Q":
        offset = pd.offsets.QuarterEnd(1)
    elif freq_code in ("A", "Y"):
        offset = pd.offsets.YearEnd(1)
    else:
        offset = pd.offsets.Day(1)
    return pd.date_range(start=last_date + offset, periods=horizon, freq=offset)


def build_results_df(
    future_dates: pd.DatetimeIndex,
    fc: dict[str, object],
    interval_label: str,
    actuals: np.ndarray | None,
    metadata: dict[str, object],
) -> pd.DataFrame:
    tag = interval_label.split("(")[0].strip().replace(" ", "").replace("%", "")
    lo_col = f"Lower_{tag}"
    hi_col = f"Upper_{tag}"
    df = pd.DataFrame(
        {
            "Date": pd.DatetimeIndex(future_dates),
            "Forecast": np.asarray(fc["point"], dtype=float),
            lo_col: np.asarray(fc["q_low"], dtype=float),
            hi_col: np.asarray(fc["q_high"], dtype=float),
            "ModelID": metadata["model_id"],
            "RunTimestampUTC": metadata["run_timestamp_utc"],
            "Horizon": metadata["horizon"],
            "Interval": metadata["interval_label"],
            "ContextUsed": metadata["ctx_used"],
            "BusinessDaysOnly": metadata["business_days_only"],
            "BacktestMode": metadata["backtest_mode"],
            "Autopilot": metadata["autopilot"],
        }
    )
    if actuals is not None:
        df["Actual"] = np.asarray(actuals, dtype=float)
        df["AbsError"] = np.abs(df["Actual"] - df["Forecast"])
        denom = np.where(np.abs(df["Actual"]) < 1e-8, np.nan, np.abs(df["Actual"]))
        df["APE_%"] = np.abs((df["Actual"] - df["Forecast"]) / denom) * 100
        df["InInterval"] = (df["Actual"] >= df[lo_col]) & (df["Actual"] <= df[hi_col])
    return df


def backtest_metrics(fdf: pd.DataFrame) -> dict[str, float]:
    err = fdf["Actual"] - fdf["Forecast"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape_series = (
        fdf.get("APE_%", pd.Series(dtype=float))
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    mape = float(mape_series.mean()) if not mape_series.empty else math.nan
    coverage = (
        float(fdf["InInterval"].mean() * 100)
        if "InInterval" in fdf.columns
        else math.nan
    )
    bias = float(np.mean(err))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Coverage": coverage, "Bias": bias}


def compute_crps(actuals: np.ndarray, q_all: np.ndarray) -> float:
    actuals = np.asarray(actuals, dtype=float)
    q_all = np.asarray(q_all, dtype=float)
    if q_all.ndim != 2 or q_all.shape[0] != len(actuals) or q_all.shape[1] < 10:
        return math.nan
    quantile_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q_preds = q_all[:, 1:10]
    pinball_losses = []
    for i, q in enumerate(quantile_levels):
        diff = actuals - q_preds[:, i]
        loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
        pinball_losses.append(float(np.mean(loss)))
    return float(2 * np.mean(pinball_losses))


def rolling_backtest(
    df: pd.DataFrame,
    horizon: int,
    n_windows: int,
    model: object,
    interval_label: str,
    all_non_negative: bool,
    use_full_context: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    n = len(df)
    need = MIN_CONTEXT_LENGTH + horizon
    if n < need:
        return results

    n_windows = min(n_windows, max(1, n - need + 1))
    windows_meta: list[dict[str, object]] = []
    step = max(1, horizon // max(1, n_windows))

    for w in range(n_windows):
        train_end = n - horizon - w * step
        if train_end < MIN_CONTEXT_LENGTH:
            break
        test_end = train_end + horizon
        train_vals = df["y"].iloc[:train_end].to_numpy(np.float32)
        test_vals = df["y"].iloc[train_end:test_end].to_numpy(np.float32)
        test_dates = pd.DatetimeIndex(df["ds"].iloc[train_end:test_end])
        windows_meta.append(
            {
                "window": w + 1,
                "train_end": df["ds"].iloc[train_end - 1],
                "test_dates": test_dates,
                "train_vals": train_vals,
                "test_vals": test_vals,
            }
        )

    if not windows_meta:
        return results

    max_ctx_cap = MODEL_CONTEXT_LIMIT if use_full_context else DEFAULT_MAX_CONTEXT
    ctx = min(
        max_ctx_cap,
        max(MIN_CONTEXT_LENGTH, min(len(w["train_vals"]) for w in windows_meta)),
    )

    ensure_model_compiled(model, ctx, max(horizon, 128), all_non_negative)

    batch_inputs = [w["train_vals"][-ctx:] for w in windows_meta]
    points, quantiles = model.forecast(horizon=horizon, inputs=batch_inputs)

    q_low_idx, q_high_idx, _ = QUANTILE_INDEX_MAP[interval_label]

    for meta_window, point_w, q_all_w in zip(windows_meta, points, quantiles):
        point_arr = np.asarray(point_w, dtype=float)
        q_all_arr = np.asarray(q_all_w, dtype=float)
        if q_all_arr.ndim == 1:
            q_all_arr = np.column_stack([point_arr] * 10)
        q_low = q_all_arr[:, q_low_idx]
        q_high = q_all_arr[:, q_high_idx]
        err = meta_window["test_vals"] - point_arr
        results.append(
            {
                "window": meta_window["window"],
                "train_end": meta_window["train_end"],
                "test_dates": meta_window["test_dates"],
                "actuals": meta_window["test_vals"],
                "point": point_arr,
                "q_low": q_low,
                "q_high": q_high,
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err**2))),
                "CRPS": compute_crps(meta_window["test_vals"], q_all_arr),
                "Coverage": float(
                    np.mean(
                        (meta_window["test_vals"] >= q_low)
                        & (meta_window["test_vals"] <= q_high)
                    )
                    * 100
                ),
            }
        )

    return results


def readiness_badge(report: dict[str, object]) -> str:
    errors = len(report.get("errors", []))
    warnings_n = len(report.get("warnings", []))
    if errors:
        return f'<span class="badge badge-red"> {errors} error(s)</span>'
    if warnings_n:
        return f'<span class="badge badge-yellow">️ {warnings_n} warning(s)</span>'
    return '<span class="badge badge-green"> Ready</span>'


def explain_series(profile: dict[str, object], report: dict[str, object]) -> list[str]:
    stats = profile["basic_stats"]
    trend = profile["trend"]
    seas = profile["seasonality"]
    notes = []
    notes.append(
        f"The series contains {stats['n_observations']:,} observations with inferred frequency {stats['freq_label']}."
    )
    notes.append(
        f"Trend is {trend['direction'].lower()} and the strongest detected seasonal period is {seas['period'] or 'not available'}."
    )
    if stats["outlier_count"]:
        notes.append(
            f"About {stats['outlier_count']} potential outlier(s) were detected, so sharp jumps may widen uncertainty bands."
        )
    if report.get("warnings"):
        notes.append(
            "Warnings do not block forecasting, but they reduce trust and should be reviewed before production use."
        )
    return notes


def explain_forecast_results(
    profile: dict[str, object], forecast_df: pd.DataFrame, meta: dict[str, object]
) -> list[str]:
    stats = profile["basic_stats"]
    first_val = float(forecast_df["Forecast"].iloc[0])
    last_val = float(forecast_df["Forecast"].iloc[-1])
    direction = (
        "upward"
        if last_val > first_val
        else "downward" if last_val < first_val else "flat"
    )
    lo_col = [c for c in forecast_df.columns if c.startswith("Lower_")][0]
    hi_col = [c for c in forecast_df.columns if c.startswith("Upper_")][0]
    avg_band = float((forecast_df[hi_col] - forecast_df[lo_col]).mean())
    text = [
        f"The forecast points {direction} over the selected horizon of {meta['horizon']} step(s).",
        f"Average uncertainty band width is {avg_band:.2f}, so larger numbers indicate less certainty.",
        f"The model used {meta['ctx_used']:,} recent data point(s) out of the {MODEL_CONTEXT_LIMIT:,}-point maximum context.",
    ]
    if stats["all_non_negative"]:
        text.append(
            "Because the historical series is non-negative, the model was configured to keep future values non-negative as well."
        )
    return text


def explain_backtest_results(
    metrics: dict[str, float], meta: dict[str, object]
) -> list[str]:
    text = [
        f"Backtest MAE is {metrics['MAE']:.3f} and RMSE is {metrics['RMSE']:.3f}; lower values are better.",
        f"Interval coverage is {metrics['Coverage']:.1f}% against a nominal {meta['interval_pct']}% interval.",
    ]
    if not math.isnan(metrics["MAPE"]):
        text.append(
            f"MAPE is {metrics['MAPE']:.2f}%, which is easier to interpret across scales."
        )
    if metrics["Coverage"] < max(35, meta["interval_pct"] - 20):
        text.append(
            "Coverage is meaningfully below the nominal interval, so the forecast intervals may be too narrow."
        )
    elif metrics["Coverage"] > min(99, meta["interval_pct"] + 15):
        text.append(
            "Coverage is far above the nominal interval, so the forecast intervals may be conservative."
        )
    return text


def style_figure(
    fig: go.Figure, title: str, height: int, yaxis_title: str = "Value"
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        margin=dict(l=12, r=12, t=46, b=12),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.08)", zeroline=False)
    fig.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor="rgba(15,23,42,0.08)",
        zeroline=False,
    )
    return fig


def plot_forecast(
    hist_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    fc: dict[str, object],
    interval_label: str,
    actuals: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_df["ds"],
            y=hist_df["y"],
            mode="lines",
            name="Historical",
            line=dict(color="#6366f1", width=2.2),
        )
    )
    if actuals is not None:
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=actuals,
                mode="lines+markers",
                name="Actual",
                line=dict(color="#059669", width=2, dash="dot"),
                marker=dict(size=5),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=fc["q_high"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=fc["q_low"],
            mode="lines",
            fill="tonexty",
            name=f"Interval ({interval_label.split('(')[0].strip()})",
            line=dict(width=0),
            fillcolor="rgba(99,102,241,0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=fc["point"],
            mode="lines",
            name="Forecast",
            line=dict(color="#111827", width=3),
        )
    )
    if len(future_dates) > 0:
        fig.add_vline(x=future_dates[0], line_dash="dot", line_color="#94a3b8")
    fig = style_figure(fig, "Forecast", 520, "Value")
    fig.update_xaxes(title="Date")
    return fig


def plot_decomposition(
    dates: pd.DatetimeIndex, decomp: dict[str, np.ndarray]
) -> go.Figure:
    fig = go.Figure()
    for (name, vals), color in zip(
        [
            ("Trend", decomp["trend"]),
            ("Seasonal", decomp["seasonal"]),
            ("Residual", decomp["resid"]),
        ],
        ["#4f46e5", "#f59e0b", "#10b981"],
    ):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=vals,
                mode="lines",
                name=name,
                line=dict(color=color, width=1.7),
            )
        )
    return style_figure(fig, "Seasonal decomposition", 340, "Component value")


def plot_context_window(df: pd.DataFrame, ctx_used: int) -> go.Figure:
    fig = go.Figure()
    if len(df) > ctx_used:
        fig.add_trace(
            go.Scatter(
                x=df["ds"].iloc[:-ctx_used],
                y=df["y"].iloc[:-ctx_used],
                mode="lines",
                name="Outside context",
                line=dict(color="#d1d5db", width=1.2),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df["ds"].iloc[-ctx_used:],
            y=df["y"].iloc[-ctx_used:],
            mode="lines",
            name=f"Context window ({ctx_used:,} pts)",
            line=dict(color="#4f46e5", width=2),
        )
    )
    return style_figure(fig, "Context used by TimesFM", 280, "Value")


def plot_error_distribution(fdf: pd.DataFrame) -> go.Figure:
    errors = fdf["Actual"] - fdf["Forecast"]
    fig = go.Figure(
        go.Histogram(
            x=errors,
            nbinsx=24,
            marker_color="#4f46e5",
            marker_line_color="#312e81",
            marker_line_width=0.6,
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#dc2626")
    fig = style_figure(fig, "Error distribution", 320, "Count")
    fig.update_xaxes(title="Error (actual - forecast)")
    return fig


def plot_rolling_backtest(
    windows: list[dict[str, object]], hist_df: pd.DataFrame
) -> go.Figure:
    fig = go.Figure()
    ctx = hist_df.tail(min(180, len(hist_df)))
    fig.add_trace(
        go.Scatter(
            x=ctx["ds"],
            y=ctx["y"],
            mode="lines",
            name="Historical",
            line=dict(color="#6366f1", width=1.5),
        )
    )
    colors = ["#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
    for w in windows:
        c = colors[(w["window"] - 1) % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=w["test_dates"],
                y=w["actuals"],
                mode="lines",
                name=f"W{w['window']} Actual",
                line=dict(color=c, width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=w["test_dates"],
                y=w["point"],
                mode="lines",
                name=f"W{w['window']} Forecast",
                line=dict(color=c, width=2),
            )
        )
    return style_figure(fig, "Rolling backtest", 420, "Value")


def render_sidebar(
    profile_results: dict[str, object] | None,
) -> tuple[ForecastConfigUI, str]:
    st.sidebar.markdown("## ️ Settings")
    data_source = st.sidebar.radio(
        "Choose data source",
        ["Synthetic demo data", "Upload CSV"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    suggested = DEFAULT_HORIZON
    suggestion_label = "30 steps"
    if profile_results is not None:
        suggested, suggestion_label = recommended_horizon(
            profile_results["basic_stats"]["inferred_frequency"],
            profile_results["basic_stats"]["n_observations"],
        )
    autopilot = st.sidebar.toggle(
        "Use recommended settings",
        value=True,
        help="Applies the recommended horizon and safer defaults after analysis.",
    )
    horizon = st.sidebar.slider(
        "Horizon (steps ahead)",
        7,
        MAX_UI_HORIZON,
        suggested if autopilot else DEFAULT_HORIZON,
        help=f"Recommended for current data: {suggestion_label}. Longer horizons increase uncertainty.",
        disabled=autopilot,
    )
    interval_label = st.sidebar.selectbox(
        "Prediction interval",
        list(QUANTILE_INDEX_MAP.keys()),
        index=2,
        help="Wider intervals are more conservative.",
    )
    business_days_only = False
    backtest_mode = False
    n_backtest_windows = 3
    use_full_context = False
    auto_fix_data = True
    with st.sidebar.expander("Advanced options", expanded=False):
        auto_fix_data = st.checkbox(
            "Auto-fix duplicates, gaps, and irregular spacing when possible", value=True
        )
        business_days_only = st.checkbox(
            "Business days only",
            value=False,
            help="Skip weekends when generating future dates for daily data.",
        )
        backtest_mode = st.checkbox(
            "Backtesting mode",
            value=True,
            help="Uses the last horizon as holdout data and reports accuracy metrics.",
        )
        n_backtest_windows = st.slider(
            "Rolling windows",
            1,
            5,
            3,
            help="More windows give a more robust estimate but take longer.",
        )
        use_full_context = st.checkbox(
            "Use full 16k context",
            value=False,
            help="Useful for very long series. Slower on CPU.",
        )
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.sidebar.download_button(
        "Download sample CSV",
        data=sample_csv_bytes(),
        file_name="sample_timeseries.csv",
        mime="text/csv",
        use_container_width=True,
        help="A ready-to-run two-column example with enough rows for diagnostics and backtesting.",
    )
    st.sidebar.caption("""Model: `google/timesfm-2.5-200m-pytorch`

Zero-shot · 16k context · probabilistic quantile head""")
    return (
        ForecastConfigUI(
            horizon=horizon,
            interval_label=interval_label,
            business_days_only=business_days_only,
            backtest_mode=backtest_mode,
            n_backtest_windows=n_backtest_windows,
            use_full_context=use_full_context,
            autopilot=autopilot,
            auto_fix_data=auto_fix_data,
        ),
        data_source,
    )


def render_tab_data(data_source: str, ui_config: ForecastConfigUI) -> None:
    st.markdown("### 1. Data")
    if data_source == "Synthetic demo data":
        raw_df = generate_demo_data()
        signature = "demo-data-v2"
        st.markdown(
            '<div class="callout">Using built-in demo data with trend, seasonality, and noise. Switch to <b>Upload CSV</b> to use your own series.</div>',
            unsafe_allow_html=True,
        )
    else:
        uploaded = st.file_uploader(
            "Upload CSV with a date column and one numeric target",
            type=["csv"],
            help="CSV should contain at least 32 rows for forecasting.",
        )
        if uploaded is None:
            st.markdown(
                '<div class="callout">Upload a CSV or use the sample file from the sidebar.</div>',
                unsafe_allow_html=True,
            )
            return
        raw_df = load_csv(uploaded.getvalue())
        signature = hashlib.md5(uploaded.getvalue()).hexdigest()
    if signature != st.session_state.raw_signature:
        clear_results(keep_signature=False)
        st.session_state.raw_signature = signature
    cols = list(raw_df.columns)
    date_idx, target_idx = infer_column_candidates(raw_df)
    c1, c2 = st.columns(2)
    date_col = c1.selectbox(
        "Date column", cols, index=date_idx, key=f"date_col_{signature}"
    )
    target_col = c2.selectbox(
        "Target column", cols, index=target_idx, key=f"target_col_{signature}"
    )
    with st.expander("Preview raw data", expanded=True):
        st.dataframe(raw_df.head(25), use_container_width=True, height=240)
        st.caption(f"Rows: {len(raw_df):,} · Columns: {len(raw_df.columns)}")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    btn1, btn2 = st.columns([1, 1])
    prepare_clicked = btn1.button(
        "Prepare data", type="primary", use_container_width=True
    )
    auto_prepare_clicked = btn2.button(
        "Prepare with recommended settings", use_container_width=True
    )
    if prepare_clicked or auto_prepare_clicked:
        with st.spinner("Preparing, validating, and profiling the series…"):
            try:
                prepared, prep_info = prepare_dataframe(
                    raw_df, date_col, target_col, ui_config.auto_fix_data
                )
                val = validate_timeseries(prepared, prep_info)
                prof = profile_timeseries(prepared)
            except Exception as exc:
                st.error(str(exc))
                return
        st.session_state.df = prepared
        st.session_state.date_col = date_col
        st.session_state.target_col = target_col
        st.session_state.validation_report = val
        st.session_state.profile_results = prof
        st.session_state.forecast_results = None
        st.session_state.future_dates = None
        st.session_state.forecast_df = None
        st.session_state.run_metadata = None
        st.session_state.backtest_windows = None
        st.session_state.analysis_ready = True
    if (
        st.session_state.validation_report is None
        or st.session_state.profile_results is None
    ):
        return
    val = st.session_state.validation_report
    prof = st.session_state.profile_results
    stats = prof["basic_stats"]
    st.markdown(f"**Data readiness:** {readiness_badge(val)}", unsafe_allow_html=True)
    for msg in val.get("errors", []):
        st.error(msg)
    for msg in val.get("warnings", []):
        st.warning(msg)
    for msg in val.get("notes", []):
        st.info(msg)
    if val["valid"] and not val.get("warnings"):
        st.success("All checks passed.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{stats['n_observations']:,}")
    c2.metric("Frequency", stats["freq_label"])
    c3.metric("Outliers ~", str(stats["outlier_count"]))
    c4.metric("Context used", f"{stats['context_used']:,}")
    c5.metric(
        "Recommended horizon",
        str(
            recommended_horizon(stats["inferred_frequency"], stats["n_observations"])[0]
        ),
    )
    st.markdown("#### What this means")
    for note in explain_series(prof, val):
        st.markdown(f"- {note}")


def render_tab_diagnostics() -> None:
    st.markdown("### 2. Diagnostics")
    if st.session_state.profile_results is None:
        st.info("Prepare data first in the Data tab.")
        return
    prof = st.session_state.profile_results
    stats = prof["basic_stats"]
    trend = prof["trend"]
    stationarity = prof["stationarity"]
    seasonality = prof["seasonality"]
    df = st.session_state.df
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Global linear trend", trend["direction"])
    c2.metric("Trend R²", f"{trend['r_squared']:.3f}")
    c3.metric("Stationary", "Yes" if stationarity.get("is_stationary") else "No")
    c4.metric("Seasonal period", str(seasonality["period"] or "N/A"))
    c5.metric("Seasonal strength", seasonality.get("strength_interpretation", "N/A"))
    st.caption(
        f"Date range: {stats['date_range']} · Inferred frequency: {stats['freq_label']} · Irregular gaps: {stats['irregular_gap_count']}"
    )
    st.markdown(
        f"**Global linear trend:** {trend['direction']} (slope {trend['slope']:.4f}, p-value {trend['p_value']:.4f}, R² {trend['r_squared']:.3f})"
    )
    st.markdown(f"**Stationarity:** {stationarity['interpretation']}")
    st.markdown(f"**Seasonality:** {seasonality['interpretation']}")
    if seasonality.get("additional_periods"):
        extras = ", ".join(
            [f"{x['period']} ({x['label']})" for x in seasonality["additional_periods"]]
        )
        if extras:
            st.markdown(f"**Additional candidate periods:** {extras}")
    if stats["all_non_negative"]:
        st.markdown(
            '<div class="callout-good">All historical values are non-negative, so the model will keep the forecast non-negative as well.</div>',
            unsafe_allow_html=True,
        )
    if stats["context_utilization"] < 0.05:
        st.markdown(
            '<div class="callout-warn">This series is short relative to the model maximum context. Results may still be useful, but uncertainty should be interpreted conservatively.</div>',
            unsafe_allow_html=True,
        )
    ctx = stats["context_used"]
    if st.session_state.run_metadata is not None:
        ctx = st.session_state.run_metadata.get("ctx_used", ctx)
    st.plotly_chart(plot_context_window(df, int(ctx)), use_container_width=True)
    if seasonality.get("decomposition") is not None:
        st.plotly_chart(
            plot_decomposition(df["ds"], seasonality["decomposition"]),
            use_container_width=True,
        )
    with st.expander("Show full historical series"):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["ds"],
                y=df["y"],
                mode="lines",
                line=dict(color="#4f46e5", width=1.6),
                name="Series",
            )
        )
        fig = style_figure(fig, "Historical series", 300, "Value")
        fig.update_xaxes(title="Date")
        st.plotly_chart(fig, use_container_width=True)


def render_tab_forecast(ui_config: ForecastConfigUI) -> None:
    st.markdown("### 3. Forecast")
    df = st.session_state.df
    val = st.session_state.validation_report
    prof = st.session_state.profile_results
    if df is None or val is None or prof is None:
        st.info("Prepare data first in the Data tab.")
        return
    if not val["valid"]:
        st.error("Fix the data issues before forecasting.")
        return
    stats = prof["basic_stats"]
    inferred_freq = stats["inferred_frequency"]
    rec_horizon, rec_label = recommended_horizon(inferred_freq, len(df))
    all_non_negative = bool(stats["all_non_negative"])
    effective_horizon = rec_horizon if ui_config.autopilot else ui_config.horizon
    st.markdown(
        f'<div class="callout">Recommended horizon for <b>{stats["freq_label"]}</b> data is <b>{rec_label}</b>. Autopilot is <b>{"on" if ui_config.autopilot else "off"}</b>, so the effective horizon is <b>{effective_horizon}</b> step(s).</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([2, 1])
    run_manual = left.button(
        "▶️ Run forecast", type="primary", use_container_width=True
    )
    run_autopilot = right.button(
        "Run with recommended settings", use_container_width=True
    )
    should_run = run_manual or run_autopilot
    autopilot_now = ui_config.autopilot or run_autopilot
    if should_run:
        horizon = rec_horizon if autopilot_now else ui_config.horizon
        with st.spinner("Loading TimesFM and generating the forecast…"):
            try:
                model = load_model()
                train_df = df.copy()
                actuals = None
                if ui_config.backtest_mode:
                    need = horizon + MIN_CONTEXT_LENGTH
                    if len(df) < need:
                        st.error(
                            f"Need at least {need} observations for backtesting with horizon {horizon}."
                        )
                        return
                    train_df = df.iloc[:-horizon].copy()
                    actuals = df.iloc[-horizon:]["y"].to_numpy(np.float32)
                    future_dates = pd.DatetimeIndex(df.iloc[-horizon:]["ds"].to_numpy())
                else:
                    future_dates = build_future_dates(
                        df["ds"].iloc[-1],
                        horizon,
                        ui_config.business_days_only,
                        inferred_freq,
                    )
                fc = run_forecast(
                    model,
                    train_df["y"].to_numpy(np.float32),
                    horizon,
                    ui_config.interval_label,
                    all_non_negative,
                    ui_config.use_full_context or autopilot_now,
                )
                meta = {
                    "model_id": TIMESFM_MODEL_ID,
                    "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                    "horizon": horizon,
                    "interval_label": ui_config.interval_label,
                    "ctx_used": fc["ctx_used"],
                    "business_days_only": ui_config.business_days_only,
                    "backtest_mode": ui_config.backtest_mode,
                    "autopilot": bool(autopilot_now),
                    "interval_pct": fc["interval_pct"],
                }
                fdf = build_results_df(
                    future_dates, fc, ui_config.interval_label, actuals, meta
                )
                st.session_state.forecast_results = fc
                st.session_state.future_dates = future_dates
                st.session_state.forecast_df = fdf
                st.session_state.run_metadata = meta
                if ui_config.backtest_mode:
                    with st.spinner(
                        f"Running {ui_config.n_backtest_windows}-window rolling backtest…"
                    ):
                        st.session_state.backtest_windows = rolling_backtest(
                            df,
                            horizon,
                            ui_config.n_backtest_windows,
                            model,
                            ui_config.interval_label,
                            all_non_negative,
                            ui_config.use_full_context or autopilot_now,
                        )
                else:
                    st.session_state.backtest_windows = None
            except Exception as exc:
                st.exception(exc)
                return
    if (
        st.session_state.forecast_results is None
        or st.session_state.forecast_df is None
        or st.session_state.run_metadata is None
    ):
        return
    fc = st.session_state.forecast_results
    fdf = st.session_state.forecast_df
    meta = st.session_state.run_metadata
    future_dates = st.session_state.future_dates
    is_bt = bool(meta["backtest_mode"])
    actuals = (
        fdf["Actual"].to_numpy(float) if is_bt and "Actual" in fdf.columns else None
    )
    hist_df = df if not is_bt else df.iloc[: -meta["horizon"]]
    st.plotly_chart(
        plot_forecast(
            hist_df, future_dates, fc, meta["interval_label"], actuals=actuals
        ),
        use_container_width=True,
    )
    st.markdown("#### Forecast summary")
    for line in explain_forecast_results(prof, fdf, meta):
        st.markdown(f"- {line}")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    head_col, dl_col = st.columns([3, 1])
    head_col.markdown("#### Forecast table")
    dl_col.download_button(
        "Download CSV",
        data=fdf.to_csv(index=False).encode(),
        file_name="timesfm_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.dataframe(fdf, use_container_width=True, height=280)
    st.caption(
        f"Run: {meta['run_timestamp_utc']} UTC · Context used: {meta['ctx_used']:,} pts · Horizon: {meta['horizon']} · Interval: {meta['interval_label']} · Autopilot: {meta['autopilot']}"
    )


def render_tab_evaluation() -> None:
    st.markdown("### 4. Evaluation")
    fdf = st.session_state.forecast_df
    meta = st.session_state.run_metadata
    windows = st.session_state.backtest_windows
    fc = st.session_state.forecast_results
    prof = st.session_state.profile_results
    if fdf is None or meta is None or fc is None:
        st.info("Run a forecast first in the Forecast tab.")
        return
    if not meta["backtest_mode"]:
        st.info("Enable backtesting mode in the sidebar for accuracy metrics.")
        return
    if "Actual" not in fdf.columns:
        st.warning("Actual values are not available for evaluation.")
        return
    metrics = backtest_metrics(fdf)
    crps = compute_crps(
        fdf["Actual"].to_numpy(float), np.asarray(fc["q_all"], dtype=float)
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "MAE", f"{metrics['MAE']:.3f}", help="Average absolute error. Lower is better."
    )
    c2.metric(
        "RMSE",
        f"{metrics['RMSE']:.3f}",
        help="Penalizes large errors more strongly than MAE.",
    )
    c3.metric(
        "MAPE",
        "N/A" if math.isnan(metrics["MAPE"]) else f"{metrics['MAPE']:.2f}%",
        help="Scale-free percentage error.",
    )
    c4.metric(
        "CRPS",
        "N/A" if math.isnan(crps) else f"{crps:.4f}",
        help="Probabilistic accuracy metric. Lower is better.",
    )
    c5.metric(
        "Coverage",
        f"{metrics['Coverage']:.1f}%",
        help="Share of actual values inside the selected interval.",
    )
    st.markdown("#### Evaluation summary")
    for line in explain_backtest_results({**metrics, "CRPS": crps}, meta):
        st.markdown(f"- {line}")
    if windows:
        st.plotly_chart(
            plot_rolling_backtest(windows, st.session_state.df),
            use_container_width=True,
        )
        wdf = pd.DataFrame(
            [
                {
                    "Window": w["window"],
                    "Train ends": str(pd.Timestamp(w["train_end"]).date()),
                    "MAE": round(float(w["MAE"]), 3),
                    "RMSE": round(float(w["RMSE"]), 3),
                    "CRPS": (
                        round(float(w["CRPS"]), 4)
                        if not math.isnan(float(w["CRPS"]))
                        else np.nan
                    ),
                    "Coverage %": round(float(w["Coverage"]), 1),
                }
                for w in windows
            ]
        )
        st.dataframe(wdf, use_container_width=True, hide_index=True)
    lo_col = [c for c in fdf.columns if c.startswith("Lower_")][0]
    hi_col = [c for c in fdf.columns if c.startswith("Upper_")][0]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fdf["Date"],
                y=fdf[hi_col],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fdf["Date"],
                y=fdf[lo_col],
                mode="lines",
                fill="tonexty",
                name="Interval",
                line=dict(width=0),
                fillcolor="rgba(99,102,241,0.12)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fdf["Date"],
                y=fdf["Forecast"],
                mode="lines",
                name="Forecast",
                line=dict(color="#111827", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fdf["Date"],
                y=fdf["Actual"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="#059669", width=2, dash="dot"),
                marker=dict(size=5),
            )
        )
        fig = style_figure(fig, "Forecast vs actual", 380, "Value")
        fig.update_xaxes(title="Date")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.plotly_chart(plot_error_distribution(fdf), use_container_width=True)
    if prof is not None and metrics["Coverage"] < meta["interval_pct"] - 20:
        st.markdown(
            '<div class="callout-warn">Backtest intervals appear too narrow for this dataset. Consider keeping autopilot on, using a wider interval, or cleaning outliers before production use.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="callout-good">Backtest completed successfully. Review MAE/RMSE for point accuracy and Coverage/CRPS for probabilistic reliability.</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="TimesFM 2.5 Forecasting Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_session_state()
    col1, col2 = st.columns([5, 1])
    col1.markdown("# TimesFM 2.5 Forecasting Dashboard")
    col1.markdown(
        "A streamlined workflow for preparing data, reviewing diagnostics, running forecasts, and checking backtest quality."
    )
    if st.session_state.validation_report:
        col2.markdown("<br>", unsafe_allow_html=True)
        col2.markdown(
            readiness_badge(st.session_state.validation_report), unsafe_allow_html=True
        )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    ui_config, data_source = render_sidebar(st.session_state.profile_results)
    st.markdown(
        '<div class="small-note"><span class="step-chip">1 Data</span><span class="step-chip">2 Diagnostics</span><span class="step-chip">3 Forecast</span><span class="step-chip">4 Evaluation</span></div>',
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Diagnostics", "Forecast", "Evaluation"])
    with tab1:
        render_tab_data(data_source, ui_config)
    with tab2:
        render_tab_diagnostics()
    with tab3:
        render_tab_forecast(ui_config)
    with tab4:
        render_tab_evaluation()


if __name__ == "__main__":
    main()
