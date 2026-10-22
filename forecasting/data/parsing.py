"""Data parsing and validation utilities."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParsedData:
    """Result of parsing and preparing a CSV."""

    df: pd.DataFrame  # Columns: ds (datetime), y (float) — plus any covariates
    date_column: str
    target_columns: list[str]
    past_covariate_columns: list[str]
    known_covariate_columns: list[str]
    prep_report: dict[str, Any]
    raw_signature: str


@dataclass(frozen=True)
class ValidationReport:
    """Validation results for prepared data."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    notes: list[str]


def infer_column_candidates(df: pd.DataFrame) -> tuple[int, int]:
    """Guess date and target column indices."""
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
            for token in ["value", "target", "y", "sales", "price", "demand", "volume"]
        ):
            target_idx = i
            break

    if len(cols) >= 2 and date_idx == target_idx:
        for i in range(len(cols)):
            if i != date_idx:
                target_idx = i
                break

    return date_idx, target_idx


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Load CSV from bytes."""
    return pd.read_csv(io.BytesIO(file_bytes))


def compute_data_signature(file_bytes: bytes) -> str:
    """Compute stable hash for cache invalidation."""
    return hashlib.md5(file_bytes).hexdigest()


def canonical_freq(freq: str) -> str:
    """Normalize pandas frequency string to canonical form."""
    if not freq:
        return "Unknown"
    freq = freq.upper().split("-")[0]
    mapping = {
        "MS": "M", "ME": "M", "BMS": "M", "BME": "M",
        "QS": "Q", "QE": "Q",
        "YS": "A", "YE": "A", "AS": "A", "AE": "A",
    }
    return mapping.get(freq, freq)


def infer_frequency_safe(series: pd.Series) -> str:
    """Robust frequency inference with fallback to delta mode."""
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
    """Find best resampling rule for irregular data."""
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
    """Candidate seasonal periods for a given frequency."""
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


def prepare_dataframe(
    raw_df: pd.DataFrame,
    date_col: str,
    target_cols: list[str],
    past_covariate_cols: list[str] | None = None,
    known_covariate_cols: list[str] | None = None,
    auto_fix: bool = True,
    aggregation: str = "mean",  # "mean", "sum", "first", "last"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare and clean dataframe. Returns prepared df and prep report."""

    past_covariate_cols = past_covariate_cols or []
    known_covariate_cols = known_covariate_cols or []

    all_cols = [date_col] + target_cols + past_covariate_cols + known_covariate_cols
    data = raw_df[all_cols].copy()

    prep_info: dict[str, Any] = {
        "dropped_bad_dates": 0,
        "dropped_bad_values": 0,
        "duplicate_rows_aggregated": 0,
        "auto_resampled_to": None,
        "filled_missing_points": 0,
        "target_columns": target_cols,
        "past_covariate_columns": past_covariate_cols,
        "known_covariate_columns": known_covariate_cols,
    }

    # Parse dates
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    prep_info["dropped_bad_dates"] = int(data[date_col].isna().sum())

    # Parse numeric columns
    numeric_cols = target_cols + past_covariate_cols + known_covariate_cols
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        prep_info["dropped_bad_values"] += int(data[col].isna().sum())

    # Drop rows with any NaN in key columns
    data = data.dropna(subset=[date_col] + target_cols).rename(columns={date_col: "ds"})

    # Normalize first target column to 'y' for internal consistency
    if target_cols:
        primary_target = target_cols[0]
        if primary_target != "y":
            data = data.rename(columns={primary_target: "y"})
            # Update target_cols list for downstream use
            target_cols = ["y"] + target_cols[1:]

    if data.empty:
        raise ValueError("No valid rows after parsing date and target columns.")

    # Handle duplicate timestamps
    duplicate_count = int(data["ds"].duplicated().sum())
    if duplicate_count:
        prep_info["duplicate_rows_aggregated"] = duplicate_count
        agg_func = {col: aggregation for col in numeric_cols}
        data = data.groupby("ds", as_index=False).agg(agg_func)

    data = data.sort_values("ds").reset_index(drop=True)

    # Handle irregular frequency
    freq_before = infer_frequency_safe(data["ds"])
    # Store raw frequency for date generation (preserves MS/ME, QS/QE, etc.)
    raw_freq = pd.infer_freq(pd.DatetimeIndex(data["ds"]))
    if auto_fix and freq_before == "Irregular":
        rule = dominant_resample_rule(data["ds"])
        if rule:
            before_len = len(data)
            # Set index for resampling
            indexed = data.set_index("ds")
            # Resample all numeric columns
            rs = indexed.resample(rule).mean()  # Use mean for resampling
            missing_before_fill = int(rs[target_cols].isna().sum().sum())
            # Interpolate missing
            for col in numeric_cols:
                rs[col] = rs[col].interpolate(limit_direction="both").ffill().bfill()
            data = rs.reset_index()
            prep_info["auto_resampled_to"] = rule
            prep_info["filled_missing_points"] = max(0, len(data) - before_len) + missing_before_fill
            # Update raw_freq after resampling
            raw_freq = rule

    # Fill any remaining missing values in target columns
    for col in target_cols:
        if data[col].isna().any():
            missing_before = int(data[col].isna().sum())
            data[col] = data[col].interpolate(limit_direction="both").ffill().bfill()
            prep_info["filled_missing_points"] = int(prep_info["filled_missing_points"]) + missing_before

    # Final check
    for col in target_cols:
        if data[col].isna().any():
            raise ValueError(f"Unable to fill all missing values in target column '{col}'.")

    prep_info["raw_frequency"] = raw_freq
    return data, prep_info


def validate_timeseries(
    df: pd.DataFrame,
    target_cols: list[str],
    prep_info: dict[str, Any],
    min_observations: int = 32,
) -> ValidationReport:
    """Validate prepared time series data.

    Note: target_cols contains original column names. After prepare_dataframe,
    the first target column is normalized to 'y'. We map original names to actual
    dataframe column names.
    """

    # Map original target column names to actual dataframe column names
    # First target column is normalized to 'y'
    actual_target_cols = []
    for i, col in enumerate(target_cols):
        if i == 0 and "y" in df.columns:
            actual_target_cols.append("y")
        elif col in df.columns:
            actual_target_cols.append(col)
        else:
            actual_target_cols.append(col)  # fallback

    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    for orig_col, actual_col in zip(target_cols, actual_target_cols):
        if df["ds"].duplicated().any():
            errors.append(f"Duplicate timestamps remain after preparation (target: {orig_col}).")

        if actual_col in df.columns and df[actual_col].nunique(dropna=True) < 2:
            warnings.append(
                f"Target column '{orig_col}' is constant — forecasting may not be meaningful."
            )

        if len(df) < min_observations:
            errors.append(
                f"Need at least {min_observations} observations for '{orig_col}'; found {len(df)}."
            )

    if prep_info.get("dropped_bad_dates", 0):
        warnings.append(f"Dropped {prep_info['dropped_bad_dates']} row(s) with invalid dates.")

    if prep_info.get("dropped_bad_values", 0):
        warnings.append(
            f"Dropped {prep_info['dropped_bad_values']} row(s) with invalid numeric values."
        )

    if prep_info.get("duplicate_rows_aggregated", 0):
        warnings.append(
            f"Aggregated {prep_info['duplicate_rows_aggregated']} duplicate timestamp row(s) "
            f"by {prep_info.get('aggregation', 'mean')}."
        )

    if prep_info.get("auto_resampled_to"):
        warnings.append(
            f"Auto-resampled irregular data to '{prep_info['auto_resampled_to']}' frequency."
        )

    if prep_info.get("filled_missing_points", 0):
        warnings.append(
            f"Filled {prep_info['filled_missing_points']} missing/interpolated point(s)."
        )

    # Check for irregular gaps
    gaps, gap_share = detect_irregular_gaps(df["ds"])
    if gaps > 0:
        warnings.append(
            f"Irregular spacing remains in {gaps} interval(s) ({gap_share:.0%} of gaps)."
        )

    # Outlier detection per target
    for orig_col, actual_col in zip(target_cols, actual_target_cols):
        if actual_col in df.columns:
            n_out = detect_outliers(df[actual_col])
            if n_out > 0:
                warnings.append(f"Target '{orig_col}': detected about {n_out} potential outlier(s).")

    if len(df) > 16384:
        notes.append(
            f"Some models will use at most the most recent 16,384 points."
        )

    return ValidationReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        notes=notes,
    )


def detect_irregular_gaps(ds: pd.Series) -> tuple[int, float]:
    """Count irregular gaps in datetime series."""
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
    """Detect outliers using modified Z-score (MAD-based)."""
    clean = pd.Series(values).dropna()
    if len(clean) < 10:
        return 0
    median = clean.median()
    mad = np.median(np.abs(clean - median))
    if mad == 0:
        return 0
    modified_z = 0.6745 * (clean - median) / mad
    return int((np.abs(modified_z) > z_thresh).sum())


def build_future_dates(
    last_date: pd.Timestamp,
    horizon: int,
    business_days_only: bool,
    freq_code: str,
    raw_freq: str | None = None,
) -> pd.DatetimeIndex:
    """Generate future timestamps based on inferred frequency.
    
    If raw_freq is provided (e.g., 'MS', 'ME', 'QS', 'QE', 'YS', 'YE'),
    it preserves the original frequency semantics (start vs end).
    """
    # Use raw_freq if available to preserve start/end semantics
    if raw_freq:
        freq_code = raw_freq
    else:
        freq_code = canonical_freq(freq_code)

    if business_days_only and freq_code in ("D", "B"):
        return pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)

    offset_map = {
        "T": pd.offsets.Minute(1),
        "MIN": pd.offsets.Minute(1),
        "H": pd.offsets.Hour(1),
        "D": pd.offsets.Day(1),
        "B": pd.offsets.BDay(1),
        "W": pd.offsets.Week(weekday=last_date.weekday()),
        # Month start/end
        "MS": pd.offsets.MonthBegin(1),
        "ME": pd.offsets.MonthEnd(1),
        # Quarter start/end
        "QS": pd.offsets.QuarterBegin(1),
        "QE": pd.offsets.QuarterEnd(1),
        # Year start/end
        "YS": pd.offsets.YearBegin(1),
        "YE": pd.offsets.YearEnd(1),
        "AS": pd.offsets.YearBegin(1),
        "AE": pd.offsets.YearEnd(1),
        # Legacy normalized codes (fallback to end)
        "M": pd.offsets.MonthEnd(1),
        "Q": pd.offsets.QuarterEnd(1),
        "A": pd.offsets.YearEnd(1),
        "Y": pd.offsets.YearEnd(1),
    }

    offset = offset_map.get(freq_code, pd.offsets.Day(1))
    return pd.date_range(start=last_date + offset, periods=horizon, freq=offset)