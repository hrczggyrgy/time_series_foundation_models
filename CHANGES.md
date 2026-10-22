# Code Review Fixes - October 22, 2026

This document summarizes the critical fixes implemented in response to the code review dated October 22, 2026.

## Multi-Model Comparison Feature - October 22, 2026

This section documents the implementation of the multi-model comparison feature and additional fixes from the second code review.

### P1 - Critical Fixes for Comparison

#### 1. Backtest Context Limit Handling
**Issue**: Backtests could fail on histories longer than a model's context limit, even though the adapter could use a permitted recent context window.

**Fix**:
- Modified `run_backtest()` to truncate training context to model's `max_context_length` in each window
- Uses the most recent `max_context_length` points when training history exceeds limit
- Truncates covariates to match truncated training history
- Context used is reported in backtest results for transparency

**Files**: `forecasting/evaluation/backtesting.py`

#### 2. Observation Mask Accuracy
**Issue**: Observation mask marked resampled points incorrectly - newly created timestamps could appear throughout a resampled series, not just at the end.

**Fix**:
- Changed observation mask to track original timestamps by checking if timestamp existed in original data
- Uses `data["ds"].isin(original_timestamps)` to accurately identify original vs resampled points
- More reliable for model rankings since test labels are more accurate

**Files**: `forecasting/data/parsing.py`

#### 3. TimesFM 2.5 Quantile Fabrication
**Issue**: When quantile output was `None`, adapter repeated point forecast across nine quantile columns, creating degenerate intervals.

**Fix**:
- Removed fallback that fabricated quantiles by repeating point forecast
- Now raises explicit `ValueError` when model does not return quantile forecasts
- Users get clear error message instead of misleading probabilistic output

**Files**: `forecasting/adapters/timesfm25.py`

### P2 - Important Fixes for Comparison

#### 4. Registry Capabilities Alignment
**Issue**: Registry advertised multivariate and covariate support for Timestral variants, while adapter reported those as unsupported. TimesFM 3.0 was registered twice.

**Fix**:
- Updated registry capabilities to match actual adapter implementations
- Timestral variants now marked as `supports_multivariate=False`, `supports_past_covariates=False`, `supports_known_future_covariates=False`
- Removed duplicate TimesFM 3.0 registration, kept single experimental entry
- Added `experimental` field to `ModelRegistryEntry` and `ModelCapabilities`
- T0-Gamma covariate support marked as `False` until context+horizon alignment is implemented

**Files**: `forecasting/registry.py`, `forecasting/contracts.py`

#### 5. Memory Management for Comparison
**Issue**: Loading multiple foundation models for comparison could exhaust memory.

**Fix**:
- Added `_release_adapter()` method to release adapters from cache after use
- `compare_models()` releases each adapter after its forecast completes
- Sequential model loading to avoid concurrent memory usage

**Files**: `forecasting/service.py`

### Multi-Model Comparison Implementation

#### Step 1: Comparison Result Types
**Files**: `forecasting/comparison.py`
- `ModelComparisonItem`: Result for single model (forecast, backtest, error, timing, context used, warnings)
- `ComparisonReport`: Complete comparison with model results, settings, evaluation origins, timestamp

#### Step 2: `ForecastingService.compare_models()`
**Files**: `forecasting/service.py`
- Accepts list of model keys, shared parsed data, horizon, backtest settings
- Generates shared evaluation splits (same for all models)
- Runs each model sequentially with progress callback support
- Records success/failure separately for each model
- Releases adapter memory after each model completes
- Returns `ComparisonReport` with all results

#### Step 3: Fair Comparison Protocol
- Uses same target series, timestamps, forecast horizon for all models
- Uses same training cutoff dates and held-out observations
- Common quantile levels (P10, P50, P90) for 80% interval comparison
- Point-only baselines contribute MAE, RMSE, MASE, bias; interval metrics are N/A

#### Step 4: Tests
**Files**: `tests/unit/test_comparison.py`
- Test comparison report structure
- Test model comparison item structure
- Test validation (requires models, sufficient data)
- Test progress callback functionality

**Results**: 5 new tests pass, total 28 unit tests pass

### Comparison Feature Status

**Implemented**:
- `compare_models()` service method
- Comparison result types
- Sequential model execution with memory management
- Progress callback support
- Shared evaluation splits
- Error handling (one model failure doesn't stop others)
- Unit tests for orchestration

**Not Yet Implemented** (future work):
- UI integration (model multiselect, overlay charts, score table)
- CSV export for comparison reports
- Visual comparison of forecasts
- Per-model warnings panel in UI

### Recommended First Release

**Models**: TimesFM 2.5, T0-Gamma, Naive, Seasonal Naive (exclude TimesFM 3.0 until adapter implemented)

**Input**: One target series, same data preparation across all models

**Comparison**: Point forecasts + common P10/P50/P90 quantiles where supported

**Evaluation**: Common rolling-origin splits, MAE, RMSE, MASE, bias; interval coverage/width/pinball where valid

**Execution**: Sequential model loading, clear progress/error messages

---

## Original Code Review Fixes - October 22, 2026

## Critical Issues Fixed (P1)

### 1. T0-Gamma Quantile Handling (Likely Crash)
**Issue**: Adapter was interpolating quantiles that T0 already returned at requested levels, causing IndexError when requested levels exceeded the 11 values returned.

**Fix**:
- Removed redundant `_interpolate_quantiles` method from T0 adapter
- T0's `model.predict()` returns quantiles at exact requested levels - no interpolation needed
- Corrected `native_quantile_levels` to match official documentation: P01, P05-P95 in 5% increments, P99
- Marked covariate support as `False` until proper context+horizon alignment is implemented

**Files**: `forecasting/adapters/t0.py`

### 2. TimesFM 2.5 Quantile Clamping (Misleading Intervals)
**Issue**: Adapter clamped P05 to P10 and P95 to P90 without warning, displaying 90% intervals that were actually 80% coverage.

**Fix**:
- Added explicit warning when requested levels (e.g., P05/P95) are outside native range (P10-P90)
- Corrected `point_definition` from `"native_mean"` to `"median"` to match TimesFM API
- Removed fake-quantile fallback - now raises `ValueError` if quantile shape is unexpected
- Users now see clear warnings when intervals are narrower than advertised

**Files**: `forecasting/adapters/timesfm25.py`

### 3. Timestral Silent Median Fallback (Misleading Intervals)
**Issue**: Adapter silently substituted median when `forecast.quantile()` failed, creating intervals where medians masqueraded as quantiles.

**Fix**:
- Removed silent median fallback
- Now raises explicit `ValueError` with helpful error message when quantile computation fails
- Marked multivariate and covariate support as `False` in capabilities

**Files**: `forecasting/adapters/timestral.py`

### 4. Backtesting Data Leakage (Invalid Evaluation)
**Issue**: Preprocessing used full series including future observations before backtesting, potentially leaking information into training windows.

**Fix**:
- Added `observation_mask` field to `ParsedData` to track original vs imputed observations
- Modified `prepare_dataframe()` to return observation mask
- Updated `run_backtest()` to accept optional `observation_mask` parameter
- Test actuals are filtered to exclude imputed points during evaluation

**Note**: This is a partial fix. Truly leakage-free backtesting would require re-running preprocessing at each forecast origin. The current implementation at least excludes imputed points from scoring.

**Files**: `forecasting/data/parsing.py`, `forecasting/evaluation/backtesting.py`, `forecasting/service.py`

## Important Issues Fixed (P2)

### 5. RMSE Aggregation (Approximate)
**Issue**: RMSE was calculated as mean of window-level RMSEs, not pooled RMSE across all observations.

**Fix**:
- Added documentation explaining current aggregation is mean of window-level RMSEs
- Documented that proper pooled RMSE would require storing raw errors in `BacktestWindow`
- Added comment noting this is an approximation

**Files**: `forecasting/evaluation/metrics.py`

### 6. Backtest Error Visibility (Opaque Failures)
**Issue**: UI displayed generic "Backtest results not available" message, hiding actual error details.

**Fix**:
- UI now displays backtest error messages from `job.timing["backtest_error"]`
- Added expandable technical details section for debugging
- Users can now see specific reasons for backtest failures

**Files**: `app.py`

## Additional Improvements

### Unit Tests Added
Created comprehensive unit test suite (23 tests, all passing):
- `tests/unit/test_t0_adapter.py` - Quantile levels, covariate support, interpolation removal
- `tests/unit/test_timesfm25_adapter.py` - Native levels, point definition, error handling
- `tests/unit/test_timestral_adapter.py` - Capabilities, error handling, quantile levels
- `tests/unit/test_parsing.py` - Observation mask tracking, resampling detection
- `tests/unit/test_backtesting.py` - Split generation, observation mask filtering
- `tests/unit/test_metrics.py` - Point metrics, aggregation, NaN handling

### Documentation Updates
- Updated README to reflect actual implemented capabilities vs planned features
- Added "Known Limitations" section documenting quantile handling, backtesting, and model availability
- Updated model table to show covariate support as not implemented
- Updated test directory structure in README

### Test Documentation
- Created `tests/README.md` explaining test philosophy and coverage
- Documented what's tested (API contracts) vs what's not (actual model inference)
- Provided guidance for adding new tests

## Verification

All changes compile successfully:
```bash
python -m py_compile forecasting/adapters/t0.py forecasting/adapters/timesfm25.py \
    forecasting/adapters/timestral.py forecasting/data/parsing.py \
    forecasting/evaluation/backtesting.py forecasting/evaluation/metrics.py \
    forecasting/service.py app.py
```

All unit tests pass:
```bash
pytest tests/unit/ -v
# 23 passed in 1.85s
```

## Remaining Work (Recommended)

The code review also mentioned these items for future iterations:

1. **Add smoke tests** against actual model checkpoints to verify runtime compatibility
2. **Implement proper pooled RMSE** by storing raw errors in `BacktestWindow`
3. **Fully implement per-window preprocessing** for truly leakage-free backtesting
4. **Add covariate contract tests** for different feature counts and horizon lengths
5. **Implement joint multivariate forecasting** for Timestral and T0-Gamma
6. **Add CI workflow** to run tests automatically on push

## Impact

These fixes address the most critical reliability issues identified in the code review:

- **No more silent failures**: All adapters now raise explicit errors instead of fabricating results
- **Honest uncertainty intervals**: Users are warned when intervals are narrower than advertised
- **Reduced data leakage**: Backtesting now filters imputed points from evaluation
- **Better debugging**: Error messages are specific and actionable
- **Test coverage**: 23 unit tests verify API contract correctness

The workbench is now much more trustworthy for users, with clear communication of limitations and explicit error handling.
