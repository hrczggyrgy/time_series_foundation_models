# Test Suite

This directory contains tests for the Time-Series Foundation Models Workbench.

## Test Structure

- `unit/`: Unit tests for individual components (adapters, parsing, metrics, backtesting)
- `integration/`: Integration tests for component interactions
- `smoke/`: Smoke tests against actual model checkpoints
- `fixtures/`: Test data and mock objects

## Running Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_t0_adapter.py

# Run with coverage
pytest tests/unit/ --cov=forecasting

# Run with verbose output
pytest tests/unit/ -v
```

## Test Coverage

### Unit Tests

- **test_t0_adapter.py**: Tests T0-Gamma adapter quantile handling and covariate support
- **test_timesfm25_adapter.py**: Tests TimesFM 2.5 adapter quantile clamping and error handling
- **test_timestral_adapter.py**: Tests Timestral adapter error handling and capabilities
- **test_parsing.py**: Tests data parsing and observation mask tracking
- **test_backtesting.py**: Tests backtest split generation and observation mask filtering
- **test_metrics.py**: Tests point metrics computation and aggregation

### What's Tested

These unit tests verify the critical fixes from the code review:

1. **Quantile handling correctness**: Ensures adapters don't interpolate when models already return requested levels
2. **Error handling**: Verifies adapters raise explicit errors instead of silent fallbacks
3. **Capability flags**: Confirms adapters accurately report their implemented features
4. **Observation mask tracking**: Tests that parsing returns observation masks to prevent data leakage
5. **Metrics aggregation**: Verifies RMSE aggregation is documented as approximate

### What's Not Yet Tested

- **Actual model inference**: Smoke tests against real model checkpoints (would require downloading weights)
- **End-to-end workflows**: Full forecasting pipeline with real data
- **Covariate tensor shapes**: Contract tests for different feature counts and horizon lengths
- **Per-window preprocessing**: Truly leakage-free backtesting (requires re-running preprocessing at each origin)

## Adding New Tests

When adding new models or features:

1. Add quantile contract tests to verify the model returns quantiles at requested levels
2. Add capability flag tests to ensure implemented features are accurately reported
3. Add error handling tests to verify explicit errors instead of silent fallbacks
4. For data preprocessing, add tests that verify observation mask tracking

## Test Philosophy

These tests focus on **API contract correctness** rather than model performance. They verify that:

- Adapters follow their model's documented API
- Quantile levels are handled correctly
- Error messages are explicit and actionable
- Capabilities metadata matches implementation
- Data leakage is prevented through observation tracking

For model performance evaluation, use the smoke tests with actual model checkpoints.
