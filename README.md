---
title: TimesFM 2.5 Forecasting Dashboard
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 📈 TimesFM 2.5 Forecasting Dashboard

Zero-shot univariate time-series forecasting powered by [Google TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (200M parameters).

## Features

- **Zero-shot forecasting** — no training required
- **Multi-seasonal decomposition** — detects trend, seasonal cycles, and residuals
- **16k context window** — accepts up to 16,384 historical points
- **Probabilistic intervals** — 50%, 80%, and 60% prediction intervals (Q-learning)
- **Rolling-origin backtesting** — robust accuracy estimates across multiple windows
- **Real-time diagnostics** — validates data quality, detects outliers, and infers frequency
- **CSV export** — download forecasts with confidence bounds

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

### 3. Upload data or use demo

- **Synthetic demo**: 365-day daily series with trend, weekly, and monthly seasonality
- **Upload CSV**: Two-column format (date, value). Minimum 32 observations required

## System Requirements

- **Python**: 3.10+
- **RAM**: 8GB recommended (GPU optional but significantly faster)
- **Storage**: ~2GB for model download (cached)

## Workflow

1. **Data tab** — load CSV or use synthetic demo, auto-validate
2. **Diagnostics tab** — inspect trend, stationarity, seasonality, outliers
3. **Forecast tab** — generate zero-shot forecast with configurable horizon
4. **Evaluation tab** — review backtest metrics (MAE, RMSE, CRPS, coverage)

## Model Details

- **Model**: `google/timesfm-2.5-200m-pytorch`
- **Frequency tokens**: Automatically inferred (high=0, medium=1, low=2)
- **Context**: Up to 16,384 points; default 2,048
- **Horizon**: Up to 512 steps; recommended 30–48 for best accuracy
- **Quantiles**: 10 levels (mean, Q10, Q20, …, Q90) for calibration

## Performance Tips

- Use **at least 32–64 observations** for best results
- Enable **backtesting mode** to estimate accuracy on your data
- Increase **horizon** cautiously — longer forecasts have higher uncertainty
- Check **Diagnostics** for data issues (gaps, outliers, stationarity)

## License

Apache 2.0 (TimesFM model from Google Research)

## References

- [TimesFM GitHub](https://github.com/google-research/timesfm)
- [HuggingFace Model Card](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [Streamlit Docs](https://docs.streamlit.io/)
