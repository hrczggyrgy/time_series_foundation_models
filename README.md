---
title: Multi-Model Time-Series Forecasting Workbench
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 📈 Multi-Model Time-Series Forecasting Workbench

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/hrczggyrgy/TSFMs_forecasting)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/hrczggyrgy/time_series_foundation_models)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)

**Live demo:** [https://huggingface.co/spaces/hrczggyrgy/TSFMs_forecasting](https://huggingface.co/spaces/hrczggyrgy/TSFMs_forecasting)

A modular, extensible zero-shot forecasting workbench supporting **7 foundation models** with unified probabilistic outputs, leakage-free backtesting, and license-aware model selection.

---

## 🎯 Supported Models

| Model | Parameters | License | Policy | Multivariate | Covariates | Context | Horizon | Quantiles |
|-------|------------|---------|--------|--------------|------------|---------|---------|-----------|
| **TimesFM 2.5** | 200M | Apache-2.0 | ✅ Permissive | ❌ | Future-only | 16k | 512 | 9 (0.1–0.9) |
| **TimesFM 3.0** | 330M | Non-commercial v1.0 | 🔬 Research | ✅ | Full | 32k+ | 1024 | 9 (0.1–0.9) |
| **t0-gamma** | 100M+ | Apache-2.0 | ✅ Permissive | ✅ | Full | 512 | 1024 | 21 (0.01–0.99) |
| **Timestral 209M** | 209M | Modified MIT ($20M cap) | 🔬 Research | ✅ | Full | 32k | 1024 | 21 (0.01–0.99) |
| **Timestral 28M** | 28M | Modified MIT ($20M cap) | 🔬 Research | ✅ | Full | 32k | 1024 | 21 (0.01–0.99) |
| **Timestral 123M** | 123M | Modified MIT ($20M cap) | 🔬 Research | ✅ | Full | 32k | 1024 | 21 (0.01–0.99) |
| **Timestral 5M** | 5M | Modified MIT ($20M cap) | 🔬 Research | ✅ | Full | 32k | 1024 | 21 (0.01–0.99) |
| **Naive** | — | — | ✅ Permissive | ❌ | ❌ | Any | Any | None |
| **Seasonal Naive** | — | — | ✅ Permissive | ❌ | ❌ | Any | Any | None |

> **License Policy**: Default `permissive_only` enables only Apache-2.0 models. Switch to `research` mode to access restricted models after acknowledging their license terms.

---

## ✨ Features

### Core Forecasting
- **Zero-shot forecasting** — no training required, works out-of-the-box
- **Unified probabilistic interface** — all models return calibrated quantiles (not just point forecasts)
- **Multi-model comparison** — run identical backtests across models with controlled context/horizon
- **Native multivariate support** — t0-gamma, Timestral, TimesFM 3.0 forecast multiple targets jointly
- **Covariate support** — past-only & known-future covariates where model permits

### Data & Diagnostics
- **CSV upload** with auto column detection (date + multiple targets)
- **Data validation** — irregular gaps, outliers, duplicates, stationarity (ADF), trend analysis
- **Seasonal decomposition** — STL with multiple candidate periods
- **Frequency inference** — robust detection with fallback to delta-mode

### Evaluation & Backtesting
- **Leakage-free rolling-origin backtesting** — expanding/sliding windows, per-window preprocessing
- **Comprehensive metrics**: MAE, RMSE, MAPE, MASE, Bias, Pinball Loss, CRPS, Interval Coverage/Width
- **Per-window & aggregate reporting** — CSV export with forecast origin tracking
- **Calibration diagnostics** — coverage vs. nominal intervals, quantile crossing detection

### Reproducibility & Export
- **Long & wide format CSV exports** — forecasts, quantiles, actuals, errors
- **Metadata JSON** — git commit, data fingerprint, model checkpoint, preprocessing, timing
- **Run ID tracking** — unique identifier for every forecast/backtest job

### UX & Operations
- **License policy selector** — permissive-only vs research mode with acknowledgment flow
- **Model capability badges** — multivariate, covariates, context, horizon, quantiles at a glance
- **Autopilot mode** — recommended horizon/context based on data frequency
- **Docker-ready** — CPU/GPU, persistent model cache volume

---

## 🚀 Quick Start

### Local (Python 3.10+)

```bash
# Clone
git clone https://github.com/hrczggyrgy/time_series_foundation_models.git
cd time_series_foundation_models

# Create venv & install (use uv for speed)
uv venv && source venv/bin/activate
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt

# Run
streamlit run app.py
```

Open `http://localhost:8501`

### Docker (Recommended)

```bash
# Build
docker build -t tsfm-workbench .

# Run (persistent model cache)
docker run -d \
  -p 7860:7860 \
  -v $(pwd)/models:/home/user/.cache/huggingface \
  tsfm-workbench
```

### Hugging Face Spaces

Deploy directly from this repo — the Space auto-builds from the `Dockerfile`.

---

## 📋 Usage Workflow

```
1. DATA ──────────────────► 2. DIAGNOSTICS ──────► 3. FORECAST ───────► 4. EVALUATION
   • Upload CSV / Demo          • Trend (p-value)        • Model select       • MAE / RMSE / MAPE
   • Column mapping             • Stationarity (ADF)     • Horizon / interval • MASE / CRPS
   • Auto-fix gaps/dups         • Seasonal periods       • Permissive/Research • Coverage / Width
   • Validation report          • Outlier count          • Backtest toggle     • Per-window table
```

### Data Format
- **Required**: Date column + at least one numeric target column
- **Optional**: Past covariates, known-future covariates (future columns must extend `horizon` steps beyond last date)
- **Minimum**: 32 observations (model-dependent)

### Model Selection
1. Choose **License Policy** in sidebar:
   - `Permissive only` → TimesFM 2.5, t0-gamma, Naive baselines
   - `Research` → All models (requires acknowledgment for restricted)
2. Select **Model** — capability badges show multivariate, covariates, context, horizon
3. Acknowledge license if prompted (TimesFM 3.0, Timestral)

---

## 🔧 Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face token for higher rate limits / private models |
| `TORCH_THREADS` | 4 | CPU threads for PyTorch |

### Model Cache
Models download to `~/.cache/huggingface/hub/` on first use (~400MB–2GB each). Mount a volume in Docker for persistence:
```bash
-v /host/path/models:/home/user/.cache/huggingface
```

### GPU Support
Install CUDA-enabled PyTorch:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
The app auto-detects CUDA/MPS/CPU.

---

## 📦 Project Structure

```
├── app.py                    # Streamlit UI (orchestrator)
├── pyproject.toml            # Packaging & optional dependencies
├── requirements.txt          # Pinned deployment requirements
├── Dockerfile                # Multi-stage Docker build
├── .streamlit/config.toml    # Streamlit server config
│
├── forecasting/              # Core forecasting library
│   ├── contracts.py          # ForecastRequest, ForecastResult, ModelCapabilities, ForecastAdapter
│   ├── registry.py           # ModelRegistry with license policy enforcement
│   ├── exceptions.py         # Domain exceptions
│   ├── service.py            # ForecastingService (high-level orchestration)
│   │
│   ├── adapters/             # Model adapters (implement ForecastAdapter)
│   │   ├── base.py           # BaseAdapter: quantile interp, crossing repair
│   │   ├── timesfm25.py      # TimesFM 2.5
│   │   ├── timesfm3.py       # TimesFM 3.0 (stub)
│   │   ├── timestral.py      # Timestral 5M/28M/123M/209M
│   │   ├── t0.py             # t0-gamma
│   │   └── naive.py          # Naive / Seasonal Naive
│   │
│   ├── data/
│   │   └── parsing.py        # CSV prep, validation, freq inference, date gen
│   │
│   ├── evaluation/
│   │   ├── metrics.py        # MAE, RMSE, MAPE, MASE, Pinball, CRPS, intervals
│   │   └── backtesting.py    # Rolling-origin, leakage-free splits
│   │
│   └── reporting/
│       └── exports.py        # Long/wide CSV, metadata JSON
│
├── docs/
│   └── model_compatibility.md # Verified model API/license matrix
│
└── tests/                    # Unit/integration/smoke tests (WIP)
```

---

## 📄 License

**Application code**: Apache-2.0 (see `LICENSE`)

**Model weights** — see table above:
- Apache-2.0: TimesFM 2.5, t0-gamma, Naive baselines
- Non-commercial v1.0: TimesFM 3.0 (self-hosted only; commercial via BigQuery ML)
- Modified MIT ($20M revenue cap): Timestral family (commercial license from Mistral AI)

> This application does not grant rights absent from model licenses. The `research` policy is informational only.

---

## 📚 References

- [TimesFM GitHub](https://github.com/google-research/timesfm) / [HF](https://huggingface.co/google/timesfm-2.5-200m-pytorch) / [TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch)
- [Timestral HF](https://huggingface.co/mistralai/Timestral-209M) / [Mistral Blog](https://mistral.ai/news/timestral)
- [t0-gamma HF](https://huggingface.co/theforecastingcompany/t0-gamma) / [TFC Blog](https://www.theforecastingcompany.com/blog/t0-gamma)
- [Model Compatibility Matrix](docs/model_compatibility.md)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## 🤝 Contributing

1. Fork & create feature branch
2. Add model adapter in `forecasting/adapters/` implementing `ForecastAdapter`
3. Register in `forecasting/registry.py` with correct license policy
4. Add entry to `docs/model_compatibility.md`
5. Submit PR with smoke test results

---

## 📝 Citation

If you use this workbench in research, please cite the underlying models:

```bibtex
@article{das2023timesfm,
  title={A decoder-only foundation model for time-series forecasting},
  author={Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen},
  journal={arXiv:2310.10688}, year={2023}
}
@article{meyer2026t0,
  title={t0: A Time-Series Foundation Model for Forecasting with Context},
  author={Meyer, Lucas and Sole, Claudio and others},
  journal={arXiv:2609.24559}, year={2026}
}
@article{mistral2026timestral,
  title={Timestral: Mistral's Time-Series Foundation Models},
  author={Mistral AI}, year={2026}
}
```