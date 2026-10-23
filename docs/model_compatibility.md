# Model Compatibility Matrix

**Last Updated:** 2026-10-22  
**Verification Method:** Documentation review, API inspection, license verification  
**Status:** Pre-implementation audit — all capabilities marked `UNVERIFIED` until smoke-tested locally

---

## 1. TimesFM 2.5 (Legacy Baseline)

| Field | Value |
|-------|-------|
| **Model Key** | `timesfm_2_5` |
| **Checkpoint ID** | `google/timesfm-2.5-200m-pytorch` |
| **Checkpoint Revision** | `main` (default) |
| **Runtime Package** | `timesfm` (PyTorch extra) |
| **Runtime Version** | `timesfm>=0.0.7` (current), `timesfm[torch]==2.0.2` for 2.5 |
| **Python Requirement** | `>=3.10` |
| **PyTorch Requirement** | `>=2.0.0` |
| **Supported Devices (Verified)** | CPU, CUDA — MPS: UNVERIFIED |
| **Input/Output Shapes** | Input: `[batch, context]` → Output: point `[batch, horizon]`, quantiles `[batch, horizon, 10]` |
| **Context Limit** | 16,384 (compiled in 32-step increments) |
| **Horizon Behavior** | Up to 512 recommended; compiled in 32-step increments; autoregressive patch decoding |
| **Native Quantile Levels** | 10 deciles: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 (plus mean) |
| **Covariate Support** | Limited future-known covariates only (TimesFM 2.x) |
| **Source-Code License** | Apache-2.0 |
| **Weights License** | Apache-2.0 |
| **Verification Date** | 2026-10-22 |
| **Smoke-Test Status** | **PASSED** (existing app) |

**Notes:**
- Current implementation in `app.py` uses `TimesFM_2p5_200M_torch.from_pretrained()`
- Compilation via `ForecastConfig` with `use_continuous_quantile_head=True`, `force_flip_invariance=True`, `fix_quantile_crossing=True`
- Quantile indices in current app: P10=1, P20=2, P25=3, P50=5, P75=7, P80=8, P90=9 (0-indexed after mean)
- **VERIFIED:** Current app produces working forecasts and backtests

---

## 2. TimesFM 3.0 (Research-Gated)

| Field | Value |
|-------|-------|
| **Model Key** | `timesfm_3` |
| **Checkpoint ID** | `google/timesfm-3.0-pytorch` |
| **Checkpoint Revision** | `main` (default) |
| **Runtime Package** | `timesfm[torch]` |
| **Runtime Version** | `timesfm>=3.0.0` (PyPI release v3.0.0) |
| **Python Requirement** | `>=3.10` |
| **PyTorch Requirement** | `>=2.0.0` (likely `>=2.4` per t0-gamma constraint) |
| **Supported Devices (Verified)** | CPU, CUDA — MPS: UNVERIFIED |
| **Input/Output Shapes** | Input: multivariate `[targets, context]` or batch → Output: point `[targets, horizon]`, quantiles `[targets, horizon, 9]` |
| **Context Limit** | Patch-based (32-step patches); effective limit depends on GPU memory; documented ~32k+ |
| **Horizon Behavior** | Up to 1,024 steps in single forward pass (contiguous patch masking) |
| **Native Quantile Levels** | 9 deciles: 0.1–0.9 (no mean in native output) |
| **Covariate Support** | Native multivariate targets + past-only + known-future covariates |
| **Source-Code License** | Apache-2.0 |
| **Weights License** | **TimesFM Non-Commercial License v1.0** — **RESTRICTED** to non-commercial, non-production use |
| **Verification Date** | 2026-10-22 |
| **Smoke-Test Status** | **UNVERIFIED** |

**Critical Licensing Notice:**
- Self-hosted weights **cannot** be used for commercial or production purposes
- Commercial use only permitted via authorized Google Cloud services (BigQuery ML)
- Application must enforce `research` policy gate before enabling this model
- Do not make this the default for any user

**API Reference (from upstream examples):**
```python
from timesfm3 import TimesFM3Evaluator, ModelConfig

config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=16,
    device="cuda",  # or "cpu"
)
forecaster = TimesFM3Evaluator(config)

# Univariate
target = np.random.randn(128).astype(np.float32)
outputs = list(forecaster.forecast([target], horizon=24))
# outputs[0].point, outputs[0].quantiles (9 levels: 0.1-0.9)

# Multivariate (3 targets)
target = np.random.randn(3, 128).astype(np.float32)
outputs = list(forecaster.forecast([target], horizon=24))
# outputs[0].point.shape == (3, 24), outputs[0].quantiles.shape == (3, 24, 9)
```

---

## 3. Timestral-209M (Main Integration)

| Field | Value |
|-------|-------|
| **Model Key** | `timestral_209m` |
| **Checkpoint ID** | `mistralai/Timestral-209M` |
| **Checkpoint Revision** | `main` (default) |
| **Runtime Package** | `timestral` |
| **Runtime Version** | Latest from Hugging Face / GitHub (public preview Oct 2026) |
| **Python Requirement** | `>=3.10` (assumed) |
| **PyTorch Requirement** | `>=2.0.0` (assumed) |
| **Supported Devices (Verified)** | CPU (10-140ms), CUDA — MPS: UNVERIFIED |
| **Input/Output Shapes** | Input: `[context]` or multivariate → Output: `Forecast` object with `.median`, `.quantile(q)` |
| **Context Limit** | 32,768 steps |
| **Horizon Behavior** | Up to 1,024 steps in single forward pass |
| **Native Quantile Levels** | 21 quantiles (exact levels need verification) |
| **Covariate Support** | Full: multivariate targets + past-only + known-future covariates |
| **Source-Code License** | Apache-2.0 |
| **Weights License** | **Modified MIT** — **Revenue cap**: $20M global consolidated monthly revenue |
| **Verification Date** | 2026-10-22 |
| **Smoke-Test Status** | **UNVERIFIED** |

**Critical Licensing Notice:**
- Weights under Modified MIT with revenue restriction: **not authorized if company/employer monthly revenue > $20M**
- Applies to model, derivatives, modifications, combined works
- Commercial license available from Mistral AI (sales@mistral.ai)
- Users must review and comply with the applicable model and weights terms

**API Reference (from Hugging Face model card):**
```python
import numpy as np
import torch
from timestral import TimestralForecaster

model = TimestralForecaster.from_pretrained(
    "mistralai/Timestral-209M",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

context = np.random.default_rng(42).standard_normal(1000).cumsum()
forecast = model.forecast(context, prediction_length=96)
median = forecast.median
lower, upper = forecast.quantile(0.1), forecast.quantile(0.9)
```

**Quantile Levels:** Need verification — documented as "21 quantiles" but exact probabilities not specified in model card.

---

## 4. Timestral-28M (Lightweight Option)

| Field | Value |
|-------|-------|
| **Model Key** | `timestral_28m` |
| **Checkpoint ID** | `mistralai/Timestral-28M` |
| **Checkpoint Revision** | `main` (default) |
| **Runtime Package** | `timestral` (same as 209M) |
| **Runtime Version** | Same as 209M |
| **Python Requirement** | `>=3.10` (assumed) |
| **PyTorch Requirement** | `>=2.0.0` (assumed) |
| **Supported Devices (Verified)** | CPU, CUDA — MPS: UNVERIFIED |
| **Input/Output Shapes** | Same as 209M |
| **Context Limit** | 32,768 steps (same architecture) |
| **Horizon Behavior** | Up to 1,024 steps |
| **Native Quantile Levels** | 21 quantiles (same as 209M) |
| **Covariate Support** | Full (same as 209M) |
| **Source-Code License** | Apache-2.0 |
| **Weights License** | **Modified MIT** — **Revenue cap**: $20M global consolidated monthly revenue |
| **Verification Date** | 2026-10-22 |
| **Smoke-Test Status** | **UNVERIFIED** |

**Benchmark Comparison (from model card):**
| Benchmark | Metric | Timestral-28M | Timestral-209M |
|-----------|--------|---------------|----------------|
| GIFT-Eval | MASE ↓ | 0.678 | 0.658 |
| GIFT-Eval | CRPS ↓ | 0.461 | 0.446 |
| TIME | MASE ↓ | 0.638 | 0.626 |
| TIME | CRPS ↓ | 0.534 | 0.522 |

---

## 5. t0-gamma (Permissive Multivariate + Covariates)

| Field | Value |
|-------|-------|
| **Model Key** | `t0_gamma` |
| **Checkpoint ID** | `theforecastingcompany/t0-gamma` |
| **Checkpoint Revision** | `main` (default) |
| **Runtime Package** | `tfc-t0[torch]` |
| **Runtime Version** | `>=0.6.0` (**CRITICAL**: older versions load checkpoint but interpret config incorrectly, silently degrading forecasts) |
| **Python Requirement** | `>=3.10` |
| **PyTorch Requirement** | `>=2.4` (documented) |
| **Supported Devices (Verified)** | CPU, CUDA, Apple Silicon (MLX runtime available) — MPS: UNVERIFIED for torch |
| **Input/Output Shapes** | Input: `[batch, features, context]` → Output: `quantiles [batch, horizon, levels]`, `median [batch, horizon]` |
| **Context Limit** | 512 steps (documented in examples) |
| **Horizon Behavior** | Up to 1,024 steps in single forward pass |
| **Native Quantile Levels** | 21 quantile levels (user-requested) |
| **Covariate Support** | Full: past-only + known-future covariates; group_ids for joint forecasting |
| **Source-Code License** | Apache-2.0 |
| **Weights License** | Apache-2.0 |
| **Verification Date** | 2026-10-22 |
| **Smoke-Test Status** | **UNVERIFIED** |

**Critical Version Constraint:**
- **MUST use `tfc-t0[torch]>=0.6.0`** — older releases silently degrade forecasts
- Install: `pip install "tfc-t0[torch]>=0.6.0"`

**API Reference (from Hugging Face model card):**
```python
import torch
from t0 import T0Forecaster

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma").eval()

# Univariate (1 series, 512 steps)
context = torch.randn(1, 512)
out = model.predict(context, horizon=64, quantile_levels=[0.1, 0.5, 0.9])
# out.quantiles.shape == (1, 64, 3), out.median.shape == (1, 64)

# Multivariate (2 series, 3 known-future covariates)
context = torch.randn(2, 512)
future_covariates = torch.randn(2, 3, 512 + 64)  # [batch, features, context+horizon]
out = model.predict(context, horizon=64, quantile_levels=[0.1, 0.5, 0.9], future_covariates=future_covariates)

# Group forecasting (rows sharing group_id forecast jointly)
# Note: group_ids incompatible with future_covariates — use appropriate native representation
```

**Known Incompatibility:** `group_ids` and `future_covariates` cannot be combined; must choose one representation.

---

## 6. Naive Baselines (Zero-Dependency)

| Field | Value |
|-------|-------|
| **Model Keys** | `naive`, `seasonal_naive` |
| **Checkpoint ID** | N/A (algorithmic) |
| **Runtime Package** | None (stdlib + numpy/pandas) |
| **Supported Devices** | Any (pure Python) |
| **Input/Output Shapes** | Input: `[history]` → Output: point `[horizon]`, quantiles: **N/A** (deterministic) |
| **Context Limit** | N/A |
| **Horizon Behavior** | Any |
| **Native Quantile Levels** | None (deterministic point forecasts only) |
| **Covariate Support** | None |
| **Source-Code License** | Apache-2.0 (application code) |
| **Weights License** | N/A |
| **Verification Date** | 2026-10-22 |
| **Smoke-Test Status** | **UNVERIFIED** (to be implemented) |

**Naive:** Last observed value repeated for horizon  
**Seasonal Naive:** Value from same seasonal period repeated (requires frequency inference)

---

## Model License Information

The model license terms are documented in each model's section above and displayed as informational notices in the app. The app does not filter model availability or enforce license eligibility; users are responsible for complying with the applicable terms.

---

## Verification Checklist (Per Model)

Before marking any model `VERIFIED`, the following must pass:

- [ ] Package installs without conflicts in test environment
- [ ] Model loads on CPU without error
- [ ] Univariate forecast produces finite outputs of expected shape
- [ ] Quantile levels are strictly increasing, finite, within (0,1)
- [ ] Quantile ordering validated (no crossing) or violations reported
- [ ] Multivariate forecast tested (if supported) — shape correctness
- [ ] Covariate path tested (if supported) — alignment validation
- [ ] Backtest run on synthetic data — no leakage, metrics computable
- [ ] Device fallback tested (CUDA → CPU)
- [ ] License notice displayed in UI when model selected
- [ ] Cache key includes checkpoint revision, package version, device, dtype, compilation config

---

## Environment Constraints (Reconciled)

| Constraint | Value | Source |
|------------|-------|--------|
| Python | `>=3.10` | t0-gamma, TimesFM 3.0 |
| PyTorch | `>=2.4` | t0-gamma requirement (reconciles with TimesFM `>=2.0`) |
| NumPy | `>=1.24` | Current app |
| Pandas | `>=2.0` | Current app |

**Action Required:** Pin a mutually compatible dependency set in `pyproject.toml` and verify no version conflicts between `timesfm[torch]>=3.0.0`, `tfc-t0[torch]>=0.6.0`, `timestral`, and `torch>=2.4`.

---

## Next Steps

1. Create `pyproject.toml` with pinned dependencies
2. Implement `forecasting/contracts.py` with typed dataclasses
3. Implement `forecasting/registry.py` with model registry metadata
4. Refactor TimesFM 2.5 into `forecasting/adapters/timesfm25.py`
5. Implement naive baselines in `forecasting/adapters/naive.py`
6. Implement t0-gamma adapter (permissive, good first integration)
7. Implement Timestral adapters
8. Implement TimesFM 3.0 adapter
9. Rebuild backtesting around forecast origins
10. Add multivariate/covariate support