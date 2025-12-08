# 24-Hour Ahead Photovoltaic (PV) Power Forecasting

[![CI](https://github.com/claudio-dragotta/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/workflows/CI/badge.svg)](https://github.com/claudio-dragotta/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Introduction

Accurate forecasting of photovoltaic (PV) power generation is crucial for efficient grid management, energy storage optimization, and renewable energy integration. This project implements a deep learning-based approach for 24-hour ahead PV power forecasting using historical production data and meteorological observations.

The system currently provides a hybrid CNN-BiLSTM baseline and is being upgraded to a competition‑ready pipeline based on a modern Temporal Fusion Transformer (TFT) model and a strong LightGBM baseline with an ensemble on top. The upgrade adds physics‑informed features (pvlib: solar position, clear‑sky, POA), robust anti‑leakage handling, walk‑forward validation, and optional probabilistic outputs.

### Key Features (Current)

- **Multi-step forecasting**: Predicts power output for the next 24 hours in a single forward pass
- **Robust time handling**: Comprehensive management of timezone conversions and daylight saving time transitions
- **Hybrid architecture**: Combines CNN feature extraction with BiLSTM temporal modeling
- **Feature engineering**: Incorporates cyclical time features, lag variables, and rolling statistics
- **Rigorous evaluation**: Uses MASE and RMSE metrics with naive baseline comparison

### Key Features (Planned Upgrade)

- **State‑of‑the‑art model**: Temporal Fusion Transformer (via Darts/PyTorch Forecasting)
- **Strong baseline**: LightGBM multi‑orizzonte (24 ore) con feature ingegnerizzate
- **Ensemble**: Combinazione pesata TFT + LightGBM ottimizzata su validation
- **Physics‑informed features**: posizione solare, clear‑sky (Ineichen), POA, clearness index (CSI)
- **Anti‑leakage day‑ahead**: uso di sole informazioni disponibili a t e (se presenti) covariate meteo future
- **Validazione robusta**: walk‑forward multi‑fold, metriche per orizzonte e skill vs baseline fisiche

### Dataset Overview

The project utilizes two years (2010-2012) of data from a solar installation in Australia:

- **PV Production Data**: 17,542 hourly measurements with millisecond precision
- **Weather Data**: 14 meteorological variables including temperature, humidity, wind, and solar irradiance (GHI, DNI, DHI)
- **Location**: Australia (Sydney region, UTC+10:00 timezone)
- **Installed Capacity**: 82.41 kWp

### Professional Development Setup

This project follows professional software engineering best practices:

- **Comprehensive Testing**: Full test suite with pytest (40+ tests covering features, metrics, data processing, and pipeline integration)
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing, linting, and building across Python 3.9-3.12
- **Code Quality**: Automated formatting (Black), import sorting (isort), and type checking (mypy)
- **Pre-commit Hooks**: Automatic code quality checks before every commit
- **Modern Packaging**: Full `pyproject.toml` setup with proper dependencies and metadata
- **Structured Logging**: Professional logging framework for debugging and monitoring
- **Versioning**: Semantic versioning with `__version__` attribute

### Experimental Setup: 3-Model Ensemble with Solar Weighting

This project implements a **state-of-the-art ensemble system** with physics-informed sample weighting:

**Unified Feature Set**:
- Lags: 1h, 24h, 48h, 72h, 96h, 168h (multi-day memory)
- Rolling mean & variance: 3h, 6h, 12h, 24h
- Solar position, clear-sky estimates, clearness index, and calendar flags (weekend/holiday)

**Three Model Architectures:**

- **LightGBM**: Fast tree-based gradient boosting (24 independent horizon models)
- **CNN-BiLSTM**: Deep learning with Conv1D + bidirectional LSTM (sequence-to-sequence)
- **TFT** (Temporal Fusion Transformer): State-of-the-art attention-based forecasting

**Solar-Weighted Training:**

All three models use **sample weighting based on solar zenith angle** to prioritize daylight hours:
- Weight formula: `cos(zenith)^gamma + min_weight`, normalized to mean=1.0 (default gamma=1.5, min_weight=0.1)
- Effect: Night hours (weight ≈ 0.1) vs peak sun (weight boosted by gamma)
- Goal: Improve MASE by focusing on accurate daytime predictions

**Ensemble Strategy:**

1. Train all 3 models with solar weighting
2. Optimize ensemble weights using exhaustive search (tests all combinations)
3. Automatically select best combination (2 or 3 models) with optimal weights

**Key Benefits:**

- Physics-informed weighting improves daytime accuracy (where PV matters most)
- Diversifies predictions across architectures (tree-based + deep learning)
- Leverages strengths: LightGBM excels at short horizons (h=1-12), CNN/TFT at long horizons (h=13-24)

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed comparison plan and [METRICS_ANALYSIS.md](METRICS_ANALYSIS.md) for performance benchmarks.

---

## Baseline Model Results

**Training Configuration:**
- Data: 17,350 training samples with solar-weighted sample weighting
- Features: 189 features (45 base + 144 future meteo features for h=1-24)
- Validation/Test split: 20% each, chronological (no shuffle)
- Metrics: RMSE and MAE on test set

### Completed Baselines

| Model | Architecture | Epochs | Best Val Loss | Test RMSE | Test MAE | Output Directory | Status |
|-------|--------------|--------|---------------|-----------|----------|------------------|--------|
| **TFT Baseline** | Temporal Fusion Transformer | 29 (early stop) | 3.549 | **3.7060** | 2.3254 | `outputs/tft/baseline/` | ✅ Completed |
| **CNN-BiLSTM Baseline** | 3-layer CNN (64→128→256) + BiLSTM(128) | 49 (early stop) | 18.8721 | 3.7267 | 2.3294 | `outputs/cnn/baseline/` | ✅ Completed |
| **CNN-BiLSTM Fusion** | 3-branch fusion + attention | 55 (early stop) | - | 4.364 | - | `outputs/cnn/fusion_attention/` | ⚠️ Overfitted |
| **LightGBM** | 24 separate models (one per horizon) | - | - | - | - | `outputs/lgbm/baseline/` | 🚧 Pending |

### Key Observations

1. **TFT vs CNN Performance**: Both baseline models achieve nearly identical test RMSE (~3.71), demonstrating that well-tuned deep learning architectures can match attention-based models.

2. **Optimization Impact**: 
   - TFT: Reduced from 613K→176K parameters with optimized hyperparameters (hidden=32, heads=2, dropout=0.4, lr=1e-4)
   - CNN: Increased from 240K→597K parameters with deeper architecture, mixed precision training, and aggressive learning rate scheduling (lr=1e-3 with ReduceLROnPlateau)

3. **Training Efficiency**:
   - TFT: ~8 min/epoch, 29 epochs = 3.8 hours total
   - CNN: First epoch 3-5 min (XLA compilation), subsequent epochs 3-4 min each, 49 epochs = ~2.5 hours total

4. **Regularization Strategy**: Both models use solar-weighted sample training to prioritize daytime predictions where PV generation matters most.

5. **Output Organization**: All model outputs are organized in `outputs/{model_type}/{variant}/` structure for easy comparison and ensemble building.

### Hyperparameter Optimization Experiments

We tested alternative TFT configurations to find optimal model capacity:

| Configuration | Hidden Size | Attention Heads | Dropout | Parameters | Test RMSE | Test MASE | Result |
|---------------|-------------|-----------------|---------|------------|-----------|-----------|--------|
| **Baseline (Best)** | 32 | 2 | 0.4 | 176K | **3.7060** | **0.4254** | ✅ Optimal |
| Larger Capacity | 64 | 4 | 0.25 | 613K | 5.0741 | 0.7766 | ❌ Overfitting |

**Key Finding**: Despite 3.5× more parameters, the larger model overfits and performs **37% worse** (RMSE 5.07 vs 3.71). The baseline configuration with **stronger regularization** (dropout=0.4) and **smaller capacity** (hidden=32) generalizes much better to test data.

**Lesson**: For time series forecasting with limited data (~17K samples), aggressive regularization and smaller models often outperform larger architectures.

**Next Steps**: 
1. **Hyperparameter Grid Search**: Use Ray Tune to test all 243 combinations of TFT hyperparameters (3 concurrent trials, ~20 hours with ASHA early stopping)
2. **LightGBM Baseline**: Complete multi-horizon gradient boosting baseline
3. **Ensemble System**: Build and optimize weighted ensemble from TFT + CNN + LightGBM

---

## Table of Contents

1. [Introduction](#introduction)
   - [Key Features](#key-features)
   - [Dataset Overview](#dataset-overview)
   - [Baseline Model Results](#baseline-model-results)
2. [Project Structure](#project-structure)
3. [Data Description](#data-description)
4. [Dataset Merging Strategy](#dataset-merging-strategy)
   - [Why Merge the Datasets?](#why-merge-the-datasets)
   - [Challenges Addressed](#challenges-addressed)
   - [Merged Dataset Characteristics](#merged-dataset-characteristics)
5. [Time Handling and DST Management](#time-handling-and-dst-management)
6. [Forecasting Pipeline](#forecasting-pipeline)
7. [Model Architecture](#model-architecture)
8. [Installation and Setup](#installation-and-setup)
9. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Data Preparation](#data-preparation)
10. [Outputs](#outputs)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Reproducibility](#reproducibility)
13. [Notes and Recommendations](#notes-and-recommendations)
14. [Upcoming Upgrade (TFT + LightGBM + Ensemble)](#upcoming-upgrade-tft--lightgbm--ensemble)
    - [Why This Change](#why-this-change)
    - [What Will Change](#what-will-change)
    - [Dependencies and GPU Setup](#dependencies-and-gpu-setup)
    - [New Training Modes](#new-training-modes)
    - [Validation and Anti‑Leakage Policy](#validation-and-anti-leakage-policy)
    - [Deliverables and Reproducibility](#deliverables-and-reproducibility)

---

## Complete Pipeline: From Raw Data to Production

This section provides a comprehensive overview of the entire workflow, from raw data to production-ready predictions. Follow these steps sequentially to reproduce the complete system.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA (Excel files)                       │
│              pv_dataset.xlsx + wx_dataset.xlsx                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │  STEP 1: PREPROCESSING       │
          │  Feature Engineering         │
          │  - Lag features (1h, 24h,    │
          │    48h, 72h, 96h, 168h)      │
          │  - Rolling mean/var          │
          │    (3h, 6h, 12h, 24h)        │
          │  - Time + weekend/holiday    │
          │    + solar/clear-sky         │
          └──────────┬───────────────────┘
                     │
                     ▼
          outputs/processed.parquet
      (feature set esteso di default)
             (legacy: outputs/<model>/legacy_*)
                 │
                 ▼
    ┌────────────────────────────┐
    │  STEP 2: TRAIN MODELS      │
    │  3 architectures           │
    │                            │
    │  LightGBM                  │
    │  CNN-BiLSTM                │
    │  TFT                       │
    │                            │
    │  (legacy: varianti         │
    │   legacy_* in outputs/*/)  │
    └────────┬───────────────────┘
             │
             │ Each model produces:
             │ - predictions_val_*.csv (validation: 20%)
             │ - predictions_test_*.csv (test: 20%)
             │
             ▼
    ┌────────────────────────────┐
    │  STEP 3: ENSEMBLE          │
    │  OPTIMIZATION              │
    │  (on VALIDATION set)       │
    │                            │
    │  Exhaustive search:        │
    │  - Tests 57 combinations   │
    │  - Optimizes weights       │
    │  - Selects best ensemble   │
    └────────┬───────────────────┘
             │
             │ Outputs:
             │ - ensemble_weights.json
             │   {models: [...], weights: [...]}
             │
             ▼
    ┌────────────────────────────┐
    │  STEP 4: TEST EVALUATION   │
    │  (on TEST set)             │
    │                            │
    │  Apply optimized weights   │
    │  to never-seen test data   │
    │                            │
    │  → HONEST METRICS          │
    │    (report these!)         │
    └────────┬───────────────────┘
             │
             ▼
    ┌────────────────────────────┐
    │  STEP 5: PRODUCTION        │
    │  INFERENCE                 │
    │                            │
    │  Use EnsembleModel to      │
    │  predict on new data       │
    │  (professor's dataset)     │
    │                            │
    │  Input: meteo data (t)     │
    │  Output: 24h forecast      │
    └────────────────────────────┘
```

### Data Split Strategy (Critical)

```
Total Dataset: 17,374 hours
│
├─ TRAIN SET (60% = 10,410 hours)
│  Purpose: Train all 6 models
│  Used by: train_lgbm.py, train_cnn_bilstm.py, train_tft.py
│
├─ VALIDATION SET (20% = 3,470 hours)
│  Purpose: Optimize ensemble weights
│  Used by: ensemble.py (finds best combination + weights)
│  IMPORTANT: Never use this for final evaluation.
│
└─ TEST SET (20% = 3,494 hours)
   Purpose: Honest performance evaluation
   Used by: test_ensemble.py (final metrics)
   CRITICAL: This data is NEVER seen during training/optimization.

   → Test metrics = Expected performance on new data
   → These are the numbers you report in your thesis or paper.
```

**Why this split?**
- If we optimize ensemble on test set → **overfitting** → metrics too optimistic
- Using separate validation → test set truly held-out → **honest evaluation**

---

### Step-by-Step Instructions

#### STEP 1: Preprocessing (Default Extended Features)

La pipeline crea `outputs/processed.parquet` con il set esteso di feature:

- Lag: 1h, 24h, 48h, 72h, 96h, 168h
- Rolling mean & variance: 3h, 6h, 12h, 24h
- Time features, weekend/holiday flags, solar position, clear-sky, clearness index

I training script (`train_cnn_bilstm.py`, `train_tft.py`, `train_lgbm.py`) rigenerano automaticamente il parquet se non esiste. Se vuoi confrontare con i vecchi esperimenti legacy, usa le cartelle sotto `outputs/<model>/legacy_*` (es. `outputs/cnn/legacy_*`, `outputs/tft/legacy_*`); il default è il set esteso sopra.

---

#### STEP 2: Train Models

Per il flusso standard con la struttura attuale:

```bash
python scripts/training/train_lgbm.py --outdir outputs/lgbm/baseline
python scripts/training/train_cnn_bilstm.py --outdir outputs/cnn/baseline
python scripts/training/train_tft.py --outdir outputs/tft/baseline --use-future-meteo  # se hai NWP future
```

I comandi legacy sotto servono solo se vuoi replicare i vecchi esperimenti (opzionali).

**2b. CNN-BiLSTM Baseline:**
```bash
source .venv/bin/activate
bash scripts/training/train_cnn_baseline.sh

# Or directly:
python scripts/training/train_cnn_bilstm.py \
  --processed-path outputs/processed.parquet \
  --outdir outputs/cnn/baseline \
  --epochs 200 \
  --batch-size 64
```

**Training time:** ~280 minutes (~4 hours 40 minutes) - 28 epochs × ~10 min/epoch

**Outputs:**

- `outputs/<model>/legacy_* (best model)
- `outputs/<model>/legacy_* (validation predictions)
- `outputs/<model>/legacy_* (test predictions)
- `outputs/<model>/legacy_* (validation metrics)

**2c. TFT Baseline:**
```bash
python scripts/training/train_tft.py \
  --processed-path outputs/processed.parquet \
  --outdir outputs/tft/baseline
```

**Dove finiscono gli output (struttura attuale):**
- `outputs/cnn/baseline` (CNN-BiLSTM)
- `outputs/tft/baseline` (TFT)
- `outputs/lgbm/baseline` (LightGBM)
- Eventuali esperimenti legacy: `outputs/<model>/legacy_*`

---

#### STEP 3: Ensemble Optimization (on Validation Set)

Find the best combination of models and their optimal weights.

```bash
python scripts/evaluation/ensemble.py \
  --lgbm-baseline outputs/<model>/legacy_* \
  --lgbm-lag72 outputs/<model>/legacy_* \
  --cnn-baseline outputs/<model>/legacy_* \
  --cnn-lag72 outputs/<model>/legacy_* \
  --tft-baseline outputs/<model>/legacy_* \
  --tft-lag72 outputs/<model>/legacy_* \
  --outdir outputs_ensemble \
  --method exhaustive \
  --metric rmse
```

**What this does:**

1. **Loads** all 6 validation predictions
2. **Tests** all 57 possible combinations:
   - 15 pairs (2 models)
   - 20 triplets (3 models)
   - 15 quadruplets (4 models)
   - 6 quintuplets (5 models)
   - 1 sextet (all 6 models)
3. **Optimizes** weights for each combination using grid search
4. **Selects** the absolute best combination based on validation RMSE

**Time:** ~10-30 minutes

**Output: `outputs_ensemble/ensemble_weights.json`**
```json
{
  "model_names": ["LightGBM-Baseline", "CNN-BiLSTM-Lag72", "TFT-Lag72"],
  "weights": [0.40, 0.35, 0.25],
  "optimization_method": "exhaustive",
  "best_rmse": 6.720,
  "timestamp": "2025-12-04T..."
}
```

**Interpretation:**

- Best ensemble uses 3 models (the algorithm selects between 2 and 6 models based on validation performance)
- Each model has a weight (sum = 1.0)
- Final prediction = weighted average of selected models

---

#### STEP 4: Test Evaluation (on Test Set)

Apply the optimized weights to the test set for honest evaluation.

```bash
python scripts/evaluation/test_ensemble.py \
  --weights outputs_ensemble/ensemble_weights.json \
  --lgbm-baseline outputs/<model>/legacy_* \
  --lgbm-lag72 outputs/<model>/legacy_* \
  --cnn-baseline outputs/<model>/legacy_* \
  --cnn-lag72 outputs/<model>/legacy_* \
  --tft-baseline outputs/<model>/legacy_* \
  --tft-lag72 outputs/<model>/legacy_* \
  --outdir outputs_ensemble
```

**What this does:**

1. Loads ensemble weights from Step 3
2. Loads test predictions (never seen during optimization)
3. Combines predictions using optimized weights
4. Computes final metrics (RMSE, MASE, MAE) per horizon

**Output:**
```
TEST SET EVALUATION RESULTS
============================================================
Overall Performance:
  RMSE: 6.850 kW
  MASE: 0.850
  MAE:  5.420 kW

Per-Horizon Performance:
  h= 1: RMSE=3.52, MASE=0.36
  h= 6: RMSE=5.12, MASE=0.65
  h=12: RMSE=6.45, MASE=0.82
  h=18: RMSE=7.89, MASE=0.98
  h=24: RMSE=8.07, MASE=1.02
============================================================
```

**Note:** These test set metrics represent expected performance on unseen data and should be reported in the thesis or paper as the final performance evaluation.

---

#### STEP 5: Production Inference (Professor's Dataset)

Use the optimized ensemble to make predictions on completely new data.

**Option A: Using EnsembleModel (Recommended)**

```python
from pv_forecasting.ensemble_model import EnsembleModel
import pandas as pd

# Load ensemble (automatically loads all 6 models + weights)
ensemble = EnsembleModel.from_outputs(
    ensemble_dir="outputs_ensemble",
    baseline_dir="outputs/<model>/legacy_*
    lag72_dir="outputs/<model>/legacy_*
)

# Load professor's data (already preprocessed)
prof_data = pd.read_parquet("prof_processed_data.parquet")

# Make predictions
predictions = ensemble.predict(prof_data)
# Shape: (N_samples, 24) - 24-hour forecasts

# Save results
output_df = pd.DataFrame(
    predictions,
    columns=[f"h{i}" for i in range(1, 25)]
)
output_df.to_csv("predictions_prof/forecast_24h.csv", index=False)
```

**Option B: Using Command-Line Script**

```bash
python scripts/inference/predict.py \
  --input-pv prof_pv_data.xlsx \
  --input-wx prof_wx_data.xlsx \
  --ensemble-weights outputs_ensemble/ensemble_weights.json \
  --outdir predictions_prof
```

**Output format:**
```csv
timestamp,forecast_hour,predicted_power_kW
2025-12-05 00:00:00,1,5.23
2025-12-05 01:00:00,2,4.87
2025-12-05 02:00:00,3,3.12
...
2025-12-05 23:00:00,24,12.45
```

**Interpretation:**
- Input: Current weather conditions + recent PV production
- Output: Predicted PV power for next 24 hours
- Use for: Grid management, energy trading, dispatch planning

---

### Quick Command Reference

After training all 6 models (legacy baseline/lag72 setup), run these commands in sequence. For i nuovi run con feature estese, sostituisci i percorsi con le tue cartelle (`outputs_tft`, `outputs_cnn`, `outputs_lgbm`, ecc.).

```bash
# 1. Optimize ensemble (validation set)
python scripts/evaluation/ensemble.py \
  --lgbm-baseline outputs/<model>/legacy_* \
  --lgbm-lag72 outputs/<model>/legacy_* \
  --cnn-baseline outputs/<model>/legacy_* \
  --cnn-lag72 outputs/<model>/legacy_* \
  --tft-baseline outputs/<model>/legacy_* \
  --tft-lag72 outputs/<model>/legacy_* \
  --method exhaustive \
  --outdir outputs_ensemble

# 2. Test ensemble (test set - final evaluation metrics)
python scripts/evaluation/test_ensemble.py \
  --weights outputs_ensemble/ensemble_weights.json \
  --lgbm-baseline outputs/<model>/legacy_* \
  --lgbm-lag72 outputs/<model>/legacy_* \
  --cnn-baseline outputs/<model>/legacy_* \
  --cnn-lag72 outputs/<model>/legacy_* \
  --tft-baseline outputs/<model>/legacy_* \
  --tft-lag72 outputs/<model>/legacy_* \
  --outdir outputs_ensemble

# 3. Production inference (new data)
python -c "
from pv_forecasting.ensemble_model import EnsembleModel
ensemble = EnsembleModel.from_outputs('outputs_ensemble')
# predictions = ensemble.predict(your_data)
"
```

---

### Key Takeaways

1. **Feature set esteso** (lag multi-day, rolling mean/var, fisiche, calendario); i set baseline/lag72 restano solo per confronto legacy
2. **Three architectures** (LightGBM, CNN-BiLSTM, TFT) → Diversification
3. **Ensemble combines best models** → Better than any single model
4. **Validation for optimization** → Find best combination/weights
5. **Test for evaluation** → Honest performance estimate (report these metrics)
6. **EnsembleModel for production** → Simple unified interface

**For detailed documentation, see:**
- [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md) - **Complete data & model specification** (input/output formats, commands)
- [WORKFLOW.md](WORKFLOW.md) - Complete step-by-step guide
- [EXPERIMENTS.md](EXPERIMENTS.md) - Experimental design & tracking
- [METRICS_ANALYSIS.md](METRICS_ANALYSIS.md) - Performance benchmarks

---

## Project Structure

```
24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
├── pyproject.toml                 # Modern Python packaging & tool config
├── requirements.txt               # Python dependencies
│
├── .github/workflows/             # CI/CD pipelines
│   └── ci.yml                     # Automated testing & linting
├── .pre-commit-config.yaml        # Pre-commit hooks configuration
├── .gitignore                     # Git ignore rules
│
├── pv_forecasting/                # Main Python package
│   ├── __init__.py                # Package initialization with versioning
│   ├── data.py                    # Data loading utilities
│   ├── features.py                # Feature engineering
│   ├── logger.py                  # Structured logging
│   ├── metrics.py                 # Evaluation metrics
│   ├── model.py                   # Model architecture
│   ├── pipeline.py                # Feature engineering pipeline
│   ├── timeutils.py               # Timezone handling
│   └── window.py                  # Sliding window creation
│
├── tests/                         # Test suite (pytest)
│   ├── conftest.py                # Shared test fixtures
│   ├── test_features.py           # Feature engineering tests
│   ├── test_data.py               # Data loading tests
│   ├── test_metrics.py            # Metrics tests
│   └── test_pipeline.py           # Pipeline integration tests
│
├── data/                          # Data directory
│   ├── README.md                  # Data documentation
│   └── raw/                       # Original datasets
│       ├── pv_dataset.xlsx        # PV production data
│       └── wx_dataset.xlsx        # Weather data
│
├── scripts/                       # All executable scripts (organized by purpose)
│   ├── README.md                  # Scripts documentation
│   ├── data/                      # Data processing
│   │   └── preprocess_data.py     # Complete preprocessing pipeline
│   ├── preprocessing/             # Advanced preprocessing
│   │   └── generate_processed_lag72.py  # Generate dataset with 3-day lag features
│   ├── training/                  # Model training
│   │   ├── README.md              # Training guide
│   │   ├── train_lgbm.py          # LightGBM multi-horizon training
│   │   ├── train_tft.py           # TFT training
│   │   ├── train_cnn_bilstm.py    # CNN-BiLSTM training
│   │   ├── train_cnn_baseline.sh  # Helper: CNN with baseline features
│   │   └── train_cnn_lag72.sh     # Helper: CNN with 3-day lag features
│   ├── inference/                 # Prediction/deployment
│   │   └── predict.py             # Inference script
│   └── evaluation/                # Model evaluation
│       └── ensemble.py            # Ensemble optimization (flexible 2-6 models)
│
├── outputs/<model>/legacy_*              # Baseline experiments (45 features: lag1, lag24, lag168)
│   ├── processed.parquet          # Processed dataset (1.9MB)
│   ├── lgbm/                      # LightGBM baseline results
│   ├── cnn/                       # CNN-BiLSTM baseline results
│   └── tft/                       # TFT baseline results
│
├── outputs/<model>/legacy_*                 # Lag72 experiments (49 features: +lag72 3-day features)
│   ├── processed.parquet          # Processed dataset with lag72 (2.0MB)
│   ├── lgbm/                      # LightGBM with lag72 results
│   ├── cnn/                       # CNN-BiLSTM with lag72 results
│   └── tft/                       # TFT with lag72 results
│
├── outputs_ensemble/              # Final ensemble results (6 models)
│   ├── ensemble_weights.json      # Optimized model weights
│   ├── predictions_val_ensemble.csv
│   └── metrics_ensemble.json
│
├── METRICS_ANALYSIS.md            # Performance analysis & benchmarks
└── EXPERIMENTS.md                 # Experiment tracking & comparison
```

---

## Data Description

### Overview

- Builds a 24-hour ahead multistep forecast of PV power using historical PV and hourly weather data.
- Robust timestamp handling across DST: aligns PV (naive local time) with weather (timezone-aware) by localizing to site timezone and converting to UTC.
- Hybrid deep learning model: Conv1D + BiLSTM with 24 outputs (one per hour ahead).

### Dataset Files

- `pv_dataset.xlsx`: PV timestamps (no timezone) and normalized PV output (normalized on 82.41 kWp installed capacity).
- `wx_dataset.xlsx`: `dt_iso` with explicit timezone and hourly weather variables (T, dew point, pressure, humidity, wind, rain, cloud cover, DHI/DNI/GHI).

---

## Dataset Merging Strategy

### Why Merge the Datasets?

The forecasting model requires **synchronized** PV production and weather data at matching timestamps. However, the original datasets come from different sources with distinct characteristics:

- **PV Dataset**: Local timestamps without timezone information, measurements with millisecond precision (e.g., `16:59:59.985`)
- **Weather Dataset**: UTC-aware timestamps with explicit timezone offset (`+10:00`), hourly measurements

### Challenges Addressed

#### 1. Timezone Alignment

- **Problem**: PV data uses naive local time (Australia/Sydney), while weather data uses UTC+10:00
- **Solution**:
  - Localize PV timestamps to `Australia/Sydney` timezone
  - Convert both datasets to UTC for uniform alignment
  - This ensures temporal consistency across DST transitions

#### 2. Daylight Saving Time (DST) Handling

- **Spring Forward (March)**: Missing hour handled with `nonexistent='shift_forward'`
  - Times like 02:30 that don't exist are shifted to the next valid time
- **Fall Back (October)**: Ambiguous repeated hours handled with `ambiguous=False`
  - Assumes standard time (non-DST) for duplicated clock hours
- **Verification**: Correlation PV-GHI = 0.905 confirms correct alignment

#### 3. Timestamp Precision

- **Problem**: PV has millisecond timestamps (`.985`, `.980`, etc.), weather has exact hours
- **Solution**:
  - Preserve PV timestamp precision (no rounding)
  - Forward-fill weather data to match PV timestamps
  - Meteorological data changes slowly, making forward-fill appropriate

#### 4. Multi-Sheet Excel Files

- **Problem**: Each dataset has 2 sheets covering different periods:
  - `07-10--06-11`: October 2007 - June 2011
  - `07-11--06-12`: July 2011 - June 2012
- **Solution**: Automatically read all sheets and concatenate chronologically

#### 5. Data Quality

- Remove header rows with metadata (e.g., "Max kWp 82.41")
- Drop non-numeric columns from weather data
- Handle duplicate timestamps
- Validate data continuity

---

### Merged Dataset Characteristics

- **Total Records**: 17,542 hourly observations (raw) → 17,374 after feature engineering and lag removal
- **Period**: June 30, 2010 - June 30, 2012 (2 years)
- **Completeness**: 100% (no NaN values after preprocessing)
- **Columns**: 14 (raw) → 45 (after feature engineering)
- **Validation**: PV-GHI correlation = 0.905 (excellent physical alignment)

### Processed Dataset Features (45 total)

After running `python scripts/preprocess_data.py`, the final dataset includes:

#### Target Variable (1)
- `pv`: Normalized PV power output (0-1 scale, normalized on 82.41 kWp capacity)

#### Meteorological Features (8)
- `temp`: Temperature (K)
- `humidity`: Relative humidity (%)
- `wind_speed`: Wind speed (m/s)
- `clouds`: Cloud cover (%)
- `rain_1h`: Precipitation in last hour (mm)
- `pressure`: Atmospheric pressure (hPa)
- `dew_point`: Dew point temperature (K)
- `weather_description`: **Encoded weather conditions (0-10 scale)**
  - 10.0 = clear sky/sunny (maximum PV production)
  - 8.0 = few clouds (80-90% production)
  - 6.0 = scattered clouds (60-70% production)
  - 4.0 = overcast/cloudy (40-50% production)
  - 2.0 = light rain (20-30% production)
  - 1.0 = heavy rain/storm (10-20% production)
  - 0.0 = fog/mist (0-10% production)

#### Solar Irradiance (3)
- `ghi`: Global Horizontal Irradiance (W/m²)
- `dni`: Direct Normal Irradiance (W/m²)
- `dhi`: Diffuse Horizontal Irradiance (W/m²)

#### Physics-Based Features (7)
- `sp_zenith`: Solar zenith angle (degrees)
- `sp_azimuth`: Solar azimuth angle (degrees)
- `cs_ghi`: Clear-sky Global Horizontal Irradiance (W/m²)
- `cs_dni`: Clear-sky Direct Normal Irradiance (W/m²)
- `cs_dhi`: Clear-sky Diffuse Horizontal Irradiance (W/m²)
- `kc`: **Clearness index** = measured_GHI / clearsky_GHI
  - 1.0 = perfect clear sky
  - 0.5 = 50% blocked by clouds
  - 0.0 = nighttime or fully overcast

#### Time Features (4)
- `hour_sin`, `hour_cos`: Cyclical hour-of-day encoding (0-23h)
- `doy_sin`, `doy_cos`: Cyclical day-of-year encoding (1-365/366)

#### Lag Features (12)
Historical values at t-1, t-24 (1 day), t-168 (1 week):
- `pv_lag1`, `pv_lag24`, `pv_lag168`
- `ghi_lag1`, `ghi_lag24`, `ghi_lag168`
- `dni_lag1`, `dni_lag24`, `dni_lag168`
- `dhi_lag1`, `dhi_lag24`, `dhi_lag168`

#### Rolling Statistics (6)
Moving averages over 3h and 6h windows:
- `pv_roll3h`, `pv_roll6h`
- `ghi_roll3h`, `ghi_roll6h`
- `dni_roll3h`, `dni_roll6h`

#### Metadata (4)
- `lat`, `lon`: Site location coordinates
- `time_idx`: Sequential integer index (0, 1, 2, ...)
- `series_id`: Constant identifier "pv_site_1"

### Dataset Processing Pipeline

The project uses a streamlined 2-stage data processing pipeline:

#### Stage 1: Raw Data
**Location:** `data/raw/`

Two Excel files containing the source data:
- `pv_dataset.xlsx`: PV production measurements (2 sheets covering 2010-2012, 17,542 total rows)
- `wx_dataset.xlsx`: Weather observations (2 sheets covering 2010-2012, 17,544 total rows)

#### Stage 2: Processed Dataset (Ready for Training)
**Location:** `outputs/processed.parquet`

A single script processes everything in one step:

```bash
python scripts/preprocess_data.py
```

**What this script does:**
1. Loads both raw Excel files
2. Merges them with timezone-aware alignment (UTC)
3. Validates temporal alignment (PV-GHI correlation = 0.905)
4. Applies complete feature engineering:
   - Physics-based features (solar position, clear-sky, clearness index)
   - Weather description encoding (text → numerical 0-10 scale)
   - Temporal features (cyclical time, lags, rolling statistics)
5. Cleans data (fills NaN, removes lag initialization rows)
6. Saves optimized Parquet format

**Output characteristics:**
- 17,374 samples (after lag removal)
- 45 features (target + 44 engineered features)
- 0% NaN values (100% complete)
- 1.88 MB file size (fast loading, 10x faster than CSV)

**All training scripts use this file** with the `--processed-path outputs/processed.parquet` argument.

---

## Time Handling and DST Management

1. Interpret PV timestamps as local site time (default `Australia/Sydney`)
2. Apply `tz_localize(local_tz, ambiguous=False, nonexistent='shift_forward')` on PV data
3. Convert both PV and weather datasets to UTC timezone
4. Merge datasets on UTC timestamps
5. Validate: ensure no duplicates, no gaps, and perfect timestamp alignment

---

## Forecasting Pipeline

1. Load and normalize timestamps with UTC alignment
2. Merge datasets on UTC timestamp
3. Clean gaps and duplicates; sort chronologically
4. Feature engineering:
   - Cyclical time features: hour-of-day (sin/cos) and day-of-year (sin/cos)
   - Lag features: t-1, t-24, t-168 for PV and GHI
   - Rolling statistics: 3h and 6h means on key meteorological and irradiance variables
5. Build sliding windows: input 168 hours to predict next 24 hours PV output
6. Train hybrid CNN-BiLSTM model
7. Evaluate metrics: MASE (primary), RMSE; compare to naive baseline PV(t)=PV(t-24)
8. Export predictions on test split to CSV format

Note: this section describes the current baseline pipeline. See the upgrade section for the new TFT + LightGBM + ensemble workflow.

---

## Model Architecture

The current baseline uses a **hybrid CNN-BiLSTM architecture**:

- **Input Layer**: 168 timesteps × n features
- **Conv1D Layers**: Extract local patterns and reduce dimensionality
- **Bidirectional LSTM**: Capture temporal dependencies in both directions
- **Dense Output**: 24 neurons (one per forecast hour)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with early stopping

---

## Development & Testing

### Testing

The project includes a comprehensive test suite with 40+ tests covering all major components:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=pv_forecasting --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"

# Run specific test file
pytest tests/test_features.py

# Run with verbose output
pytest -v
```

Test coverage reports are generated in `htmlcov/index.html`.

### Code Quality & Linting

The project uses automated code quality tools:

```bash
# Format code with Black (line length 120)
black pv_forecasting/ tests/ *.py

# Sort imports with isort
isort pv_forecasting/ tests/ *.py

# Type check with mypy
mypy pv_forecasting/ --ignore-missing-imports

# Run all checks at once
pre-commit run --all-files
```

### Pre-commit Hooks

Install pre-commit hooks to automatically run checks before each commit:

```bash
pip install pre-commit
pre-commit install
```

Now every `git commit` will automatically:
- Format code with Black
- Sort imports with isort
- Check types with mypy
- Validate YAML/JSON files
- Check for large files and secrets

### Continuous Integration

GitHub Actions automatically runs:
- Tests on Python 3.9, 3.10, 3.11, 3.12
- Linting and formatting checks
- Code coverage analysis
- Package building and validation

See `.github/workflows/ci.yml` for the full CI configuration.

---

## Installation and Setup

### Requirements

Python 3.9+ and pip.

### Create Environment and Install Dependencies

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

#### Quickstart su WSL (Ubuntu)

```bash
# attiva WSL (Ubuntu) e spostati nella repo
cd ~/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting
# crea e attiva il venv
python3 -m venv .venv
source .venv/bin/activate
# installa le dipendenze (GPU CUDA già inclusa in torch 2.9.1+cu128 installata sopra)
pip install -r requirements.txt
# verifica GPU
python - <<'PY'
import torch
print(torch.__version__, 'cuda:', torch.cuda.is_available())
PY
```

### Pipeline B: TFT (PyTorch Forecasting) + LightGBM + Ensemble (WSL/Ubuntu)

1) **Data & anti‑leakage**
   - Colonne meteo normalizzate a lowercase (`ghi`, `dni`, `dhi`, `temp`, `humidity`, `clouds`, `wind_speed`).
   - `pv_forecasting/pipeline.py` crea feature (cicliche + flag weekend/holiday, lag 1/24/48/72/96/168, rolling mean/var 3/6/12/24h, solar/clear‑sky) e salva `outputs/processed.parquet`.
   - Due modalità: `--use-future-meteo` se hai covariate future (NWP); altrimenti solo lag + feature fisiche future (derivabili dal timestamp).

2) **LightGBM multi‑orizzonte (24 modelli)**
```bash
python scripts/training/train_lgbm.py --outdir outputs_lgbm --walk-forward-folds 3  # opzionale WF
```
   - Salva modelli per h=1..24 in `outputs_lgbm/models/`, predizioni/metriche validation e (se richiesto) walk‑forward.

3) **TFT (PyTorch Forecasting)**
```bash
python scripts/training/train_tft.py --outdir outputs_tft --use-future-meteo   # se hai meteo future
```
   - Crea dataset encoder 168h / decoder 24h, quantile loss (0.1/0.5/0.9), early stopping, checkpoint `tft-best.ckpt`, predizioni/metriche validation per orizzonte.

4) **Ensemble su validation**
```bash
python scripts/ensemble.py \
  --tft-preds outputs_tft/predictions_val_tft.csv \
  --lgbm-preds outputs_lgbm/predictions_val_lgbm.csv \
  --processed-path outputs/processed.parquet
```
   - Cerca il peso TFT che minimizza RMSE, salva blend e metriche in `outputs_ensemble/`.

5) **Inference (dataset prof)**
```bash
python scripts/inference/predict.py \
  --pv-path data/raw/pv_dataset.xlsx \
  --wx-path data/raw/wx_dataset.xlsx \
  --future-wx-path <opzionale NWP> \
  --lgbm-dir outputs_lgbm \
  --tft-ckpt outputs_tft/tft-best.ckpt \
  --ensemble-weights outputs_ensemble/ensemble_weights.json
```
   - Auto‑switch con/senza meteo future; genera `outputs_predictions/predictions_next24.csv`.

---

## Usage

### Training the Model

Run training with default parameters:

```bash
python scripts/training/train_cnn_bilstm.py --outdir outputs_cnn
```

With custom parameters:

```bash
python scripts/training/train_cnn_bilstm.py \
  --pv-path data/raw/pv_dataset.xlsx \
  --wx-path data/raw/wx_dataset.xlsx \
  --local-tz Australia/Sydney \
  --seq-len 168 \
  --horizon 24 \
  --epochs 100 \
  --batch-size 64 \
  --outdir outputs_cnn
```

### LightGBM Baseline (24 regressori)

```bash
python scripts/training/train_lgbm.py --outdir outputs_lgbm
```

### Temporal Fusion Transformer (PyTorch Forecasting)

```bash
python scripts/training/train_tft.py --outdir outputs_tft --use-future-meteo   # se hai NWP/meteo future
```
   - Default: encoder 168h / decoder 24h, hidden 64, 4 heads, dropout 0.25, AdamW (wd=1e-4), quantile loss (0.1/0.5/0.9).
   - Pesi diurni nella loss via `--dayweight-gamma` (default 1.5) e `--dayweight-min` (default 0.1); campioni notturni esclusi dalle metriche con `--metrics-zenith-max 90` (puoi disattivare con `--metrics-zenith-max None`).
   - Se la tua versione di PyTorch Forecasting supporta `output_dropout`, puoi impostarlo con `--output-dropout 0.1` per ulteriore regolarizzazione del gating.

### Ensemble: Combining Multiple Models

The project supports flexible ensemble optimization for 2-6 models:

#### Example: 2-Model Ensemble (TFT + LightGBM)

```bash
python scripts/evaluation/ensemble.py \
  --tft outputs_tft/predictions_val_tft.csv \
  --lgbm outputs_lgbm/predictions_val_lgbm.csv \
  --outdir outputs_ensemble \
  --method grid \
  --metric rmse
```

#### Example: 3-Model Ensemble (All Baseline Models)

```bash
python scripts/evaluation/ensemble.py \
  --tft outputs/<model>/legacy_* \
  --lgbm outputs/<model>/legacy_* \
  --bilstm outputs/<model>/legacy_* \
  --outdir outputs_ensemble \
  --method optuna \
  --optuna-trials 200 \
  --metric rmse
```

#### Example: 6-Model Grand Ensemble with Exhaustive Search (RECOMMENDED)

**BEST APPROACH:** Let the algorithm try ALL possible combinations (2-6 models) and find the optimal ensemble:

```bash
python scripts/evaluation/ensemble.py \
  --lgbm-baseline outputs/<model>/legacy_* \
  --lgbm-lag72 outputs/<model>/legacy_* \
  --cnn-baseline outputs/<model>/legacy_* \
  --cnn-lag72 outputs/<model>/legacy_* \
  --tft-baseline outputs/<model>/legacy_* \
  --tft-lag72 outputs/<model>/legacy_* \
  --outdir outputs_ensemble \
  --method exhaustive \
  --metric rmse
```

This will:
- Test all 57 possible combinations (15 pairs + 20 triplets + 15 quads + 6 quints + 1 sextet)
- For each combination, optimize weights using grid search
- Automatically select the best-performing subset
- **Guarantees optimal solution** (may take longer: ~10-30 minutes for 6 models)

**Alternative:** Force using all 6 models with Optuna optimization

```bash
python scripts/evaluation/ensemble.py \
  --lgbm-baseline outputs/<model>/legacy_* \
  --lgbm-lag72 outputs/<model>/legacy_* \
  --cnn-baseline outputs/<model>/legacy_* \
  --cnn-lag72 outputs/<model>/legacy_* \
  --tft-baseline outputs/<model>/legacy_* \
  --tft-lag72 outputs/<model>/legacy_* \
  --outdir outputs_ensemble \
  --method optuna \
  --optuna-trials 200 \
  --metric rmse
```

**Optional: Subset of models** (example with 3 best performers)

```bash
# After comparing individual model metrics, select top 3
python scripts/evaluation/ensemble.py \
  --lgbm-baseline outputs/<model>/legacy_* \
  --cnn-lag72 outputs/<model>/legacy_* \
  --tft-lag72 outputs/<model>/legacy_* \
  --outdir outputs_ensemble \
  --method grid \
  --metric rmse
```

**Outputs:**

- `ensemble_weights.json`: Optimized weights for each model
- `predictions_val_ensemble.csv`: Blended predictions
- `metrics_ensemble.json`: Performance comparison (individual vs ensemble)

### Inference

Previsione day-ahead (auto-switch con/senza meteo future se forniti):

```bash
python scripts/inference/predict.py \
  --pv-path data/raw/pv_dataset.xlsx \
  --wx-path data/raw/wx_dataset.xlsx \
  --future-wx-path <opzionale>
```

### Data Preparation

Generate the complete preprocessed dataset with all features:

```bash
python scripts/preprocess_data.py
```

This script:
- Loads raw PV and weather Excel files
- Merges them with timezone-aware UTC alignment
- Applies complete feature engineering (45 features total)
- Validates data quality and alignment
- Saves to `outputs/processed.parquet` (ready for training)

---

## Outputs

After training, the following files are generated in the `outputs/` directory:

- `processed.parquet`: Merged, feature-engineered time series (UTC-indexed)
- `outputs_lgbm/`: LightGBM per-horizon models (`models/lgbm_h*.joblib`), test predictions/metrics
- `outputs_tft/`: TFT checkpoint (`tft-best.ckpt`), test predictions/metrics
- `outputs_cnn/`: CNN-BiLSTM model (`model_best.keras`), scaler, test predictions
  - `scalers.joblib`: Fitted feature scaler (StandardScaler)
  - `model_best.keras`: Best model weights (lowest validation loss)
  - `history.json`: Training and validation loss history
  - `predictions_test_cnn.csv`: Long-format predictions for the test split
- `outputs_ensemble/`: Ensemble blending weights and blended predictions

### Predictions CSV Format

- **Columns**: `origin_timestamp_utc`, `forecast_timestamp_utc`, `horizon_h`, `y_true`, `y_pred`
- `origin_timestamp_utc`: Last timestamp of the input window (UTC)
- `forecast_timestamp_utc`: Timestamp of each horizon target (UTC)
- `horizon_h`: Forecast horizon (1 to 24 hours ahead)

---

## Evaluation Metrics

The model is evaluated using:

1. **MASE (Mean Absolute Scaled Error)**: Primary metric
   - Scales error relative to naive seasonal baseline (24h lag)
   - MASE < 1 indicates better than naive forecast
   
2. **RMSE (Root Mean Squared Error)**: Secondary metric
   - Measures prediction accuracy in kW units
   - Penalizes larger errors more heavily

3. **Baseline Comparison**: Naive persistence model (PV(t) = PV(t-24))

---

## Reproducibility

- Fixed random seeds for NumPy and TensorFlow ensure deterministic results
- Chronological train/validation/test split: 70%/10%/20% on windowed samples
- Deterministic training with consistent data preprocessing pipeline

---

## Notes and Recommendations

- If the site local timezone is not `Australia/Sydney`, pass `--local-tz <IANA_TZ>` when running scripts
- DST transitions are handled automatically with `ambiguous=False` (standard time) and `nonexistent='shift_forward'`
- The model requires at least 168 hours (7 days) of historical data for each prediction
- Correlation PV-GHI = 0.905 validates correct temporal alignment of merged data

---

## Upcoming Upgrade (TFT + LightGBM + Ensemble)

This repository is being upgraded to a competition‑ready pipeline to maximize day‑ahead accuracy and robustness.

### Why This Change

- Achieve state‑of‑the‑art accuracy on hidden test sets with diverse conditions.
- Leverage physics‑informed features to generalize across seasons and weather regimes.
- Provide probabilistic outputs (quantiles) when needed and ensure strict anti‑leakage.

### What Will Change

- Features (pvlib): add solar position (zenith/azimuth), clear‑sky (Ineichen), optional POA, and clearness index (CSI).
- Baselines: add clear‑sky persistence and a strong LightGBM multi‑horizon forecaster.
- SOTA model: add TFT (via Darts or PyTorch Forecasting) with quantile loss (e.g., q10/50/90).
- Ensemble: combine TFT + LightGBM with validation‑optimized weights; night‑time clipping and [0, capacity] constraints.
- Validation: move from single split to walk‑forward multi‑fold; report metrics by horizon and skill scores.
- Data hygiene: unify weather column naming to lowercase (e.g., `ghi`, `dni`, `dhi`) and rename rolling features to `roll{w}h`.
- Processed format: standardize on Parquet in `outputs/processed.parquet` (CSV in `data/processed` kept for reference but deprecated).

### Dependencies and GPU Setup

- Add: `pvlib`, `lightgbm`, `optuna`, and for TFT either `darts[u]` or `pytorch-forecasting` + `pytorch-lightning`.
- Install PyTorch with CUDA (example for CUDA 12.1 on Windows):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Then one of:
pip install darts[u] pytorch-lightning
# or
pip install pytorch-forecasting pytorch-lightning
```

- Verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### New Training Modes

- Baseline (current): `python scripts/training/train_cnn_bilstm.py` trains CNN‑BiLSTM.
- LightGBM (new): `python scripts/training/train_cnn_bilstm.py --model lightgbm` trains 24 horizons with engineered features.
- TFT (new): `python scripts/training/train_tft.py` (or `python scripts/training/train_cnn_bilstm.py --model tft`) trains a Temporal Fusion Transformer with quantile loss.
- Ensemble (new): `python scripts/ensemble.py` finds validation weights and merges predictions.

Notes:
- Two operational modes will be supported for day‑ahead:
  - `with_future_meteo`: if future weather covariates are provided (NWP/test file)
  - `no_future_meteo`: if not available; the model uses only past lags and future time/solar/clear‑sky features

### Validation and Anti‑Leakage Policy

- Walk‑forward expanding validation with 3–5 folds (no shuffle).
- Strict anti‑leakage: features at time t use only information available at t; future covariates are allowed only if provided as known inputs (e.g., NWP).
- Metrics reported globally and per horizon (1–24h), plus skill vs persistence (t‑24) and clear‑sky persistence.

### Deliverables and Reproducibility

- Models: `outputs/model_lgbm/*.bin`, `outputs/model_tft/*.ckpt`.
- Data artifacts: `outputs/processed.parquet`, scalers/encoders.
- Predictions and metrics: per‑fold CSVs, final `predictions_test.csv`.
- A `predict.py` script will load the final models and produce predictions from the professor’s test file with automatic mode detection.

---

## Project Information

**Author**: Claudio Dragotta
**Date**: November 2025
**Institution**: Deep Learning Course - Magistrale  
**Repository**: [24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting](https://github.com/claudio-dragotta/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting)

---

## Technical Reference: Complete Data & Model Specification

This section provides exhaustive technical details about data formats, model I/O, and step-by-step commands.

### 1. Data Format Specification

#### Raw Data (Input Excel Files)

**`data/raw/pv_dataset.xlsx`** - PV Production Data
```
Sheets: ['07-10--06-11', '07-11--06-12']
Columns:
  - timestamp: datetime (naive, Australia/Sydney local time)
  - pv: float [0.0, 1.0] - Normalized power (actual_kW / 82.41 kWp)

Example row:
  timestamp: 2010-07-01 06:00:00.000
  pv: 0.0023
```

**`data/raw/wx_dataset.xlsx`** - Weather Data
```
Sheets: ['07-10--06-11', '07-11--06-12']
Columns (14 total):
  - dt_iso: datetime string with timezone (e.g., "2010-07-01 10:00:00 +10:00")
  - lat, lon: float (site location: -33.86, 151.21 Sydney)
  - temp: float [K] (temperature in Kelvin, ~280-310K)
  - dew_point: float [K]
  - pressure: float [hPa] (~1000-1030)
  - humidity: float [%] (0-100)
  - wind_speed: float [m/s]
  - wind_deg: float [degrees] (0-360)
  - rain_1h: float [mm] (precipitation)
  - clouds_all: float [%] (cloud cover 0-100)
  - Ghi: float [W/m²] (Global Horizontal Irradiance, 0-1200)
  - Dni: float [W/m²] (Direct Normal Irradiance, 0-1000)
  - Dhi: float [W/m²] (Diffuse Horizontal Irradiance, 0-400)
```

#### Processed Data (Parquet Format)

**`outputs/<model>/legacy_* - 45 Features
```
Shape: (17,374 rows x 45 columns)
Size: ~1.9 MB

Index: DatetimeIndex (UTC timezone)
  - Range: 2010-07-08 00:00:00 to 2012-06-29 23:00:00
  - Frequency: Hourly (17,374 hours = ~2 years)

Columns by category:

TARGET (1):
  pv                float64   [0, 1]        Normalized PV output

METEOROLOGICAL (8):
  temp              float64   [K]           Temperature
  humidity          float64   [%]           Relative humidity
  wind_speed        float64   [m/s]         Wind speed
  clouds_all        float64   [%]           Cloud cover
  rain_1h           float64   [mm]          Hourly precipitation
  pressure          float64   [hPa]         Atmospheric pressure
  dew_point         float64   [K]           Dew point
  weather_desc      float64   [0-10]        Encoded weather condition

SOLAR IRRADIANCE (3):
  ghi               float64   [W/m²]        Global Horizontal Irradiance
  dni               float64   [W/m²]        Direct Normal Irradiance
  dhi               float64   [W/m²]        Diffuse Horizontal Irradiance

PHYSICS-BASED (7):
  sp_zenith         float64   [degrees]     Solar zenith angle
  sp_azimuth        float64   [degrees]     Solar azimuth angle
  cs_ghi            float64   [W/m²]        Clear-sky GHI (pvlib)
  cs_dni            float64   [W/m²]        Clear-sky DNI
  cs_dhi            float64   [W/m²]        Clear-sky DHI
  kc                float64   [0, 1+]       Clearness index (GHI/cs_GHI)
  is_night          int64     [0, 1]        Night flag (zenith > 90)

TIME FEATURES (4):
  hour_sin          float64   [-1, 1]       sin(2*pi*hour/24)
  hour_cos          float64   [-1, 1]       cos(2*pi*hour/24)
  doy_sin           float64   [-1, 1]       sin(2*pi*day_of_year/365)
  doy_cos           float64   [-1, 1]       cos(2*pi*day_of_year/365)

LAG FEATURES (12):
  pv_lag1           float64                 PV at t-1h
  pv_lag24          float64                 PV at t-24h (yesterday same hour)
  pv_lag168         float64                 PV at t-168h (last week same hour)
  ghi_lag1, ghi_lag24, ghi_lag168          GHI lags
  dni_lag1, dni_lag24, dni_lag168          DNI lags
  dhi_lag1, dhi_lag24, dhi_lag168          DHI lags

ROLLING STATISTICS (6):
  pv_roll3h         float64                 3-hour rolling mean of PV
  pv_roll6h         float64                 6-hour rolling mean of PV
  ghi_roll3h, ghi_roll6h                   GHI rolling means
  dni_roll3h, dni_roll6h                   DNI rolling means

METADATA (4):
  time_idx          int64     [0, N-1]      Sequential index for TFT
  series_id         str       "pv_site_1"   Series identifier
  lat               float64   -33.86        Latitude
  lon               float64   151.21        Longitude
```

**`outputs/<model>/legacy_* - 49 Features (+4 lag72)
```
Same as above PLUS:

ADDITIONAL LAG72 FEATURES (4):
  pv_lag72          float64                 PV at t-72h (3 days ago)
  ghi_lag72         float64                 GHI at t-72h
  dni_lag72         float64                 DNI at t-72h
  dhi_lag72         float64                 DHI at t-72h

Shape: (17,374 rows x 49 columns)
```

---

### 2. Data Split Strategy

```
+------------------------------------------------------------------+
|                    CHRONOLOGICAL SPLIT                            |
|              (NO SHUFFLE - Time Series Integrity)                 |
+------------------------------------------------------------------+

Total samples: 17,374 hours

TRAIN SET (60%):
  - Samples: 0 to 10,424 (10,425 samples)
  - Period: 2010-07-08 to 2011-09-03
  - Purpose: Model parameter learning

VALIDATION SET (20%):
  - Samples: 10,425 to 13,899 (3,475 samples)
  - Period: 2011-09-03 to 2012-01-27
  - Purpose: Ensemble weight optimization (NO data leakage!)

TEST SET (20%):
  - Samples: 13,900 to 17,373 (3,474 samples)
  - Period: 2012-01-27 to 2012-06-29
  - Purpose: Final honest evaluation (NEVER seen during training/tuning)
```

---

### 3. Model Specifications

#### LightGBM Multi-Horizon

**Architecture:**
- 24 independent LightGBM regressors (one per forecast horizon h=1..24)
- Each model: gradient boosting with 500 trees, max_depth=8, learning_rate=0.05

**Input:** Feature vector X at time t
```python
X.shape = (n_samples, n_features)  # n_features = 45 or 49

# Features used (all columns except target and metadata):
FEATURE_COLS = [
    'temp', 'humidity', 'wind_speed', 'clouds_all', 'rain_1h',
    'pressure', 'dew_point', 'ghi', 'dni', 'dhi',
    'sp_zenith', 'sp_azimuth', 'cs_ghi', 'cs_dni', 'cs_dhi', 'kc',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'pv_lag1', 'pv_lag24', 'pv_lag168',
    'ghi_lag1', 'ghi_lag24', 'ghi_lag168', ...
]
```

**Output:** 24 predictions (one per horizon)
```python
predictions.shape = (n_samples, 24)
# predictions[i, h-1] = predicted PV at time t+h for sample i
```

**Files produced:**
```
outputs/<model>/legacy_*
├── models/
│   ├── lgbm_h1.joblib      # Model for h=1 (1 hour ahead)
│   ├── lgbm_h2.joblib      # Model for h=2
│   └── lgbm_h24.joblib     # Model for h=24 (24 hours ahead)
├── predictions_val_lgbm.csv   # Validation predictions (for ensemble)
├── predictions_test_lgbm.csv  # Test predictions (for evaluation)
└── metrics_test_lgbm.json     # Per-horizon metrics
```

---

#### CNN-BiLSTM Sequence-to-Sequence

**Architecture:**
```
Input (168, n_features)     # 168 hours = 7 days lookback
       |
Conv1D(64 filters, kernel=3, ReLU)
       |
MaxPool1D(2) -> (84, 64)
       |
Conv1D(128 filters, kernel=3, ReLU)
       |
MaxPool1D(2) -> (42, 128)
       |
Bidirectional(LSTM(128, return_sequences=False)) -> (256,)
       |
Dense(128, ReLU)
       |
Dropout(0.3)
       |
Dense(24)                   # 24 hours forecast
       |
Output (24,)
```

**Input:** Sliding window of 168 hours
```python
X.shape = (n_windows, 168, n_features)
# Window at position i contains:
# - Features from time t-167 to t (168 consecutive hours)
# - Target: PV values from t+1 to t+24 (next 24 hours)
```

**Output:** 24-hour forecast vector
```python
predictions.shape = (n_windows, 24)
```

**Files produced:**
```
outputs/<model>/legacy_*
├── model_best.keras        # Best model (lowest val loss)
├── scalers.joblib          # StandardScaler for features
├── history.json            # Training history (loss curves)
├── predictions_val_cnn.csv    # Validation predictions
└── predictions_test_cnn.csv   # Test predictions
```

---

#### TFT (Temporal Fusion Transformer)

**Architecture:**
- PyTorch Forecasting implementation
- Variable selection networks for feature importance
- Multi-head attention over encoder sequence
- Gated residual connections
- Encoder length: 168 hours, Prediction length: 24 hours

**Files produced:**
```
outputs/<model>/legacy_*
├── checkpoints/
│   └── epoch=XX-step=YY.ckpt   # PyTorch Lightning checkpoint
├── predictions_val_tft.csv
└── predictions_test_tft.csv
```

---

### 4. Predictions CSV Format

All models output predictions in identical long format:

```csv
origin_timestamp_utc,forecast_timestamp_utc,horizon_h,y_true,y_pred
2012-01-28T00:00:00+00:00,2012-01-28T01:00:00+00:00,1,0.0,0.002
2012-01-28T00:00:00+00:00,2012-01-28T02:00:00+00:00,2,0.0,0.001
2012-01-28T00:00:00+00:00,2012-01-29T00:00:00+00:00,24,0.0,0.003
```

**Columns:**
- `origin_timestamp_utc`: Time when forecast is made
- `forecast_timestamp_utc`: Time being predicted
- `horizon_h`: Forecast horizon (1-24 hours ahead)
- `y_true`: Actual PV value
- `y_pred`: Model prediction

---

### 5. Ensemble System

#### What Ensemble Does

1. **Loads** validation predictions from all models
2. **Aligns** predictions by (origin_timestamp, forecast_timestamp, horizon)
3. **Searches** all possible model combinations (1 to N models)
4. **Optimizes** weights using grid search to minimize RMSE
5. **Selects** best combination based on validation performance

#### Combinations Tested (3 models = 7 combinations)

```
1-model: LightGBM, CNN-BiLSTM, TFT                    (3)
2-model: LightGBM+CNN, LightGBM+TFT, CNN+TFT          (3)
3-model: LightGBM+CNN+TFT                             (1)
Total: 7 combinations
```

#### Ensemble Formula

```python
y_pred_ensemble = sum(weight_i * y_pred_i)
# weights must sum to 1.0

# Example:
# weights = [0.4, 0.35, 0.25]
# y_pred = 0.4 * lgbm + 0.35 * cnn + 0.25 * tft
```

#### Output Files
```
outputs_ensemble/
├── ensemble_weights.json       # Optimized weights
├── predictions_val_ensemble.csv
├── predictions_test_ensemble.csv
└── metrics_ensemble.json
```

**`ensemble_weights.json` example:**
```json
{
  "model_names": ["LightGBM", "CNN-BiLSTM", "TFT"],
  "weights": [0.45, 0.30, 0.25],
  "optimization_method": "exhaustive",
  "best_val_rmse": 6.234,
  "note": "Weights optimized on VALIDATION set to prevent data leakage"
}
```

---

### 6. Complete Command Reference

#### Step 1: Preprocessing

```bash
# Generate baseline processed data (45 features)
python scripts/data/preprocess_data.py \
  --pv-path data/raw/pv_dataset.xlsx \
  --wx-path data/raw/wx_dataset.xlsx \
  --output-path outputs/<model>/legacy_*

# Generate lag72 processed data (49 features)
python scripts/preprocessing/generate_processed_lag72.py
```

#### Step 2: Train Models

```bash
# LightGBM (fast, ~5 minutes)
python scripts/training/train_lgbm.py \
  --processed-path outputs/<model>/legacy_* \
  --outdir outputs/<model>/legacy_* \
  --use-future-meteo

# CNN-BiLSTM (slow, ~3-4 hours on GPU)
python scripts/training/train_cnn_bilstm.py \
  --processed-path outputs/<model>/legacy_* \
  --outdir outputs/<model>/legacy_* \
  --epochs 200 \
  --batch-size 64 \
  --use-future-meteo

# TFT (medium, ~1-2 hours on GPU)
python scripts/training/train_tft.py \
  --processed-path outputs/<model>/legacy_* \
  --outdir outputs/<model>/legacy_* \
  --max-epochs 100 \
  --use-future-meteo
```

#### Step 3: Ensemble Optimization

```bash
python scripts/evaluation/ensemble.py \
  --lgbm-val outputs/<model>/legacy_* \
  --cnn-val outputs/<model>/legacy_* \
  --tft-val outputs/<model>/legacy_* \
  --lgbm-test outputs/<model>/legacy_* \
  --cnn-test outputs/<model>/legacy_* \
  --tft-test outputs/<model>/legacy_* \
  --outdir outputs_ensemble \
  --method exhaustive \
  --metric rmse
```

---

### 7. Testing Professor's Dataset

#### Scenario A: Raw Excel files provided

```bash
# 1. Preprocess the new data
python scripts/data/preprocess_data.py \
  --pv-path professor_pv.xlsx \
  --wx-path professor_wx.xlsx \
  --output-path professor_processed.parquet

# 2. Run inference
python scripts/inference/predict.py \
  --processed-data professor_processed.parquet \
  --ensemble-weights outputs_ensemble/ensemble_weights.json \
  --model-dir outputs/<model>/legacy_* \
  --outdir predictions_professor
```

#### Scenario B: Python Script

```python
import pandas as pd
from pv_forecasting.ensemble_model import EnsembleModel

# Load trained ensemble
ensemble = EnsembleModel.from_outputs(
    ensemble_dir="outputs_ensemble",
    outputs_dir="outputs/<model>/legacy_*
)

# Load professor's data
prof_data = pd.read_parquet("professor_processed.parquet")

# Make predictions
predictions = ensemble.predict(prof_data)
# predictions.shape = (n_samples, 24)

# Save results
output_df = pd.DataFrame({
    "timestamp": prof_data.index,
    **{f"h{h}": predictions[:, h-1] for h in range(1, 25)}
})
output_df.to_csv("forecast_24h.csv", index=False)
```

---

### 8. Output Interpretation

**Predictions values:**
- Each value is normalized PV power [0, 1]
- To get actual kW: `actual_kW = value * 82.41`
- h1 = 1 hour ahead, h24 = 24 hours ahead

**Metrics interpretation:**
- MASE < 1.0 = Better than naive 24h persistence baseline
- RMSE in kW = Lower is better
- Performance degrades with horizon (h=1 best, h=24 worst)
