# 24-Hour Ahead Photovoltaic (PV) Power Forecasting

[![CI](https://github.com/Claude-debug/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/workflows/CI/badge.svg)](https://github.com/Claude-debug/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/actions)
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

---

## Table of Contents

1. [Introduction](#introduction)
   - [Key Features](#key-features)
   - [Dataset Overview](#dataset-overview)
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
│   ├── training/                  # Model training
│   │   ├── README.md              # Training guide
│   │   ├── train_lgbm.py          # LightGBM multi-horizon training
│   │   ├── train_tft.py           # TFT training
│   │   └── train_cnn_bilstm.py    # CNN-BiLSTM training
│   ├── inference/                 # Prediction/deployment
│   │   └── predict.py             # Inference script
│   └── evaluation/                # Model evaluation
│       └── ensemble.py            # Ensemble optimization
│
├── outputs/                       # Model outputs (generated)
├── outputs_lgbm/                  # LightGBM outputs (generated)
├── outputs_tft/                   # TFT outputs (generated)
└── outputs_ensemble/              # Ensemble outputs (generated)
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
   - `pv_forecasting/pipeline.py` crea feature (cicliche, lag 1/24/168, rolling 3/6h, solar/clear‑sky) e salva `outputs/processed.parquet`.
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

### Ensemble e Inference

- Valida il peso dell'ensemble:
  ```bash
  python scripts/ensemble.py --tft-preds outputs_tft/predictions_val_tft.csv --lgbm-preds outputs_lgbm/predictions_val_lgbm.csv
  ```
- Previsione day-ahead (auto-switch con/senza meteo future se forniti):
  ```bash
  python scripts/inference/predict.py --pv-path data/raw/pv_dataset.xlsx --wx-path data/raw/wx_dataset.xlsx --future-wx-path <opzionale>
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
**Repository**: [24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting](https://github.com/Claude-debug/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting)
