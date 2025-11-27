# 24-Hour Ahead Photovoltaic (PV) Power Forecasting

## Introduction

Accurate forecasting of photovoltaic (PV) power generation is crucial for efficient grid management, energy storage optimization, and renewable energy integration. This project implements a deep learning-based approach for 24-hour ahead PV power forecasting using historical production data and meteorological observations.

The system leverages a hybrid CNN-BiLSTM architecture to capture both spatial patterns and temporal dependencies in the data. By combining convolutional layers for feature extraction with bidirectional LSTM networks for sequence modeling, the model achieves robust predictions across varying weather conditions and seasonal patterns.

### Key Features

- **Multi-step forecasting**: Predicts power output for the next 24 hours in a single forward pass
- **Robust time handling**: Comprehensive management of timezone conversions and daylight saving time transitions
- **Hybrid architecture**: Combines CNN feature extraction with BiLSTM temporal modeling
- **Feature engineering**: Incorporates cyclical time features, lag variables, and rolling statistics
- **Rigorous evaluation**: Uses MASE and RMSE metrics with naive baseline comparison

### Dataset Overview

The project utilizes two years (2010-2012) of data from a solar installation in Australia:

- **PV Production Data**: 17,542 hourly measurements with millisecond precision
- **Weather Data**: 14 meteorological variables including temperature, humidity, wind, and solar irradiance (GHI, DNI, DHI)
- **Location**: Australia (Sydney region, UTC+10:00 timezone)
- **Installed Capacity**: 82.41 kWp

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

---

## Project Structure

```
24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── train.py                   # Main training script
│
├── data/                      # Data directory
│   ├── README.md              # Data documentation
│   ├── raw/                   # Original datasets
│   │   ├── pv_dataset.xlsx    # PV production data
│   │   └── wx_dataset.xlsx    # Weather data
│   └── processed/             # Processed datasets
│       └── merged_dataset.csv # Merged and aligned data
│
├── scripts/                   # Utility scripts
│   ├── merge_datasets.py      # Dataset merging script
│   ├── verify_merge.py        # Data integrity verification
│   ├── check_merged.py        # Quick data check
│   └── analyze_errors.py      # Error analysis utilities
│
├── notebooks/                 # Jupyter notebooks
├── outputs/                   # Model outputs (generated)
└── pv_forecasting/            # Main Python package
    ├── __init__.py
    ├── data.py                # Data loading utilities
    ├── features.py            # Feature engineering
    ├── metrics.py             # Evaluation metrics
    ├── model.py               # Model architecture
    ├── timeutils.py           # Timezone handling
    └── window.py              # Sliding window creation
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

- **Total Records**: 17,542 hourly observations
- **Period**: June 30, 2010 - June 30, 2012 (2 years)
- **Completeness**: 99.98% (only 3 NaN values in PV)
- **Columns**: 14 (1 PV + 13 weather features)
- **Validation**: PV-GHI correlation = 0.905 (excellent physical alignment)

### Dataset Generation

Generate the merged dataset:

```bash
python scripts/merge_datasets.py
```

Verify data integrity:

```bash
python scripts/verify_merge.py
```

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

---

## Model Architecture

The forecasting model uses a **hybrid CNN-BiLSTM architecture**:

- **Input Layer**: 168 timesteps × n features
- **Conv1D Layers**: Extract local patterns and reduce dimensionality
- **Bidirectional LSTM**: Capture temporal dependencies in both directions
- **Dense Output**: 24 neurons (one per forecast hour)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with early stopping

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

---

## Usage

### Training the Model

Run training with default parameters:

```bash
python train.py
```

With custom parameters:

```bash
python train.py \
  --pv-path data/raw/pv_dataset.xlsx \
  --wx-path data/raw/wx_dataset.xlsx \
  --local-tz Australia/Sydney \
  --seq-len 168 \
  --horizon 24 \
  --epochs 50 \
  --batch-size 64
```

### Data Preparation

Generate the merged dataset:

```bash
python scripts/merge_datasets.py
```

Verify data integrity:

```bash
python scripts/verify_merge.py
```

---

## Outputs

After training, the following files are generated in the `outputs/` directory:

- `processed.parquet`: Merged, feature-engineered time series (UTC-indexed)
- `scalers.joblib`: Fitted feature scaler (StandardScaler)
- `model_best.keras`: Best model weights (lowest validation loss)
- `history.json`: Training and validation loss history
- `predictions_test.csv`: Long-format predictions for the test split

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

## Project Information

**Author**: Ludovico  
**Date**: November 2025  
**Institution**: Deep Learning Course - Magistrale  
**Repository**: [24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting](https://github.com/Claude-debug/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting)


