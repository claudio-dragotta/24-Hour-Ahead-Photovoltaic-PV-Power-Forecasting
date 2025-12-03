# Training Scripts

This directory contains all training scripts for the PV forecasting models.

## Quick Start

### 1. Train LightGBM (Fast, Strong Baseline)

```bash
python training_scripts/train_lgbm.py --outdir outputs_lgbm
```

**Time:** ~5-10 minutes
**Output:** `outputs_lgbm/`

---

### 2. Train Temporal Fusion Transformer (State-of-the-Art)

```bash
python training_scripts/train_tft.py --outdir outputs_tft
```

**Time:** ~30-60 minutes (GPU recommended)
**Output:** `outputs_tft/`

---

### 3. Train CNN-BiLSTM (Original Baseline)

```bash
python training_scripts/train_cnn_bilstm.py --outdir outputs_cnn
```

**Time:** ~15-30 minutes
**Output:** `outputs_cnn/`

---

## Create Ensemble

After training multiple models, combine them for best performance:

```bash
python scripts/ensemble.py \
  --tft outputs_tft/predictions_test_tft.csv \
  --lgbm outputs_lgbm/predictions_test_lgbm.csv \
  --outdir outputs_ensemble
```

---

## Scripts Overview

| Script | Model | Description |
|--------|-------|-------------|
| `train_lgbm.py` | LightGBM | Multi-horizon gradient boosting (fastest, strong baseline) |
| `train_tft.py` | Temporal Fusion Transformer | Attention-based deep learning (state-of-the-art) |
| `train_cnn_bilstm.py` | CNN-BiLSTM | Hybrid convolutional-recurrent (original baseline) |
| `predict.py` | All models | Inference script for generating forecasts |

---

## Common Arguments

All training scripts support:

- `--outdir` - Output directory (default: `outputs_<model>`)
- `--epochs` - Maximum training epochs
- `--batch-size` - Training batch size
- `--seed` - Random seed for reproducibility

### Model-Specific Arguments

**LightGBM:**
- `--walk-forward-folds` - Number of walk-forward validation folds

**TFT:**
- `--hidden-size` - TFT hidden layer size
- `--attention-heads` - Number of attention heads
- `--use-future-meteo` - Use future weather as known covariates

**CNN-BiLSTM:**
- `--filters` - Number of CNN filters
- `--lstm-units` - Number of LSTM units

---

## Pre-processed Data Cache

All scripts automatically use `outputs/processed.parquet` if available (faster).

To regenerate the cache:
```bash
rm outputs/processed.parquet
python training_scripts/train_lgbm.py  # Will recreate it
```

---

## For More Details

See the main documentation:
- [TFT_ENSEMBLE_GUIDE.md](../TFT_ENSEMBLE_GUIDE.md) - Complete TFT and ensemble guide
- [QUICKSTART_CONFIG.md](../QUICKSTART_CONFIG.md) - Configuration system guide
- [README.md](../README.md) - Project overview
