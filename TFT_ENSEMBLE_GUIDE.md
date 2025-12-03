# TFT & Ensemble Guide

Complete guide for training Temporal Fusion Transformer (TFT) models and creating ensemble predictions for optimal 24h-ahead PV forecasting.

## Overview

This project now supports three state-of-the-art forecasting models:

1. **Temporal Fusion Transformer (TFT)** - Attention-based deep learning with interpretability
2. **LightGBM** - Gradient boosting multi-horizon model
3. **CNN-BiLSTM** - Hybrid convolutional-recurrent architecture

The **ensemble module** combines these models with optimized weights to achieve the best possible forecast accuracy.

---

## 1. Training the TFT Model

### Quick Start

```bash
# Standard mode (physics + time features only)
python training_scripts/train_tft.py --outdir outputs_tft

# With future meteorological data (NWP mode)
python training_scripts/train_tft.py --outdir outputs_tft --use-future-meteo

# Custom hyperparameters
python training_scripts/train_tft.py \
  --outdir outputs_tft \
  --epochs 100 \
  --batch-size 128 \
  --hidden-size 128 \
  --attention-heads 8 \
  --dropout 0.3 \
  --learning-rate 0.0005
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--processed-path` | `outputs/processed.parquet` | Path to pre-processed data (faster) |
| `--pv-path` | `data/raw/pv_dataset.xlsx` | Fallback: PV dataset |
| `--wx-path` | `data/raw/wx_dataset.xlsx` | Fallback: Weather dataset |
| `--seq-len` | `168` | Encoder length (hours) - 7 days |
| `--horizon` | `24` | Prediction horizon (hours) |
| `--epochs` | `50` | Maximum training epochs |
| `--batch-size` | `64` | Training batch size |
| `--hidden-size` | `64` | TFT hidden layer size |
| `--attention-heads` | `4` | Number of attention heads |
| `--dropout` | `0.2` | Dropout rate |
| `--learning-rate` | `0.001` | Initial learning rate |
| `--val-ratio` | `0.2` | Validation set ratio |
| `--use-future-meteo` | `False` | Use future weather as known covariates |
| `--outdir` | `outputs_tft` | Output directory |

### What Gets Saved

After training, the TFT script saves:

```
outputs_tft/
├── tft-best.ckpt              # Best model checkpoint
├── predictions_val_tft.csv    # Long-format predictions with timestamps
├── metrics_val_tft.json       # Per-horizon metrics (RMSE, MASE)
├── metrics_summary.json       # Overall average metrics
└── config_tft.json            # Training configuration
```

### TFT Features

The TFT model automatically uses:

**Known Future Covariates** (always available):
- Cyclical time features: `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`
- Solar position: `sp_zenith`, `sp_azimuth`
- Clear-sky estimates: `cs_ghi`, `cs_dni`, `cs_dhi`

**Past Unknown Covariates**:
- Historical lags: `pv_lag_*`, `ghi_lag_*`, `dni_lag_*`, `dhi_lag_*`
- Rolling features: `*_roll3h`, `*_roll6h`, `*_roll12h`, `*_roll24h`
- Clearness index: `kc`
- Raw meteorological variables (if `--use-future-meteo` not set)

---

## 2. Creating an Ensemble

The ensemble module finds optimal weights to combine predictions from multiple models.

### Quick Start

```bash
# Ensemble TFT + LightGBM (grid search)
python scripts/ensemble.py \
  --tft outputs_tft/predictions_val_tft.csv \
  --lgbm outputs_lgbm/predictions_val_lgbm.csv \
  --outdir outputs_ensemble

# Add CNN-BiLSTM to the ensemble
python scripts/ensemble.py \
  --tft outputs_tft/predictions_val_tft.csv \
  --lgbm outputs_lgbm/predictions_val_lgbm.csv \
  --bilstm outputs/predictions_val_cnn_bilstm.csv \
  --outdir outputs_ensemble

# Use Optuna optimization (more sophisticated)
python scripts/ensemble.py \
  --tft outputs_tft/predictions_val_tft.csv \
  --lgbm outputs_lgbm/predictions_val_lgbm.csv \
  --method optuna \
  --optuna-trials 200 \
  --metric rmse
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tft` | `None` | Path to TFT predictions CSV |
| `--lgbm` | `None` | Path to LightGBM predictions CSV |
| `--bilstm` | `None` | Path to CNN-BiLSTM predictions CSV |
| `--outdir` | `outputs_ensemble` | Output directory |
| `--method` | `grid` | Optimization method: `grid` or `optuna` |
| `--grid-steps` | `11` | Grid search steps (11 = 0.0, 0.1, ..., 1.0) |
| `--optuna-trials` | `100` | Number of Optuna trials |
| `--metric` | `rmse` | Metric to optimize: `rmse`, `mae` |

### What Gets Saved

```
outputs_ensemble/
├── predictions_val_ensemble.csv    # Predictions with all models + ensemble
├── ensemble_weights.json           # Optimal weights and metadata
└── metrics_ensemble.json           # Comparison of all models
```

### Example Output

**ensemble_weights.json**:
```json
{
  "model_names": ["TFT", "LightGBM"],
  "weights": [0.6, 0.4],
  "optimization_method": "grid",
  "optimization_metric": "rmse",
  "best_rmse": 0.0234
}
```

**metrics_ensemble.json**:
```json
{
  "individual_models": {
    "TFT": {"rmse": 0.0245, "mae": 0.0189},
    "LightGBM": {"rmse": 0.0251, "mae": 0.0192}
  },
  "ensemble": {
    "rmse": 0.0234,
    "mae": 0.0182
  }
}
```

---

## 3. Complete Workflow

### Step 1: Train All Models

```bash
# 1. Train LightGBM (already done in your case)
python training_scripts/train_lgbm.py --outdir outputs_lgbm

# 2. Train TFT
python training_scripts/train_tft.py --outdir outputs_tft

# 3. (Optional) Train CNN-BiLSTM
python training_scripts/train_cnn_bilstm.py --outdir outputs_cnn
```

### Step 2: Create Ensemble

```bash
# Ensemble with grid search
python scripts/ensemble.py \
  --tft outputs_tft/predictions_val_tft.csv \
  --lgbm outputs_lgbm/predictions_val_lgbm.csv \
  --method grid \
  --grid-steps 21
```

### Step 3: Analyze Results

Check the metrics comparison:
```bash
cat outputs_ensemble/metrics_ensemble.json | python -m json.tool
```

The ensemble should show **lower RMSE** than individual models!

---

## 4. Advanced Usage

### Using Optuna for Optimal Weights

Optuna uses Bayesian optimization to find better weight combinations:

```bash
python scripts/ensemble.py \
  --tft outputs_tft/predictions_val_tft.csv \
  --lgbm outputs_lgbm/predictions_val_lgbm.csv \
  --method optuna \
  --optuna-trials 500 \
  --metric rmse
```

**When to use Optuna:**
- 3+ models in ensemble (grid search becomes slow)
- Want more sophisticated optimization
- Have time for longer search

**When to use Grid Search:**
- 2 models only (fast enough)
- Want guaranteed global optimum
- Interpretable weight combinations (0.0, 0.1, 0.2, ...)

### TFT with Future Weather Forecasts

If you have NWP (Numerical Weather Prediction) forecasts:

```bash
python training_scripts/train_tft.py \
  --outdir outputs_tft_nwp \
  --use-future-meteo
```

This treats raw weather variables (`ghi`, `dni`, `dhi`, `temp`, etc.) as **known future** covariates instead of past unknown.

---

## 5. Performance Tips

### TFT Training Speed

- **Use GPU**: TFT automatically detects CUDA and uses GPU
- **Pre-process data**: Use `outputs/processed.parquet` (already cached from lgbm_train.py)
- **Batch size**: Increase if you have enough GPU memory (128, 256)
- **Reduce epochs**: Early stopping usually kicks in around epoch 20-30

### Memory Usage

If you run out of GPU memory:
```bash
python tft_train.py --batch-size 32 --hidden-size 32
```

### Ensemble Optimization

For 3 models with fine grid search:
```bash
# This searches 10^3 = 1000 combinations (slow but thorough)
python scripts/ensemble.py \
  --tft outputs_tft/predictions_val_tft.csv \
  --lgbm outputs_lgbm/predictions_val_lgbm.csv \
  --bilstm outputs/predictions_val_cnn_bilstm.csv \
  --method optuna \
  --optuna-trials 500
```

---

## 6. Troubleshooting

### TFT Training Issues

**Error: CUDA out of memory**
```bash
python training_scripts/train_tft.py --batch-size 32 --hidden-size 32
```

**Error: processed.parquet not found**
```bash
# Run train_lgbm first to generate it, or TFT will create it automatically
python training_scripts/train_lgbm.py
```

### Ensemble Issues

**Error: missing required columns**

Ensure your prediction CSVs have:
- `origin_timestamp_utc`
- `forecast_timestamp_utc`
- `horizon_h`
- `y_true`
- `y_pred`

**Optuna not installed**
```bash
pip install optuna  # or use grid search instead
```

---

## 7. Expected Performance

Based on typical PV forecasting benchmarks:

| Model | Typical RMSE | Training Time |
|-------|--------------|---------------|
| Naive (24h persistence) | 0.10-0.15 | N/A |
| LightGBM | 0.025-0.035 | 5-10 min |
| CNN-BiLSTM | 0.030-0.040 | 15-30 min |
| TFT | 0.023-0.033 | 30-60 min |
| **Ensemble** | **0.020-0.030** | **+1 min** |

The ensemble typically achieves **5-15% lower error** than the best individual model.

---

## 8. Next Steps

1. **Hyperparameter tuning**: Use Optuna to optimize TFT hyperparameters
2. **Feature engineering**: Add more physics-informed features
3. **Probabilistic forecasts**: TFT quantile loss provides uncertainty estimates
4. **Production deployment**: Load `tft-best.ckpt` for real-time inference

For more details on the configuration system, see [QUICKSTART_CONFIG.md](QUICKSTART_CONFIG.md).
