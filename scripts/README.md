# Scripts Directory

This directory contains all executable scripts for the PV forecasting project, organized by purpose.

## Directory Structure

```
scripts/
├── data/           # Data processing and preparation
├── training/       # Model training scripts
├── inference/      # Prediction and deployment
└── evaluation/     # Model evaluation and ensembling
```

---

## 📊 Data Processing (`data/`)

### `preprocess_data.py`
Complete data preprocessing pipeline that generates the final training-ready dataset.

**What it does:**
- Loads raw PV and weather Excel files
- Merges datasets with timezone-aware UTC alignment
- Applies all feature engineering (45 features total)
- Encodes weather_description (0-10 ordinal scale)
- Fills clearness index (kc) NaN values with 0
- Validates data quality and alignment
- Saves to `outputs/processed.parquet`

**Usage:**
```bash
python scripts/data/preprocess_data.py
```

**Output:**
- `outputs/processed.parquet` (17,374 samples, 45 features, 0% NaN)

---

## 🎓 Model Training (`training/`)

Contains all model training scripts. See [training/README.md](training/README.md) for detailed documentation.

### `train_lgbm.py`
LightGBM multi-horizon forecasting (24 separate models, one per hour).

**Usage:**
```bash
python scripts/training/train_lgbm.py \
  --pv-path data/raw/pv_dataset.xlsx \
  --wx-path data/raw/wx_dataset.xlsx \
  --outdir outputs_lgbm
```

**Features:**
- 1200 trees, learning_rate=0.02
- 60/20/20 train/val/test split
- Early stopping on validation set
- Outputs feature importance ranking

---

### `train_tft.py`
Temporal Fusion Transformer (state-of-the-art attention-based model).

**Usage:**
```bash
python scripts/training/train_tft.py \
  --pv-path data/raw/pv_dataset.xlsx \
  --wx-path data/raw/wx_dataset.xlsx \
  --outdir outputs_tft \
  --epochs 100
```

**Features:**
- 100 epochs with early stopping
- Quantile loss (0.1, 0.5, 0.9)
- Variable selection network
- Attention mechanism

---

### `train_cnn_bilstm.py`
CNN-BiLSTM hybrid deep learning model (baseline).

**Usage:**
```bash
python scripts/training/train_cnn_bilstm.py \
  --pv-path data/raw/pv_dataset.xlsx \
  --wx-path data/raw/wx_dataset.xlsx \
  --outdir outputs_cnn \
  --epochs 100
```

**Features:**
- Conv1D + BiLSTM architecture
- 100 epochs with early stopping
- Dropout regularization (0.2-0.4)
- Learning rate: 5e-4

---

## 🔮 Inference (`inference/`)

### `predict.py`
Production inference script for generating 24-hour ahead forecasts.

**Usage:**
```bash
python scripts/inference/predict.py \
  --pv-path data/raw/pv_dataset.xlsx \
  --wx-path data/raw/wx_dataset.xlsx \
  --lgbm-dir outputs_lgbm \
  --tft-ckpt outputs_tft/tft-best.ckpt \
  --ensemble-weights outputs_ensemble/ensemble_weights.json
```

**Features:**
- Auto-detects available models
- Supports ensemble predictions
- Handles missing future weather gracefully

---

## 📈 Evaluation (`evaluation/`)

### `ensemble.py`
Ensemble optimization that combines predictions from multiple models.

**Usage:**
```bash
python scripts/evaluation/ensemble.py \
  --tft outputs_tft/predictions_test_tft.csv \
  --lgbm outputs_lgbm/predictions_test_lgbm.csv \
  --outdir outputs_ensemble
```

**Features:**
- Optimizes ensemble weights on validation set
- Grid search or Optuna optimization
- Evaluates on test set
- Saves optimal weights and blended predictions

---

## 🚀 Quick Start Workflow

1. **Preprocess data:**
   ```bash
   python scripts/data/preprocess_data.py
   ```

2. **Train models:**
   ```bash
   python scripts/training/train_lgbm.py --outdir outputs_lgbm
   python scripts/training/train_tft.py --outdir outputs_tft --epochs 100
   ```

3. **Create ensemble:**
   ```bash
   python scripts/evaluation/ensemble.py \
     --tft outputs_tft/predictions_test_tft.csv \
     --lgbm outputs_lgbm/predictions_test_lgbm.csv \
     --outdir outputs_ensemble
   ```

4. **Make predictions:**
   ```bash
   python scripts/inference/predict.py \
     --pv-path data/raw/pv_dataset.xlsx \
     --wx-path data/raw/wx_dataset.xlsx
   ```

---

## 📝 Notes

- All scripts automatically use `outputs/processed.parquet` if it exists (10x faster than re-processing)
- All scripts use 60/20/20 train/val/test chronological split
- All scripts output predictions in consistent CSV format with UTC timestamps
- Training time estimates: LightGBM ~30-60min, TFT ~2-4h (GPU), CNN-BiLSTM ~1-2h
