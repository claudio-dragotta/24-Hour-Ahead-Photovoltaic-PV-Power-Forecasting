#!/bin/bash
# Train LightGBM with LAG72 features (includes 3-day lag)

echo "Training LightGBM with LAG72 features"
echo "====================================="

python3 scripts/training/train_lgbm.py \
  --processed-path outputs_lag72/processed.parquet \
  --outdir outputs_lag72/lgbm
