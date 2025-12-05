#!/bin/bash
# Train LightGBM with BASELINE features (no lag72)

echo "Training LightGBM with BASELINE features"
echo "========================================="

python3 scripts/training/train_lgbm.py \
  --processed-path outputs_baseline/processed.parquet \
  --outdir outputs_baseline/lgbm
