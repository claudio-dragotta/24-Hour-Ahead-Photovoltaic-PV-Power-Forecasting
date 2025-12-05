#!/bin/bash
# Train LightGBM multi-horizon models

echo "Training LightGBM"
echo "================="

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

python scripts/training/train_lgbm.py \
  --processed-path outputs/processed.parquet \
  --outdir outputs/lgbm
