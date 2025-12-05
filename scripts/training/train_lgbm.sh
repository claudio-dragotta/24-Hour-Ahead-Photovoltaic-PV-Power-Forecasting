#!/bin/bash
# Train LightGBM multi-horizon models with future meteo features (simulated NWP)

echo "Training LightGBM with Future Meteo"
echo "===================================="

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

python scripts/training/train_lgbm.py \
  --processed-path outputs/processed.parquet \
  --outdir outputs/lgbm \
  --use-future-meteo
