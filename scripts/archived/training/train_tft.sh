#!/bin/bash
# Train Temporal Fusion Transformer with future meteo features (simulated NWP)

echo "Training TFT with Future Meteo"
echo "=============================="

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

python scripts/training/train_tft.py \
  --processed-path outputs/processed.parquet \
  --outdir outputs/tft \
  --use-future-meteo
