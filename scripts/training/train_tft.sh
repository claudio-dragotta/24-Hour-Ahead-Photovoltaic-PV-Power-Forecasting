#!/bin/bash
# Train Temporal Fusion Transformer

echo "Training TFT"
echo "============"

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

python scripts/training/train_tft.py \
  --processed-path outputs/processed.parquet \
  --outdir outputs/tft
