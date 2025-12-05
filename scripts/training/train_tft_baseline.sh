#!/bin/bash
# Train TFT with BASELINE features (no lag72)

echo "Training TFT with BASELINE features"
echo "===================================="

python3 scripts/training/train_tft.py \
  --processed-path outputs_baseline/processed.parquet \
  --outdir outputs_baseline/tft
