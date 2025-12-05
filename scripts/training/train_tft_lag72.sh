#!/bin/bash
# Train TFT with LAG72 features (includes 3-day lag)

echo "Training TFT with LAG72 features"
echo "================================"

python3 scripts/training/train_tft.py \
  --processed-path outputs_lag72/processed.parquet \
  --outdir outputs_lag72/tft
