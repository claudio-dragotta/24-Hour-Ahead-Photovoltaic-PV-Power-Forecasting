#!/bin/bash
# Train CNN-BiLSTM with BASELINE features (no lag72)

echo "Training CNN-BiLSTM with BASELINE features"
echo "============================================"

python scripts/training/train_cnn_bilstm.py \
  --processed-path outputs_baseline/processed.parquet \
  --outdir outputs_baseline/cnn
