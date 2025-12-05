#!/bin/bash
# Train CNN-BiLSTM with LAG72 features (includes 3-day lag)

echo "Training CNN-BiLSTM with LAG72 features"
echo "========================================"

python scripts/training/train_cnn_bilstm.py \
  --processed-path outputs_lag72/processed.parquet \
  --outdir outputs_lag72/cnn
