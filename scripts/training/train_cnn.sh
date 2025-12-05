#!/bin/bash
# Train CNN-BiLSTM model

echo "Training CNN-BiLSTM"
echo "==================="

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

python scripts/training/train_cnn_bilstm.py \
  --processed-path outputs/processed.parquet \
  --outdir outputs/cnn
