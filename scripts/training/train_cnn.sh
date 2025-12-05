#!/bin/bash
# Train CNN-BiLSTM model with future meteo features (simulated NWP)

echo "Training CNN-BiLSTM with Future Meteo"
echo "======================================"

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

python scripts/training/train_cnn_bilstm.py \
  --processed-path outputs/processed.parquet \
  --outdir outputs/cnn \
  --use-future-meteo
