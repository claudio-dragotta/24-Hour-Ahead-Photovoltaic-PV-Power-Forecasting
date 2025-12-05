#!/bin/bash
# Run ensemble optimization for PV forecasting models

echo "Running Ensemble Optimization"
echo "=============================="

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

# Default paths (adjust these to match your output directories)
LGBM_PRED="outputs/lgbm/predictions_test_lgbm.csv"
CNN_PRED="outputs/cnn/predictions_test_cnn.csv"
TFT_PRED="outputs/tft/predictions_val_tft.csv"

python scripts/evaluation/ensemble.py \
  --lgbm "$LGBM_PRED" \
  --cnn "$CNN_PRED" \
  --tft "$TFT_PRED" \
  --outdir outputs/ensemble \
  --method exhaustive \
  --exhaustive-min-models 2 \
  --exhaustive-max-models 3 \
  --metric rmse \
  --grid-steps 11

echo ""
echo "Ensemble optimization complete!"
echo "Results saved to: outputs/ensemble/"
