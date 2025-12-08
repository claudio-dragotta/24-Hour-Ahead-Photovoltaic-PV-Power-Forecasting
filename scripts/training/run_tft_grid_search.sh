#!/bin/bash
# TFT Grid Search - Automatic hyperparameter optimization

cd /home/claudio/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting
source .venv/bin/activate

# Run grid search (will take ~80 hours for all 243 combinations)
# Each run takes ~20 minutes, so 243 * 20min = 81 hours
nohup python scripts/training/tft_grid_search.py \
  --output-dir outputs_grid_search_tft \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  > grid_search_tft.log 2>&1 &

echo "Grid search started!"
echo "Monitor progress: tail -f grid_search_tft.log"
echo "Results: outputs_grid_search_tft/grid_search_results.csv"
