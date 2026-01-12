#!/bin/bash
# ARCHIVE: run_tft_grid_search.sh
# Original: scripts/training/run_tft_grid_search.sh

cd /home/claudio/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting
source .venv/bin/activate

nohup python scripts/training/tft_grid_search.py \
  --output-dir outputs_grid_search_tft \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  > grid_search_tft.log 2>&1 &

echo "Grid search started!"
echo "Monitor progress: tail -f grid_search_tft.log"
echo "Results: outputs_grid_search_tft/grid_search_results.csv"
