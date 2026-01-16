#!/usr/bin/env bash
# Training script with FIXED preprocessing (PV NOT scaled with weather features)
#
# CRITICAL FIX:
# - PV is already normalized by max capacity (68.92 kW)
# - PV is NOT scaled again with weather features
# - Only weather features are MinMax scaled
# - This prevents NaN predictions when test PV > training PV
#
# Benefits:
# - Correct handling of different PV capacities in test
# - No more PV being affected by weather feature scaling
# - Better generalization to new systems

python scripts/training/train_multi_branch.py \
  --processed-path data/processed/merged/pv_wx_simple_scaled_FIXED.parquet \
  --seq-len 24 \
  --horizon 24 \
  --window-step 24 \
  --epochs 200 \
  --batch-size 16 \
  --d-model 512 \
  --num-heads 8 \
  --num-layers 3 \
  --dim-feedforward 1024 \
  --dropout 0.1 \
  --gradient-clip-val 1.0 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --early-stopping-patience 15 \
  --scaler-type minmax \
  --target-scaler-type minmax \
  --sigmoid-output \
  --outdir outputs/multi_branch/simple_scaled_FIXED
