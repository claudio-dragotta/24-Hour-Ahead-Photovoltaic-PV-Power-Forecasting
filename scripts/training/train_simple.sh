#!/usr/bin/env bash
# Training script using simplified preprocessing with global scaling AND non-overlapping windows
#
# CRITICAL DIFFERENCES from previous training:
# 1. Uses pv_wx_simple_scaled_full_compat.parquet with:
#    - NO real lag features (pv_lag1 = pv at same timestep)
#    - NO rolling features
#    - PV normalized to [0,1] by dividing by max capacity (68.92 kW)
#    - ALL features globally MinMax scaled together (preserves PV-weather relationships)
#
# 2. window-step=24: Non-overlapping windows (fixes data leakage)
#    - Previous: 17,497 samples with stride=1 (massive overlap → leakage)
#    - Now: ~730 samples with stride=24 (independent samples → better generalization)
#
# 3. Expected improvement: Much lower MASE because:
#    - No temporal leakage from lag features
#    - No data leakage from overlapping windows
#    - Preserved PV-weather relationships (global scaling)
#    - Independent train/val samples (true generalization test)
#
# This matches the exact approach used in high-performing reference models.

python scripts/training/train_multi_branch.py \
  --processed-path data/processed/merged/pv_wx_simple_scaled_full_compat.parquet \
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
  --outdir outputs/multi_branch/simple_scaled_step24
