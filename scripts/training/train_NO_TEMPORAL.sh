#!/usr/bin/env bash
# Training script WITHOUT temporal features
#
# Key change: NO temporal encoding (hour, day, weekend, holidays)
# Focus on pure physical/weather relationships for better generalization
#
# Benefits:
# - Avoids overfitting on time-specific patterns
# - Better generalization to unseen time periods
# - Simpler model, fewer parameters

python scripts/training/train_multi_branch.py \
  --processed-path data/processed/merged/pv_wx_simple_NO_TEMPORAL.parquet \
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
  --outdir outputs/multi_branch/NO_TEMPORAL
