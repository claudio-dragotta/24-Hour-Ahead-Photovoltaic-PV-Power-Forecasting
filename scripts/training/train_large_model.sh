#!/bin/bash
# Training con MODELLO PIÙ GRANDE
# d_model 768 (+50%), num_layers 4, dim_feedforward 2048

python scripts/training/train_multi_branch.py \
  --processed-path data/processed/merged/pv_wx_simple_scaled_FIXED.parquet \
  --seq-len 24 \
  --horizon 24 \
  --window-step 24 \
  --epochs 200 \
  --batch-size 16 \
  --d-model 768 \
  --num-heads 8 \
  --num-layers 4 \
  --dim-feedforward 2048 \
  --dropout 0.1 \
  --gradient-clip-val 1.0 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --early-stopping-patience 15 \
  --scaler-type minmax \
  --target-scaler-type minmax \
  --sigmoid-output \
  --outdir outputs/multi_branch/large_model
