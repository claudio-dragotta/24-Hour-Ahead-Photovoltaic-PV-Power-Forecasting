#!/bin/bash
# Training BEST COMBINATION: step=8 + modello grande
# Più dati + più capacità del modello

python scripts/training/train_multi_branch.py \
  --processed-path data/processed/merged/pv_wx_simple_scaled_FIXED.parquet \
  --seq-len 24 \
  --horizon 24 \
  --window-step 8 \
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
  --outdir outputs/multi_branch/step8_large
