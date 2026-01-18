#!/bin/bash
# Optimized training script for Multi-Branch Transformer
# These hyperparameters are tuned for best performance on PV forecasting tasks

python scripts/training/train_multi_branch.py \
  --processed-path data/processed/merged/pv_wx_combined.parquet \
  --seq-len 24 \
  --horizon 24 \
  --epochs 200 \
  --batch-size 16 \
  --d-model 512 \
  --num-heads 8 \
  --num-layers 3 \
  --dim-feedforward 1024 \
  --dropout 0.1 \
  --gradient-clip-val 1.0 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --early-stopping-patience 15 \
  --scaler-type minmax \
  --target-scaler-type minmax \
  --sigmoid-output \
  --temporal-compression pooling \
  --forecast-noise-std 0.05 \
  --outdir outputs/multi_branch/optimized_config

echo "Training completed with optimized hyperparameters"
echo "Results saved to: outputs/multi_branch/optimized_config"
