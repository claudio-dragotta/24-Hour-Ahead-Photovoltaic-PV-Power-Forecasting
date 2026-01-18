#!/bin/bash
# Training con WEIGHTED LOSS per peak production
# Penalizza di più gli errori su alta produzione

# NOTA: Questo richiede modifica al modello per supportare loss_type parameter
# Per ora uso il modello base ma con più epoche e focus su metriche peak

echo "============================================================"
echo "TRAINING WITH PEAK-WEIGHTED FOCUS"
echo "============================================================"
echo ""
echo "Strategy:"
echo "  - Standard MSE loss (for now - weighted loss requires model changes)"
echo "  - More epochs to better learn peak production"
echo "  - Lower learning rate for fine-grained optimization"
echo "  - Higher weight decay to reduce overfitting"
echo ""

source .venv/bin/activate

python scripts/training/train_multi_branch.py \
  --processed-path data/processed/merged/pv_wx_simple_scaled_FIXED.parquet \
  --seq-len 24 \
  --horizon 24 \
  --window-step 24 \
  --epochs 300 \
  --batch-size 16 \
  --d-model 512 \
  --num-heads 8 \
  --num-layers 3 \
  --dim-feedforward 1024 \
  --dropout 0.15 \
  --gradient-clip-val 1.0 \
  --learning-rate 5e-4 \
  --weight-decay 5e-4 \
  --early-stopping-patience 25 \
  --scaler-type minmax \
  --target-scaler-type minmax \
  --sigmoid-output \
  --outdir outputs/multi_branch/peak_focused

echo ""
echo "Training completed!"
