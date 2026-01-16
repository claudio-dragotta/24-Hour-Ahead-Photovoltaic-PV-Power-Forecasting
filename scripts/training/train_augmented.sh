#!/bin/bash
# Training con DATA AUGMENTATION (noise injection)

echo "============================================================"
echo "STEP 1: Creating augmented dataset"
echo "============================================================"

source .venv/bin/activate

# Create augmented data (2x samples with noise)
python scripts/preprocessing/augment_data.py \
  --input data/processed/merged/pv_wx_simple_scaled_FIXED.parquet \
  --output data/processed/merged/pv_wx_augmented.parquet \
  --noise-level 0.02 \
  --n-augmented 1

echo ""
echo "============================================================"
echo "STEP 2: Training on augmented data"
echo "============================================================"
echo ""

# Train on augmented data
python scripts/training/train_multi_branch.py \
  --processed-path data/processed/merged/pv_wx_augmented.parquet \
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
  --outdir outputs/multi_branch/augmented

echo ""
echo "Training completed! Test with:"
echo "  python scripts/evaluation/eval_preprocessed_test.py \\"
echo "    --test-parquet data/test/processed/pv_wx_test_WITH_TEMPORAL.parquet \\"
echo "    --ckpt outputs/multi_branch/augmented/model-best.ckpt \\"
echo "    --seq-len 24 --horizon 24 --step 24 \\"
echo "    --outdir outputs/multi_branch/augmented/test_eval"
