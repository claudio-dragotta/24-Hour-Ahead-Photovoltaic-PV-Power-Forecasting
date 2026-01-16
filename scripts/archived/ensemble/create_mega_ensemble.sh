#!/bin/bash
# MEGA ENSEMBLE: Tutti i modelli incluso augmented

echo "============================================================"
echo "MEGA ENSEMBLE - ALL MODELS INCLUDING AUGMENTED"
echo "============================================================"
echo ""

source .venv/bin/activate

# Prima testa il modello augmented se non già fatto
if [ ! -f "outputs/multi_branch/augmented/test_eval/test_metrics.json" ]; then
  echo "Testing augmented model first..."
  python scripts/evaluation/eval_preprocessed_test.py \
    --test-parquet data/test/processed/pv_wx_test_WITH_TEMPORAL.parquet \
    --ckpt outputs/multi_branch/augmented/model-best.ckpt \
    --seq-len 24 --horizon 24 --step 24 \
    --outdir outputs/multi_branch/augmented/test_eval
  echo ""
fi

# Lista TUTTI i modelli
ALL_MODELS=(
  "outputs/multi_branch/WITH_TEMPORAL"
  "outputs/multi_branch/NO_TEMPORAL"
  "outputs/multi_branch/WITH_TEMPORAL_step8"
  "outputs/multi_branch/step4"
  "outputs/multi_branch/large_model"
  "outputs/multi_branch/step8_lowlr"
  "outputs/multi_branch/augmented"
)

# Filtra solo modelli disponibili
AVAILABLE_MODELS=()
for model in "${ALL_MODELS[@]}"; do
  pred_file="$model/test_eval/test_predictions.csv"
  if [ -f "$pred_file" ]; then
    AVAILABLE_MODELS+=("$model")
  else
    echo "[SKIP] Skipping $model (no test predictions yet)"
  fi
done

echo ""
echo "Creating MEGA ensemble with ${#AVAILABLE_MODELS[@]} models:"
for model in "${AVAILABLE_MODELS[@]}"; do
  model_name=$(basename "$model")
  metrics_file="$model/test_eval/test_metrics.json"
  if [ -f "$metrics_file" ]; then
    mase=$(cat "$metrics_file" | grep -o '"mase": [0-9.]*' | cut -d' ' -f2)
    printf "  %-30s MASE: %s\n" "$model_name" "$mase"
  else
    echo "  - $model_name"
  fi
done

echo ""
echo "Creating ensemble..."
echo ""

python scripts/ensemble/create_ensemble.py \
  --model-dirs "${AVAILABLE_MODELS[@]}" \
  --test-dir "test_eval" \
  --outdir "outputs/mega_ensemble"

echo ""
echo "============================================================"
echo "CHECKING IF WE REACHED TARGET"
echo "============================================================"

STACKING_MASE=$(cat outputs/mega_ensemble/ensemble_stacking_metrics.json | grep -o '"mase": [0-9.]*' | cut -d' ' -f2)

echo ""
echo "Mega Ensemble MASE: $STACKING_MASE"
echo "Target MASE: 0.45"
echo ""

# Check if target reached
if [ $(echo "$STACKING_MASE < 0.45" | bc -l) -eq 1 ]; then
  echo "TARGET REACHED! MASE < 0.45"
  echo ""
  echo "Target MASE: 0.45"
  echo "Our MASE: $STACKING_MASE"
  echo ""
else
  GAP=$(echo "scale=4; $STACKING_MASE - 0.45" | bc)
  echo "Not quite there yet. Gap: +$GAP"
  echo ""
  echo "Next steps to try:"
  echo "  1. bash scripts/training/train_very_deep.sh"
  echo "  2. bash scripts/training/train_cosine_schedule.sh"
  echo "  3. Advanced feature engineering"
fi

echo "============================================================"
