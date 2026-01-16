#!/bin/bash
# Create ensemble predictions from all trained models

echo "============================================================"
echo "CREATING ENSEMBLE PREDICTIONS"
echo "============================================================"
echo ""

source .venv/bin/activate

# Lista di tutti i modelli da combinare nell'ensemble
MODELS=(
  "outputs/multi_branch/WITH_TEMPORAL"
  "outputs/multi_branch/NO_TEMPORAL"
  "outputs/multi_branch/WITH_TEMPORAL_step8"
  "outputs/multi_branch/step4"
  "outputs/multi_branch/large_model"
  "outputs/multi_branch/step8_lowlr"
)

# Filtra solo modelli con predictions disponibili
AVAILABLE_MODELS=()
for model in "${MODELS[@]}"; do
  if [ -f "$model/test_eval/test_predictions.csv" ]; then
    AVAILABLE_MODELS+=("$model")
  fi
done

echo "Found ${#AVAILABLE_MODELS[@]} models with test predictions:"
for model in "${AVAILABLE_MODELS[@]}"; do
  echo "  - $model"
done

echo ""
echo "Creating ensemble..."
echo ""

python scripts/ensemble/create_ensemble.py \
  --model-dirs "${AVAILABLE_MODELS[@]}" \
  --test-dir "test_eval" \
  --outdir "outputs/ensemble"

echo ""
echo "Done! Check outputs/ensemble/ for results."
