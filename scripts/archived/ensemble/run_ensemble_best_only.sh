#!/bin/bash
# Create ensemble usando SOLO i migliori modelli (MASE < 0.75)

echo "============================================================"
echo "ENSEMBLE - BEST MODELS ONLY"
echo "============================================================"
echo ""

source .venv/bin/activate

# Solo i modelli con MASE < 0.75
BEST_MODELS=(
  "outputs/multi_branch/WITH_TEMPORAL"
  "outputs/multi_branch/WITH_TEMPORAL_step8"
  "outputs/multi_branch/large_model"
  "outputs/multi_branch/step8_lowlr"
)

echo "Using only best models (MASE < 0.75):"
for model in "${BEST_MODELS[@]}"; do
  echo "  - $(basename $model)"
done

echo ""
echo "Creating ensemble..."
echo ""

python scripts/ensemble/create_ensemble.py \
  --model-dirs "${BEST_MODELS[@]}" \
  --test-dir "test_eval" \
  --outdir "outputs/ensemble_best_only"

echo ""
echo "Done! Check outputs/ensemble_best_only/ for results."
