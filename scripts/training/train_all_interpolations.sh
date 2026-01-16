#!/bin/bash
# Script per addestrare il MultiBranchTransformer con tutte le tecniche di interpolazione

set -e

for interp in pooling adaptive classic; do
  echo "\n==============================="
  echo "Training con temporal_compression: $interp"
  echo "===============================\n"
  python -m pv_forecasting.training.train_multi_branch \
    --config configs/multi_branch.yaml \
    --temporal_compression $interp \
    --output_dir outputs/multi_branch/$interp

done

echo "\nTutti gli esperimenti sono stati lanciati. Controlla outputs/multi_branch/ per i risultati."
