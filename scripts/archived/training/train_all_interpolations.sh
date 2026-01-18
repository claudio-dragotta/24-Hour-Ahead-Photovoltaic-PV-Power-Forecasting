#!/bin/bash
# Script per addestrare il MultiBranchTransformer con tutte le tecniche di interpolazione

set -e

for interp in pooling adaptive classic; do
  echo ""
  echo "==============================="
  echo "Training con temporal_compression: $interp"
  echo "==============================="
  ts=$(date +%Y%m%d_%H%M%S)
  outdir="outputs/multi_branch/${interp}_${ts}"
  .venv/bin/python scripts/training/train_multi_branch.py \
    --processed-path data/processed/merged/pv_wx_combined.parquet \
    --outdir "$outdir" \
    --temporal-compression $interp \
    --forecast-noise-std 0.05
done

echo "\nTutti gli esperimenti sono stati lanciati. Controlla outputs/multi_branch/ per i risultati."
