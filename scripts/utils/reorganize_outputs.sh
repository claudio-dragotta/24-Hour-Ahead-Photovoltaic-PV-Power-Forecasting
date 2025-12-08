#!/bin/bash
# Script per riorganizzare tutti i file di output in una struttura pulita
# Struttura: outputs/{model_type}/{variant}/

PROJECT_ROOT="/home/claudio/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting"
cd "$PROJECT_ROOT"

echo "🧹 Riorganizzazione directory outputs per modello..."
echo ""

# Crea struttura pulita per ogni modello
mkdir -p outputs/{tft,cnn,lgbm}/baseline

# 1. Organizza TFT
echo "📦 Organizzando TFT..."
if [ -d "outputs/models/baseline/tft" ]; then
    mv outputs/models/baseline/tft/* outputs/tft/baseline/ 2>/dev/null || true
fi
if [ -d "outputs_baseline/tft" ]; then
    mv outputs_baseline/tft/* outputs/tft/baseline/ 2>/dev/null || true
fi
if [ -d "outputs/experiments/tft_variants/tft_new" ]; then
    mkdir -p outputs/tft/hidden64_heads4
    mv outputs/experiments/tft_variants/tft_new/* outputs/tft/hidden64_heads4/ 2>/dev/null || true
fi
if [ -d "outputs/grid_search" ]; then
    mkdir -p outputs/tft/grid_search
    mv outputs/grid_search/* outputs/tft/grid_search/ 2>/dev/null || true
fi

# 2. Organizza CNN
echo "📦 Organizzando CNN..."
if [ -d "outputs/models/baseline/cnn" ]; then
    mv outputs/models/baseline/cnn/* outputs/cnn/baseline/ 2>/dev/null || true
fi
if [ -d "outputs_baseline/cnn" ]; then
    mv outputs_baseline/cnn/* outputs/cnn/baseline/ 2>/dev/null || true
fi
# CNN fusion attention (quello che ha fallito)
if [ -d "outputs_baseline/cnn_20251207_123319" ]; then
    mkdir -p outputs/cnn/fusion_attention
    mv outputs_baseline/cnn_20251207_123319/* outputs/cnn/fusion_attention/ 2>/dev/null || true
fi
# CNN lag72
if [ -d "outputs/experiments/lag72/cnn" ]; then
    mkdir -p outputs/cnn/lag72
    mv outputs/experiments/lag72/cnn/* outputs/cnn/lag72/ 2>/dev/null || true
fi
if [ -d "outputs_lag72/cnn" ]; then
    mkdir -p outputs/cnn/lag72
    mv outputs_lag72/cnn/* outputs/cnn/lag72/ 2>/dev/null || true
fi

# 3. Organizza LightGBM
echo "📦 Organizzando LightGBM..."
if [ -d "outputs/models/baseline/lgbm" ]; then
    mv outputs/models/baseline/lgbm/* outputs/lgbm/baseline/ 2>/dev/null || true
fi
if [ -d "outputs_baseline/lgbm" ]; then
    mv outputs_baseline/lgbm/* outputs/lgbm/baseline/ 2>/dev/null || true
fi
if [ -d "outputs/experiments/lag72/lgbm" ]; then
    mkdir -p outputs/lgbm/lag72
    mv outputs/experiments/lag72/lgbm/* outputs/lgbm/lag72/ 2>/dev/null || true
fi
if [ -d "outputs_lag72/lgbm" ]; then
    mkdir -p outputs/lgbm/lag72
    mv outputs_lag72/lgbm/* outputs/lgbm/lag72/ 2>/dev/null || true
fi

# 4. Mantieni processed.parquet e logs
echo "💾 Organizzando file comuni..."
mkdir -p outputs/logs
mv outputs/logs/training/* outputs/logs/ 2>/dev/null || true

# 5. Pulisci directory vecchie
echo "🧹 Pulizia directory vecchie..."
rm -rf outputs/models outputs/experiments outputs_baseline outputs_lag72 2>/dev/null || true
find outputs -type d -empty -delete 2>/dev/null || true

echo ""
echo "✅ Riorganizzazione completata!"
echo ""
echo "📁 Nuova struttura per modello:"
echo "outputs/"
echo "├── processed.parquet      # Dataset preprocessato (cache)"
echo "├── tft/"
echo "│   ├── baseline/          # TFT baseline (hidden=32, dropout=0.4, RMSE=3.7060)"
echo "│   ├── hidden64_heads4/   # TFT con hidden=64, heads=4 (failed, RMSE=5.07)"
echo "│   └── grid_search/       # Grid search 243 configs (Ray Tune)"
echo "├── cnn/"
echo "│   ├── baseline/          # CNN-BiLSTM baseline (RMSE=3.7267)"
echo "│   ├── fusion_attention/  # CNN 3-branch fusion (failed, RMSE=4.36)"
echo "│   └── lag72/             # CNN con lag features estesi"
echo "├── lgbm/"
echo "│   ├── baseline/          # LightGBM baseline (da fare)"
echo "│   └── lag72/             # LightGBM con lag features estesi"
echo "└── logs/"
echo "    └── *.log              # Training logs"
echo ""

# Mostra dimensioni per modello
echo "💾 Spazio utilizzato per modello:"
du -sh outputs/tft outputs/cnn outputs/lgbm outputs/processed.parquet 2>/dev/null | sort -h
echo ""
echo "Totale outputs: $(du -sh outputs 2>/dev/null | cut -f1)"
