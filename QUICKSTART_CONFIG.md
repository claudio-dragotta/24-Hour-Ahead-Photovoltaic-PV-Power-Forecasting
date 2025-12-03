# Configuration System - Quick Start Guide

## ✅ Sistema Config Completato!

Il sistema di configurazione YAML è ora **perfettamente implementato** e pronto all'uso.

---

## 🚀 Come Usarlo

### 1. **Training con Config YAML**

```bash
# Metodo raccomandato: usa file YAML
python -m pv_forecasting.cli train cnn --config configs/cnn_bilstm.yaml

# Override parametri specifici
python -m pv_forecasting.cli train cnn \
    --config configs/cnn_bilstm.yaml \
    --epochs 200 \
    --learning-rate 0.0001

# Usa defaults
python -m pv_forecasting.cli train cnn --horizon 48
```

### 2. **Training Programmatico (Python)**

```python
from pv_forecasting.config import Config
from pv_forecasting.training import train_cnn_bilstm
from pathlib import Path

# Carica config da YAML
config = Config.from_yaml(Path("configs/cnn_bilstm.yaml"))

# Modifica parametri se necessario
config.model.epochs = 150
config.model.learning_rate = 0.0005
config.output.output_dir = Path("outputs/experiment_42")

# Avvia training
train_cnn_bilstm(config)
```

### 3. **Crea Nuove Configurazioni**

```yaml
# configs/my_experiment.yaml
data:
  pv_path: "data/raw/pv_dataset.xlsx"
  wx_path: "data/raw/wx_dataset.xlsx"
  local_tz: "Australia/Sydney"
  lag_hours: [1, 24, 168]      # 1h, 1day, 1week
  rolling_hours: [3, 6, 12]    # Aggiungi 12h window
  include_solar: true
  include_clearsky: true

model:
  model_type: "cnn_bilstm"
  horizon: 24
  seq_len: 336                 # 2 settimane invece di 1
  epochs: 150
  batch_size: 32               # Ridotto per più dati
  learning_rate: 0.0001        # Più conservativo
  train_ratio: 0.8             # 80% training
  val_ratio: 0.1
  early_stopping_patience: 20

output:
  output_dir: "outputs/exp_2weeks_lookback"
  save_predictions: true
  save_model: true
  save_history: true
  log_level: "INFO"
```

```bash
python -m pv_forecasting.cli train cnn --config configs/my_experiment.yaml
```

---

## 📊 Output del Training

Quando avvii il training, il sistema crea automaticamente:

```
outputs/cnn_bilstm/
├── processed.parquet          # Dati processati
├── model_best.keras           # Modello salvato (best validation)
├── scalers.joblib             # StandardScaler fitted
├── history.json               # Training history (loss curves)
├── metrics.json               # Evaluation metrics
├── predictions_test.csv       # Predizioni sul test set
└── config_used.yaml           # Config usato (per riproducibilità)
```

---

## 🎯 Funzionalità Implementate

### ✅ **Completate**
1. Config management con dataclasses
2. Caricamento YAML con validazione
3. Training CNN-BiLSTM con config
4. Logging strutturato
5. Metriche automatiche (RMSE, MASE)
6. Salvataggio modello/scaler
7. Export predizioni CSV/JSON
8. CLI interface funzionante

### ⚠️ **Da Completare** (Opzionale)
- [ ] Integrazione LightGBM con config
- [ ] Integrazione TFT con config
- [ ] Predict module con config
- [ ] Walk-forward validation

---

## 🔍 Verifica Installazione

```bash
# Test config loading
python3 -c "
from pv_forecasting.config import Config
from pathlib import Path
config = Config.from_yaml(Path('configs/cnn_bilstm.yaml'))
print(f'Horizon: {config.model.horizon}h')
print(f'Epochs: {config.model.epochs}')
print('✅ Config system OK!')
"

# Test CLI
python3 -m pv_forecasting.cli --help
```

---

## 📝 Git Commits

```bash
git log --oneline -2
# 206ea58 Complete configuration system implementation
# 4349184 Refactor project structure and add configuration management system

# Push manuale (se necessario)
git push origin main
```

---

## 💡 Tips & Best Practices

1. **Versiona i Config**: Ogni esperimento = 1 YAML file
2. **Nomina Chiaramente**: `exp001_baseline.yaml`, `exp002_long_lookback.yaml`
3. **Salva Config Usato**: Sempre salvato in output_dir per riproducibilità
4. **Override Solo Necessario**: Usa YAML per defaults, CLI per quick tests
5. **Log Level**: `INFO` per produzione, `DEBUG` per troubleshooting

---

## 🎓 Esempio Completo

```bash
# 1. Crea experiment config
cat > configs/exp_baseline.yaml << 'YAML'
model:
  horizon: 24
  seq_len: 168
  epochs: 100
  batch_size: 64
output:
  output_dir: "outputs/baseline"
YAML

# 2. Avvia training
python -m pv_forecasting.cli train cnn --config configs/exp_baseline.yaml

# 3. Verifica risultati
ls outputs/baseline/
cat outputs/baseline/metrics.json

# 4. Commit risultati
git add configs/exp_baseline.yaml
git commit -m "Add baseline experiment configuration"
```

**Sistema Config = Completo e Pronto all'Uso! 🎉**
