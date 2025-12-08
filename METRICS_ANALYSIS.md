# PV Forecasting Metrics Analysis & Benchmarks

## Metriche Attuali Utilizzate

### 1. RMSE (Root Mean Squared Error)
- **Formula**: `√(Σ(y_true - y_pred)²/n)`
- **Pro**: Penalizza errori grandi
- **Contro**: Sensibile agli outlier, scala dipendente
- **Risultato LightGBM**: 6.97 kW (medio)

### 2. MASE (Mean Absolute Scaled Error)
- **Formula**: `MAE(model) / MAE(naive_24h_persistence)`
- **Interpretazione**:
  - MASE < 1.0 = Meglio del baseline naive (OK)
  - MASE = 1.0 = Uguale al baseline
  - MASE > 1.0 = Peggio del baseline (PROBLEMA)
- **Risultato LightGBM**: 0.87 (medio)

---

## PROBLEMA CRITICO IDENTIFICATO - LightGBM

### Performance per Orizzonte Temporale

**Orizzonti Brevi (1-7h): ECCELLENTE**
- h=1: MASE=0.36 (65% skill improvement)
- h=2-7: MASE=0.52-0.82

**Orizzonti Medi (8-17h): BUONO**
- h=8-17: MASE=0.82-0.99

**Orizzonti Lunghi (18-24h): PROBLEMATICO**
- **h=18: MASE=1.046** (peggio del naive!)
- **h=20: MASE=1.030** (peggio del naive!)
- **h=23: MASE=1.059** (peggio del naive!)
- **h=24: MASE=1.063** (peggio del naive!)

### Perché Fallisce agli Orizzonti Lunghi?

1. **Previsioni meteo degradano** a 18-24h di distanza
2. **LightGBM non cattura dipendenze temporali lunghe** (è basato su alberi, non su sequenze)
3. **Overfitting sui pattern a breve termine**
4. **Mancanza di memoria temporale** (a differenza di LSTM/Transformer)

---

## Metriche da Aggiungere (TODO)

### 1. MAE (Mean Absolute Error)
- **Formula**: `Σ|y_true - y_pred|/n`
- **Perché**: Più interpretabile di RMSE, robusta agli outlier
- **Uso**: Confronto MAE vs RMSE rivela presenza di outlier

### 2. nRMSE (Normalized RMSE)
- **Formula**: `RMSE / (y_max - y_min)` oppure `RMSE / y_mean`
- **CRITICO**: Permette confronti tra impianti di diverse capacità
- **Benchmark**:
  - nRMSE < 10% = Ottimo
  - nRMSE 10-20% = Buono
  - nRMSE > 20% = Migliorabile

### 3. R² (Coefficient of Determination)
- **Formula**: `1 - SS_residual/SS_total`
- **Interpretazione**: % di varianza spiegata (0-1, dove 1=perfetto)

### 4. Forecast Skill Score
- **Formula**: `(1 - RMSE_model/RMSE_naive) × 100%`
- **Attuale**: 30.3% skill improvement su baseline

### 5. Bias (Mean Error)
- **Formula**: `Σ(y_pred - y_true)/n`
- **Interpretazione**:
  - Bias > 0 = Sovrastima sistematica
  - Bias < 0 = Sottostima sistematica
- **Importante per**: Grid management e bilanciamento

### 6. Metriche Filtrate per Ore Diurne
- **PROBLEMA ATTUALE**: Include ore notturne (pv=0) che distorcono le metriche
- **SOLUZIONE**: Filtrare per GHI > 0 o altitudine solare > 0°
- **Impatto**: Metriche più realistiche e rappresentative della difficoltà reale

---

## Benchmark Realistici (dalla letteratura)

### Target MASE
- **Eccellente**: < 0.6 (top 5% modelli)
- **Molto buono**: 0.6-0.8
- **Accettabile**: 0.8-1.0
- **Problematico**: > 1.0 (naive è meglio)

### Target per Orizzonti
- **h=1-12**: MASE < 0.70 (raggiungibile)
- **h=13-24**: MASE < 0.90 (sfidante)
- **Media globale**: MASE < 0.80

### Requisito Minimo
- **TUTTI gli orizzonti**: MASE < 1.0
  - Attualmente fallisce su h=18, 20, 23, 24

---

## Piano d'Azione per CNN-BiLSTM e TFT

### Ipotesi da Verificare
1. **CNN-BiLSTM**: Dovrebbe performare meglio agli orizzonti lunghi grazie alla memoria LSTM
2. **TFT (Temporal Fusion Transformer)**: Dovrebbe eccellere su tutti gli orizzonti grazie all'attention mechanism
3. **Ensemble**: Combinare i 3 modelli con pesi per orizzonte
   - LightGBM per h=1-12 (dove eccelle)
   - CNN-BiLSTM/TFT per h=13-24 (memoria temporale)

### Metriche da Confrontare
Quando testi CNN-BiLSTM e TFT, verifica:
- [ ] MASE < 1.0 su TUTTI gli orizzonti (minimo indispensabile)
- [ ] MASE medio < 0.80 (target)
- [ ] Performance orizzonti 18-24h (dove LightGBM fallisce)
- [ ] Confronto skill score per orizzonte
- [ ] Trade-off accuratezza vs tempo di training

### Domande Chiave
- CNN-BiLSTM risolve il problema degli orizzonti lunghi?
- TFT è significativamente migliore (giustifica la complessità)?
- L'ensemble migliora su tutti gli orizzonti?

---

## Note Implementative

### File da Modificare
- `pv_forecasting/metrics.py` - Aggiungere nRMSE, R², Bias
- `scripts/training/train_*.py` - Aggiungere nuove metriche al reporting
- Creare script per filtrare ore diurne

### Visualizzazioni Utili
- Plot MASE vs horizon per confronto modelli
- Heatmap errori per ora del giorno
- Scatter plot y_true vs y_pred per identificare bias
- Residual plot per outlier detection

---

**Data analisi**: 2025-12-04
**Modello analizzato**: LightGBM multi-horizon
**Dataset**: outputs_lgbm/
