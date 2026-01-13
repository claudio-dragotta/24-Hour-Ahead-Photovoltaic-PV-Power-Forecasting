Dove sono eseguiti i processi (preprocess -> one-hot -> train)

- Caricamento e merge (raw PV + WX -> dataset unificato `wxpv`):
  - Entry point: `pv_forecasting.pipeline.load_and_engineer_features`
  - Funzioni coinvolte: `pv_forecasting.data.load_pv_xlsx`, `pv_forecasting.data.load_wx_xlsx`, `pv_forecasting.data.align_hourly`

- Feature engineering (time, lag, rolling, fisica):
  - Implementazione: `pv_forecasting.pipeline` e `pv_forecasting.features`
  - Esempi: `add_time_cyclical`, `add_lags`, `add_rollings_h`, `add_clearsky`, `add_kc`

- Encoding meteo (opzioni):
  - Ordinale (default): `pv_forecasting.features.encode_weather_description` — chiamata automaticamente da `load_and_engineer_features`.
  - One-hot (se richiesto): `pv_forecasting.features.encode_weather_onehot` — genera colonne `wx_*`; usata in script archiviati (`scripts/_archived/data/preprocess_csv.py`).

- Salvataggio dataset processato:
  - `pv_forecasting.pipeline.persist_processed` -> salva `outputs/processed.parquet` (wrapper su `pv_forecasting.data.save_processed`).

- Training (entry point e comportamento):
  - Script: `scripts/training/train_multi_branch.py`
  - Comportamento: carica `outputs/processed.parquet` se esiste; altrimenti chiama `load_and_engineer_features` e poi `persist_processed`.
  - Nota: lo script cerca colonne `wx_*` e le include se presenti.

- Script utili / wrapper:
  - `scripts/preprocessing/generate_processed_lag72.py` — genera `processed.parquet` con lag a 72h.
  - Copie archiviate del preprocessing completo: `scripts/_archived/data/preprocess_data.py`, `scripts/_archived/data/preprocess_csv.py`.

Suggerimento: per garantire coerenza, eseguire prima il preprocessing (o `generate_processed_lag72.py`) quindi lo script di training che caricherà `outputs/processed.parquet`.
