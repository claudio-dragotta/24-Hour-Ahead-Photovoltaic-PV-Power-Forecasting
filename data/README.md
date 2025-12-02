# Data Directory

## Structure

```
data/
├── raw/              # Original datasets (do not modify)
│   ├── pv_dataset.xlsx      # PV power production data
│   └── wx_dataset.xlsx      # Weather data
└── processed/        # Processed and merged datasets (legacy CSV)
    └── merged_dataset.csv   # Merged PV + Weather data (use outputs/processed.parquet instead)
```

## Raw Data

- **pv_dataset.xlsx**: Contains photovoltaic power production measurements
  - Two sheets: '07-10--06-11' and '07-11--06-12'
  - Columns: timestamp, pv (kW)
  - Timezone: Australia/Sydney (local time)

- **wx_dataset.xlsx**: Contains weather observations
  - Two sheets: '07-10--06-11' and '07-11--06-12'
  - Columns: dt_iso, lat, lon, temp, dew_point, pressure, humidity, wind_speed, wind_deg, rain_1h, clouds_all, Dhi, Dni, Ghi
  - Timezone: UTC+10:00

## Processed Data

- **outputs/processed.parquet**: Unified dataset with PV and weather data aligned (preferred format)
  - 17,542 rows with complete data
  - Weather columns standardized to lowercase (`ghi`, `dni`, `dhi`, `temp`, `humidity`, `clouds`, `wind_speed`)
  - Rolling features follow the `roll{w}h` naming (e.g., `pv_roll3h`)
  - Preserves millisecond precision in timestamps
  - All timestamps in UTC
- **merged_dataset.csv** (legacy): Kept for backward compatibility; regenerated via `scripts/merge_datasets.py`

## Usage

To regenerate the merged dataset:
```bash
python scripts/merge_datasets.py
```
