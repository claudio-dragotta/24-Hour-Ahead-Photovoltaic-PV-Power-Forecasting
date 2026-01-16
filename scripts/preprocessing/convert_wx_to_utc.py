import pandas as pd
from dateutil import parser
import pytz

# Percorsi file
wx_path = "data/processed/merged/wx_combined.csv"
out_path = "data/processed/merged/wx_combined_utc.csv"

# Carica il file meteo
wx = pd.read_csv(wx_path)

# Converte dt_iso in UTC
# Gestisce sia +10:00 che +11:00
wx['dt_utc'] = wx['dt_iso'].apply(lambda x: parser.isoparse(x).astimezone(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'))

# Sposta la nuova colonna all'inizio
cols = wx.columns.tolist()
cols.insert(0, cols.pop(cols.index('dt_utc')))
wx = wx[cols]

# Salva il nuovo file
wx.to_csv(out_path, index=False)

print(f"File salvato: {out_path}")
