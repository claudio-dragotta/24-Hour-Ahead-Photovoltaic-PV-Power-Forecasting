# Workflow: Allenarsi su Colab da VS Code

Promemoria rapido per usare il runtime GPU di Colab via SSH/VS Code e salvare gli output nella cartella locale `aooutopus`.

## 0) Prerequisiti locali (già configurati)
- Binario `cloudflared` installato in `/home/claudio/.local/bin/cloudflared`.
- Config SSH in `~/.ssh/config` con:
  ```
  Host *.trycloudflare.com
    User root
    Port 22
    ProxyCommand /home/claudio/.local/bin/cloudflared access ssh --hostname %h
  ```
  Se Colab genera un nuovo host, aggiungi un blocco `Host <nome>` che punti a `<nome>.trycloudflare.com` con lo stesso `ProxyCommand`.

## 1) Avviare Colab con GPU e SSH
- In Colab: Runtime → Change runtime type → GPU → Save.
- In una cella Colab esegui:
  ```python
  !pip install --quiet colab_ssh
  from colab_ssh import launch_ssh_cloudflared
  launch_ssh_cloudflared(password="scegli-una-password-robusta")
  ```
  Copia la riga `ssh ...trycloudflare.com` che viene stampata (es. `horizon-pharmaceutical-cdt-picnic.trycloudflare.com`).

## 2) Connettersi da VS Code / terminale
- VS Code: Command Palette → “Remote-SSH: Connect to Host…” → incolla `ssh ...trycloudflare.com` (o usa l’alias aggiunto in `~/.ssh/config`).
- Terminale: `ssh <host-trycloudflare>` oppure l’alias `ssh <alias>`.
- Password: quella usata in `launch_ssh_cloudflared`.

## 3) Lanciare il training (es. TFT) su Colab
```bash
cd /content/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting
python scripts/training/train_tft.py --outdir /content/aooutopus
```
Tutti gli output finiscono in `/content/aooutopus`.

## 4) Copiare gli output sul PC locale
```bash
mkdir -p /home/claudio/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/aooutopus
rsync -av --progress <alias-host>:/content/aooutopus/ /home/claudio/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/aooutopus/
```
Sostituisci `<alias-host>` con l’alias/host di Colab (es. `horizon-pharmaceutical-cdt-picnic`).
Alternativa: `scp -r <alias-host>:/content/aooutopus/* /home/claudio/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/aooutopus/`.

## 5) Note utili
- Usa `tmux`/`screen` per tenere vivo il training se cade l’SSH.
- La VM Colab è effimera: quando scade, rilancia la cella SSH e aggiorna l’host in `~/.ssh/config` se cambia.
