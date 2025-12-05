from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from pv_forecasting.data import align_hourly, load_pv_xlsx, load_wx_xlsx
from pv_forecasting.features import (
    add_clearsky,
    add_kc,
    add_lags,
    add_rollings_h,
    add_solar_position,
    add_time_cyclical,
    standardize_feature_columns,
)
from pv_forecasting.pipeline import DEFAULT_LAG_HOURS, DEFAULT_ROLLING_HOURS


def parse_args():
    ap = argparse.ArgumentParser(description="Generate 24h-ahead predictions with LightGBM/TFT ensemble")
    ap.add_argument("--pv-path", type=str, default="data/raw/pv_dataset.xlsx")
    ap.add_argument("--wx-path", type=str, default="data/raw/wx_dataset.xlsx")
    ap.add_argument(
        "--future-wx-path", type=str, default=None, help="Optional future weather forecast file (csv/parquet/xlsx)"
    )
    ap.add_argument("--local-tz", type=str, default="Australia/Sydney")
    ap.add_argument("--lgbm-dir", type=str, default="outputs_lgbm")
    ap.add_argument("--tft-ckpt", type=str, default="outputs_tft/tft-best.ckpt")
    ap.add_argument("--ensemble-weights", type=str, default="outputs_ensemble/ensemble_weights.json")
    ap.add_argument("--outdir", type=str, default="outputs_predictions")
    ap.add_argument("--horizon", type=int, default=None, help="Override horizon (defaults to training config)")
    ap.add_argument("--disable-tft", action="store_true", help="Skip TFT inference even if checkpoint is present")
    return ap.parse_args()


def load_config(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def load_future_weather(path: Path) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    ts_col = df.columns[0]
    idx = pd.to_datetime(df[ts_col], utc=True)
    feat_df = df.drop(columns=[ts_col])
    for c in feat_df.columns:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
    feat_df.index = idx
    feat_df = standardize_feature_columns(feat_df)
    return feat_df.sort_index()


def build_feature_frame(
    pv_path: Path,
    wx_path: Path,
    future_wx_path: Path | None,
    local_tz: str,
    lag_hours: List[int],
    rolling_hours: List[int],
) -> pd.DataFrame:
    pv = load_pv_xlsx(pv_path, local_tz)
    wx_hist = load_wx_xlsx(wx_path)
    if future_wx_path and future_wx_path.exists():
        wx_future = load_future_weather(future_wx_path)
        wx = pd.concat([wx_hist, wx_future]).sort_index()
    else:
        wx = wx_hist

    df = align_hourly(pv, wx, keep_wx_future=True)
    df = standardize_feature_columns(df)
    df = add_time_cyclical(df)
    df = add_solar_position(df)
    df = add_clearsky(df)
    if "ghi" in df.columns:
        df = add_kc(df, ghi_col="ghi")

    lag_cols = [c for c in ["pv", "ghi", "dni", "dhi"] if c in df.columns]
    if lag_cols:
        df = add_lags(df, lag_cols, lags=lag_hours)
    roll_cols = [c for c in ["pv", "ghi", "dni"] if c in df.columns]
    if roll_cols:
        df = add_rollings_h(df, roll_cols, windows=rolling_hours)

    df = df.sort_index()
    df = df.copy()
    df["time_idx"] = np.arange(len(df))
    df["series_id"] = "pv_site_1"
    return df


def add_future_covariates(df: pd.DataFrame, future_cols: List[str], horizon: int) -> pd.DataFrame:
    out = df.copy()
    for h in range(1, horizon + 1):
        for c in future_cols:
            if c in out.columns:
                out[f"{c}_fut_h{h}"] = out[c].shift(-h)
    return out


def predict_lightgbm(
    df: pd.DataFrame,
    feature_cols: List[str],
    models_dir: Path,
    horizon: int,
) -> pd.DataFrame:
    hist_df = df[df["pv"].notna()]
    if hist_df.empty:
        raise ValueError("No historical PV data available for LightGBM inference.")
    row = hist_df.iloc[[-1]].copy()
    for c in feature_cols:
        if c not in row.columns:
            row[c] = np.nan
    X = row[feature_cols]

    preds = []
    last_ts = hist_df.index[-1]
    for h in range(1, horizon + 1):
        model_path = models_dir / f"lgbm_h{h}.joblib"
        if not model_path.exists():
            continue
        model = joblib_load(model_path)
        pred = float(model.predict(X)[0])
        preds.append(
            {
                "forecast_timestamp_utc": (last_ts + pd.Timedelta(hours=h)).isoformat(),
                "horizon_h": h,
                "y_pred_lgbm": pred,
            }
        )
    return pd.DataFrame(preds)


def predict_tft(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    known_future: List[str],
    past_unknown: List[str],
    ckpt_path: Path,
) -> pd.DataFrame:
    import torch
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import GroupNormalizer
    from pytorch_forecasting.models import TemporalFusionTransformer

    if not ckpt_path.exists():
        return pd.DataFrame()

    hist_df = df[df["pv"].notna()].copy()
    if hist_df.empty:
        return pd.DataFrame()

    training_ds = TimeSeriesDataSet(
        hist_df,
        time_idx="time_idx",
        target="pv",
        group_ids=["series_id"],
        max_encoder_length=seq_len,
        max_prediction_length=horizon,
        time_varying_known_reals=sorted(set(known_future)),
        time_varying_unknown_reals=["pv"] + sorted(set(past_unknown)),
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation="standard"),
        allow_missing_timesteps=True,
    )

    min_pred_idx = int(hist_df["time_idx"].max() - seq_len + 1)
    predict_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        df,
        predict=True,
        stop_randomization=True,
        min_prediction_idx=max(min_pred_idx, 0),
    )
    predict_loader = predict_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location=device)

    preds, x = model.predict(predict_loader, return_x=True)
    preds_np = preds.detach().cpu().numpy()
    decoder_time_idx = x["decoder_time_idx"].detach().cpu().numpy()
    encoder_time_idx = x["encoder_time_idx"].detach().cpu().numpy()

    time_lookup: Dict[int, pd.Timestamp] = {int(ti): ts for ti, ts in zip(df["time_idx"].astype(int), df.index)}
    future_start = int(hist_df["time_idx"].max()) + 1

    rows = []
    for i in range(preds_np.shape[0]):
        for h_idx in range(preds_np.shape[1]):
            forecast_ti = int(decoder_time_idx[i, h_idx])
            if forecast_ti < future_start:
                continue
            horizon_h = h_idx + 1
            forecast_ts = time_lookup.get(forecast_ti)
            origin_ti = int(encoder_time_idx[i, -1])
            origin_ts = time_lookup.get(origin_ti)
            rows.append(
                {
                    "time_idx": forecast_ti,
                    "origin_timestamp_utc": origin_ts.isoformat() if origin_ts is not None else None,
                    "forecast_timestamp_utc": forecast_ts.isoformat() if forecast_ts is not None else None,
                    "horizon_h": horizon_h,
                    "y_pred_tft": float(preds_np[i, h_idx]),
                }
            )

    pred_df = pd.DataFrame(rows)
    # Keep only first occurrence per horizon (latest origin)
    pred_df = pred_df.sort_values(["horizon_h", "time_idx"]).groupby("horizon_h", as_index=False).first()
    return pred_df


def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_lgbm = load_config(Path(args.lgbm_dir) / "config_lgbm.json")
    cfg_tft = load_config(Path(args.tft_ckpt).parent / "config_tft.json") if Path(args.tft_ckpt).exists() else {}
    horizon = args.horizon or cfg_lgbm.get("horizon") or cfg_tft.get("horizon", 24)
    lag_hours = [int(x) for x in cfg_lgbm.get("lag_hours", list(DEFAULT_LAG_HOURS))]
    rolling_hours = [int(x) for x in cfg_lgbm.get("rolling_hours", list(DEFAULT_ROLLING_HOURS))]
    use_future_meteo = bool(cfg_lgbm.get("use_future_meteo", False))
    future_cols = cfg_lgbm.get("future_cols", ["ghi", "dni", "dhi", "temp", "humidity", "clouds", "wind_speed"])

    feature_cols = cfg_lgbm.get("feature_cols")

    future_wx_path = Path(args.future_wx_path) if args.future_wx_path else None
    df = build_feature_frame(
        pv_path=Path(args.pv_path),
        wx_path=Path(args.wx_path),
        future_wx_path=future_wx_path,
        local_tz=args.local_tz,
        lag_hours=lag_hours,
        rolling_hours=rolling_hours,
    )

    if use_future_meteo:
        df = add_future_covariates(df, future_cols, horizon=horizon)
        for c in future_cols:
            if c not in df.columns:
                continue
            fallback = df[c].ffill().iloc[-1]
            for h in range(1, horizon + 1):
                col_name = f"{c}_fut_h{h}"
                if col_name in df.columns:
                    df[col_name] = df[col_name].fillna(fallback)

    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in {"pv", "series_id"}]

    preds_lgbm = predict_lightgbm(
        df, feature_cols=feature_cols, models_dir=Path(args.lgbm_dir) / "models", horizon=horizon
    )

    preds_tft = pd.DataFrame()
    if not args.disable_tft and Path(args.tft_ckpt).exists():
        known_future = cfg_tft.get(
            "known_future",
            [
                c
                for c in [
                    "hour_sin",
                    "hour_cos",
                    "doy_sin",
                    "doy_cos",
                    "sp_zenith",
                    "sp_azimuth",
                    "cs_ghi",
                    "cs_dni",
                    "cs_dhi",
                ]
                if c in df.columns
            ],
        )
        raw_unknown = [
            c
            for c in df.columns
            if c.startswith(("pv_lag", "ghi_lag", "dni_lag", "dhi_lag"))
            or c.endswith("roll3h")
            or c.endswith("roll6h")
            or c == "kc"
        ]
        past_unknown = cfg_tft.get("past_unknown", raw_unknown)
        preds_tft = predict_tft(
            df,
            seq_len=cfg_tft.get("seq_len", 168),
            horizon=horizon,
            known_future=known_future,
            past_unknown=past_unknown,
            ckpt_path=Path(args.tft_ckpt),
        )

    # Merge and ensemble
    merged = preds_lgbm
    if not preds_tft.empty:
        merged = pd.merge(merged, preds_tft, on=["forecast_timestamp_utc", "horizon_h"], how="left")

    weight_tft = 0.0
    if Path(args.ensemble_weights).exists() and not preds_tft.empty:
        try:
            wcfg = json.loads(Path(args.ensemble_weights).read_text())
            weight_tft = float(wcfg.get("weight_tft", 0.5))
        except Exception:
            weight_tft = 0.5
    elif not preds_tft.empty:
        weight_tft = 0.5

    if "y_pred_tft" in merged.columns and not merged["y_pred_tft"].isna().all():
        merged["y_pred_ensemble"] = (
            weight_tft * merged["y_pred_tft"].fillna(merged["y_pred_lgbm"]) + (1 - weight_tft) * merged["y_pred_lgbm"]
        )
    else:
        merged["y_pred_ensemble"] = merged["y_pred_lgbm"]
    merged.to_csv(out_dir / "predictions_next24.csv", index=False)

    print(f"Saved predictions to {out_dir / 'predictions_next24.csv'}")


if __name__ == "__main__":
    main()
