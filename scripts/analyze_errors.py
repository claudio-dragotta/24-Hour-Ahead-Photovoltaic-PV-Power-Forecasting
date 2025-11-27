from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pv_forecasting.metrics import rmse, mase


def parse_args():
    ap = argparse.ArgumentParser(description="Error analysis for PV forecasting")
    ap.add_argument("--processed", type=str, default="outputs/processed.parquet")
    ap.add_argument("--predictions", type=str, default="outputs/predictions_test.csv")
    ap.add_argument("--season-boundary-hemisphere", type=str, default="north", choices=["north", "south"])
    return ap.parse_args()


def season_of(ts: pd.Timestamp, hemisphere: str = "north") -> str:
    m = ts.month
    if hemisphere == "north":
        return ["winter","winter","spring","spring","spring","summer","summer","summer","autumn","autumn","autumn","winter"][m-1]
    else:
        return ["summer","summer","autumn","autumn","autumn","winter","winter","winter","spring","spring","spring","summer"][m-1]


def main():
    args = parse_args()
    df = pd.read_parquet(args.processed)
    df.index = pd.to_datetime(df.index, utc=True)

    preds = pd.read_csv(args.predictions)
    preds["forecast_timestamp_utc"] = pd.to_datetime(preds["forecast_timestamp_utc"], utc=True)

    # Join true PV and weather context
    ctx_cols = [c for c in df.columns if c != "pv"]
    ctx = df[ctx_cols].copy()
    ctx["pv_true"] = df["pv"]

    merged = preds.merge(
        ctx, left_on="forecast_timestamp_utc", right_index=True, how="left"
    )

    # Sanity: y_true should equal pv_true
    if "y_true" in merged and "pv_true" in merged:
        mism = np.nanmean(np.abs(merged["y_true"].values - merged["pv_true"].values))
        print({"y_true_vs_pv_true_mae": float(mism)})

    # Compute naive baseline for each forecast timestamp: pv at -24h
    merged["pv_naive"] = df["pv"].shift(24).reindex(merged["forecast_timestamp_utc"]).values

    # Grouping flags
    ghi_col = "GHI" if "GHI" in merged.columns else ("ghi" if "ghi" in merged.columns else None)
    clouds_col = next((c for c in ["clouds_all","cloud_cover"] if c in merged.columns), None)
    wind_col = next((c for c in ["wind_speed","wind"] if c in merged.columns), None)

    merged["season"] = [season_of(ts, args.season_boundary_hemisphere) for ts in merged["forecast_timestamp_utc"]]
    if ghi_col:
        q = merged[ghi_col].quantile(0.75)
        merged["ghi_high"] = merged[ghi_col] >= q
    if clouds_col:
        merged["sunny"] = merged[clouds_col] <= merged[clouds_col].quantile(0.25)
    if wind_col:
        # 24h variability: std of past 24h at forecast timestamp
        wind_24h_std = df[wind_col].rolling(24).std()
        merged["wind_var_high"] = wind_24h_std.reindex(merged["forecast_timestamp_utc"]).values >= wind_24h_std.quantile(0.75)

    # MASE denominator from train segment (first 70%)
    train_cut = int(len(df) * 0.7)
    train_series = df["pv"].values[:train_cut]

    def summarize(group: pd.DataFrame, name: str):
        # drop NaNs
        g = group.dropna(subset=["y_true","y_pred","pv_naive"]) 
        if len(g) == 0:
            return None
        y_true = g["y_true"].values
        y_pred = g["y_pred"].values
        y_naive = g["pv_naive"].values
        return {
            "group": name,
            "n": int(len(g)),
            "rmse_model": rmse(y_true, y_pred),
            "rmse_naive": rmse(y_true, y_naive),
            "mase_model": mase(y_true, y_pred, train_series, m=24),
            "mase_naive": mase(y_true, y_naive, train_series, m=24),
        }

    results = []
    # Seasons
    for s in merged["season"].dropna().unique():
        res = summarize(merged[merged["season"] == s], f"season={s}")
        if res: results.append(res)

    # Sunny vs cloudy
    if "sunny" in merged:
        res = summarize(merged[merged["sunny"] == True], "sunny")
        if res: results.append(res)
        res = summarize(merged[merged["sunny"] == False], "cloudy")
        if res: results.append(res)

    # High GHI
    if "ghi_high" in merged:
        res = summarize(merged[merged["ghi_high"] == True], "ghi_high")
        if res: results.append(res)
        res = summarize(merged[merged["ghi_high"] == False], "ghi_low")
        if res: results.append(res)

    # Wind variability
    if "wind_var_high" in merged:
        res = summarize(merged[merged["wind_var_high"] == True], "wind_var_high")
        if res: results.append(res)
        res = summarize(merged[merged["wind_var_high"] == False], "wind_var_low")
        if res: results.append(res)

    # Overall
    res = summarize(merged, "overall")
    if res: results.append(res)

    out = pd.DataFrame(results)
    out_path = Path("outputs") / "error_analysis.csv"
    out.to_csv(out_path, index=False)
    print(out.sort_values("mase_model"))


if __name__ == "__main__":
    main()

