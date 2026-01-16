"""Generate all visualization plots for the PV forecasting project.

Creates a comprehensive set of plots for analysis and presentation.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Output directory
PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# Load ensemble predictions
print("Loading ensemble predictions...")
pred_long = pd.read_csv("outputs/ensemble/ensemble_stacking.csv")
pred_long = pred_long.rename(columns={"y_true_kw": "actual", "y_pred_kw": "prediction"})

# Ensemble metrics
with open("outputs/ensemble/ensemble_stacking_metrics.json") as f:
    metrics = json.load(f)

print(f"Loaded {len(pred_long)} ensemble predictions")


def plot_1_actual_vs_predicted_scatter():
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(pred_long["actual"], pred_long["prediction"], alpha=0.1, s=5, c="steelblue")

    # Perfect prediction line
    max_val = max(pred_long["actual"].max(), pred_long["prediction"].max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Prediction")

    ax.set_xlabel("Actual Power (kW)", fontsize=12)
    ax.set_ylabel("Predicted Power (kW)", fontsize=12)
    ax.set_title("Actual vs Predicted PV Power\nEnsemble Stacking Model", fontsize=14)
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Add metrics text
    ax.text(
        0.05,
        0.95,
        f"RMSE: {metrics['rmse']:.2f} kW\nMASE: {metrics['mase']:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_actual_vs_predicted_scatter.png", dpi=150)
    plt.close()
    print("[OK] 01_actual_vs_predicted_scatter.png")


def plot_2_error_distribution():
    """Histogram of prediction errors."""
    errors = pred_long["prediction"] - pred_long["actual"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram only
    ax.hist(errors, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", lw=2)
    ax.axvline(errors.mean(), color="orange", linestyle="-", lw=2, label=f"Mean: {errors.mean():.2f}")
    ax.set_xlabel("Prediction Error (kW)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Prediction Errors", fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_error_distribution.png", dpi=150)
    plt.close()
    print("[OK] 02_error_distribution.png")


def plot_3_metrics_by_horizon():
    """Bar plot of RMSE and MASE by horizon."""
    horizon_metrics = pd.DataFrame(metrics["horizon_metrics"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 24))
    axes[0].bar(horizon_metrics["horizon"], horizon_metrics["rmse"], color=colors, edgecolor="black")
    axes[0].axhline(metrics["rmse"], color="red", linestyle="--", lw=2, label=f"Overall: {metrics['rmse']:.2f}")
    axes[0].set_xlabel("Forecast Horizon (hours)", fontsize=12)
    axes[0].set_ylabel("RMSE (kW)", fontsize=12)
    axes[0].set_title("RMSE by Forecast Horizon", fontsize=14)
    axes[0].legend()
    axes[0].set_xticks(range(1, 25, 2))

    # MASE
    axes[1].bar(horizon_metrics["horizon"], horizon_metrics["mase"], color=colors, edgecolor="black")
    axes[1].axhline(metrics["mase"], color="red", linestyle="--", lw=2, label=f"Overall: {metrics['mase']:.3f}")
    axes[1].axhline(1.0, color="gray", linestyle=":", lw=2, label="Naive Baseline")
    axes[1].set_xlabel("Forecast Horizon (hours)", fontsize=12)
    axes[1].set_ylabel("MASE", fontsize=12)
    axes[1].set_title("MASE by Forecast Horizon", fontsize=14)
    axes[1].legend()
    axes[1].set_xticks(range(1, 25, 2))

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_metrics_by_horizon.png", dpi=150)
    plt.close()
    print("[OK] 03_metrics_by_horizon.png")


def plot_4_sample_predictions():
    """Time series plot of sample predictions vs actual."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Select 3 random samples
    np.random.seed(2)
    samples = np.random.choice(pred_long["sample_idx"].unique(), 3, replace=False)

    for idx, (ax, sample_idx) in enumerate(zip(axes, samples)):
        sample_data = pred_long[pred_long["sample_idx"] == sample_idx].sort_values("horizon")

        ax.plot(sample_data["horizon"], sample_data["actual"], "b-o", label="Actual", markersize=6, linewidth=2)
        ax.plot(sample_data["horizon"], sample_data["prediction"], "r--s", label="Predicted", markersize=6, linewidth=2)

        # Fill between
        ax.fill_between(
            sample_data["horizon"], sample_data["actual"], sample_data["prediction"], alpha=0.2, color="gray"
        )

        ax.set_xlabel("Forecast Horizon (hours)", fontsize=11)
        ax.set_ylabel("Power (kW)", fontsize=11)
        ax.set_title(f"Sample {idx+1} (Index: {sample_idx})", fontsize=12)
        ax.legend(loc="upper right")
        ax.set_xticks(range(1, 25))
        ax.grid(True, alpha=0.3)

    fig.suptitle("24-Hour Ahead Predictions: Sample Forecasts", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 04_sample_predictions.png")


def plot_5_heatmap_errors_by_hour():
    """Heatmap of average errors by horizon and actual power level."""
    pred_copy = pred_long.copy()
    pred_copy["error"] = np.abs(pred_copy["prediction"] - pred_copy["actual"])

    # Bin actual values
    pred_copy["power_bin"] = pd.cut(
        pred_copy["actual"], bins=[0, 5, 15, 30, 50, 100], labels=["0-5", "5-15", "15-30", "30-50", "50+"]
    )

    # Pivot table
    pivot = pred_copy.pivot_table(values="error", index="power_bin", columns="horizon", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean Absolute Error (kW)"})
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Actual Power Level (kW)", fontsize=12)
    ax.set_title("Mean Absolute Error by Power Level and Horizon", fontsize=14)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_heatmap_errors.png", dpi=150)
    plt.close()
    print("[OK] 05_heatmap_errors.png")


def plot_6_residuals_analysis():
    """Residual analysis plots."""
    errors = pred_long["prediction"] - pred_long["actual"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Predicted
    axes[0, 0].scatter(pred_long["prediction"], errors, alpha=0.1, s=5, c="steelblue")
    axes[0, 0].axhline(0, color="red", linestyle="--", lw=2)
    axes[0, 0].set_xlabel("Predicted Power (kW)", fontsize=11)
    axes[0, 0].set_ylabel("Residual (kW)", fontsize=11)
    axes[0, 0].set_title("Residuals vs Predicted", fontsize=12)

    # Residuals vs Actual
    axes[0, 1].scatter(pred_long["actual"], errors, alpha=0.1, s=5, c="steelblue")
    axes[0, 1].axhline(0, color="red", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("Actual Power (kW)", fontsize=11)
    axes[0, 1].set_ylabel("Residual (kW)", fontsize=11)
    axes[0, 1].set_title("Residuals vs Actual", fontsize=12)

    # Q-Q plot
    from scipy import stats

    stats.probplot(errors, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normality Check)", fontsize=12)

    # Residuals autocorrelation (by sample)
    sample_errors = pred_long.groupby("sample_idx").apply(
        lambda x: x.sort_values("horizon")["prediction"].values - x.sort_values("horizon")["actual"].values
    )

    # Lag-1 autocorrelation
    autocorr = []
    for errs in sample_errors:
        if len(errs) > 1:
            autocorr.append(np.corrcoef(errs[:-1], errs[1:])[0, 1])

    axes[1, 1].hist(autocorr, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1, 1].axvline(np.mean(autocorr), color="red", linestyle="--", lw=2, label=f"Mean: {np.mean(autocorr):.3f}")
    axes[1, 1].set_xlabel("Lag-1 Autocorrelation", fontsize=11)
    axes[1, 1].set_ylabel("Frequency", fontsize=11)
    axes[1, 1].set_title("Residual Autocorrelation Distribution", fontsize=12)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_residuals_analysis.png", dpi=150)
    plt.close()
    print("[OK] 06_residuals_analysis.png")


def plot_7_seed_comparison():
    """Bar chart comparing seed search results."""
    with open("outputs/multi_branch/seed_search/seed_search_results.json") as f:
        seed_results = json.load(f)

    top_seeds = pd.DataFrame(seed_results["top_seeds"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top 10 seeds by MASE
    colors = ["gold" if s == 2 else "steelblue" for s in top_seeds["seed"]]
    axes[0].barh(range(10), top_seeds["mase"][:10], color=colors, edgecolor="black")
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels([f"Seed {s}" for s in top_seeds["seed"][:10]])
    axes[0].set_xlabel("MASE", fontsize=12)
    axes[0].set_title("Top 10 Seeds by MASE (Lower is Better)", fontsize=14)
    axes[0].invert_yaxis()

    # Add value labels
    for i, v in enumerate(top_seeds["mase"][:10]):
        axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=10)

    # Distribution of all seeds
    all_results = pd.DataFrame(seed_results["results"])
    axes[1].hist(all_results["mase"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].axvline(
        seed_results["stats"]["best_mase"],
        color="gold",
        linestyle="-",
        lw=3,
        label=f"Best (Seed 2): {seed_results['stats']['best_mase']:.4f}",
    )
    axes[1].axvline(
        seed_results["stats"]["mean_mase"],
        color="red",
        linestyle="--",
        lw=2,
        label=f"Mean: {seed_results['stats']['mean_mase']:.4f}",
    )
    axes[1].set_xlabel("MASE", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("MASE Distribution Across 100 Seeds", fontsize=14)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_seed_comparison.png", dpi=150)
    plt.close()
    print("[OK] 07_seed_comparison.png")


def plot_8_daily_pattern():
    """Average predictions by hour of day pattern."""
    # Group by horizon to get average daily pattern
    avg_by_horizon = pred_long.groupby("horizon").agg({"actual": "mean", "prediction": "mean"}).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(avg_by_horizon["horizon"], avg_by_horizon["actual"], "b-o", label="Actual (avg)", markersize=8, linewidth=2)
    ax.plot(
        avg_by_horizon["horizon"],
        avg_by_horizon["prediction"],
        "r--s",
        label="Predicted (avg)",
        markersize=8,
        linewidth=2,
    )

    ax.fill_between(
        avg_by_horizon["horizon"], avg_by_horizon["actual"], avg_by_horizon["prediction"], alpha=0.2, color="gray"
    )

    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Average Power (kW)", fontsize=12)
    ax.set_title("Average Daily Pattern: Actual vs Predicted", fontsize=14)
    ax.legend()
    ax.set_xticks(range(1, 25))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "08_daily_pattern.png", dpi=150)
    plt.close()
    print("[OK] 08_daily_pattern.png")


def plot_9_error_percentiles():
    """Error percentiles by horizon."""
    pred_copy = pred_long.copy()
    pred_copy["abs_error"] = np.abs(pred_copy["prediction"] - pred_copy["actual"])

    percentiles = pred_copy.groupby("horizon")["abs_error"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    fig, ax = plt.subplots(figsize=(14, 6))

    horizons = range(1, 25)
    ax.fill_between(
        horizons, percentiles["25%"], percentiles["75%"], alpha=0.3, color="steelblue", label="25th-75th percentile"
    )
    ax.fill_between(
        horizons, percentiles["5%"], percentiles["95%"], alpha=0.15, color="steelblue", label="5th-95th percentile"
    )
    ax.plot(horizons, percentiles["50%"], "b-o", label="Median", linewidth=2, markersize=6)
    ax.plot(horizons, percentiles["mean"], "r--", label="Mean", linewidth=2)

    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Absolute Error (kW)", fontsize=12)
    ax.set_title("Error Percentiles by Horizon", fontsize=14)
    ax.legend()
    ax.set_xticks(range(1, 25))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "09_error_percentiles.png", dpi=150)
    plt.close()
    print("[OK] 09_error_percentiles.png")


def plot_10_summary_dashboard():
    """Summary dashboard with key metrics."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    text = f"""
    Ensemble Stacking Model
    ════════════════════════

    Overall Metrics:
    ─────────────────
    RMSE:  {metrics['rmse']:.3f} kW
    MASE:  {metrics['mase']:.3f}
    R2:    {metrics['r2']:.3f}
    MBE:   {metrics['mbe']:.3f}

    (Per-horizon metrics not disponibili)
    """
    ax.text(
        0.1,
        0.9,
        text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle("PV Power Forecasting - Ensemble Model Summary", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(PLOT_DIR / "10_summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 10_summary_dashboard.png")


def main():
    print("=" * 60)
    print("Generating all plots...")
    print("=" * 60)

    plot_1_actual_vs_predicted_scatter()
    plot_2_error_distribution()
    # Tutti gli altri plot richiedono colonne non presenti nell'ensemble (horizon, sample_idx, ecc.)
    plot_10_summary_dashboard()
    print("[INFO] Plot avanzati disabilitati: dati insufficienti nell'ensemble.")

    print("=" * 60)
    print(f"[DONE] All plots saved to: {PLOT_DIR}")
    print("=" * 60)

    # List all files
    print("\nGenerated files:")
    for f in sorted(PLOT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
