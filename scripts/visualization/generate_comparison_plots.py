"""Generate model comparison plots and architecture diagram.

Compares Multi-Branch Transformer with TFT, CNN-BiLSTM, and shows architecture.
"""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Set style
plt.style.use("seaborn-v0_8-whitegrid")

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Model comparison data
# ============================================================

# Overall metrics from README and experiments
MODELS_OVERALL = {
    "Multi-Branch\nTransformer\n(Seed 2)": {"rmse": 3.24, "mase": 0.51, "params": "5.0M", "color": "#2ecc71"},
    "TFT\nBaseline": {"rmse": 3.71, "mase": 0.43, "params": "2.3M", "color": "#3498db"},
    "CNN-BiLSTM\nBaseline": {"rmse": 3.73, "mase": 0.47, "params": "597K", "color": "#e74c3c"},
    "CNN-BiLSTM\nFusion": {"rmse": 4.36, "mase": 0.55, "params": "1.2M", "color": "#9b59b6"},
}

# Per-horizon metrics (we need to load TFT data)
print("Loading TFT metrics...")
with open("outputs/tft/baseline/metrics_test_tft.json") as f:
    tft_metrics = pd.DataFrame(json.load(f))

print("Loading Multi-Branch metrics...")
with open("outputs/multi_branch/final_seed2/metrics_final.json") as f:
    mb_metrics = json.load(f)
mb_horizon = pd.DataFrame(mb_metrics["horizon_metrics"])


def plot_11_model_comparison_overall():
    """Bar chart comparing overall metrics across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = list(MODELS_OVERALL.keys())
    rmse_vals = [MODELS_OVERALL[m]["rmse"] for m in models]
    mase_vals = [MODELS_OVERALL[m]["mase"] for m in models]
    colors = [MODELS_OVERALL[m]["color"] for m in models]

    # RMSE comparison
    bars1 = axes[0].bar(models, rmse_vals, color=colors, edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel("RMSE (kW)", fontsize=12)
    axes[0].set_title("RMSE Comparison (Lower is Better)", fontsize=14)
    axes[0].set_ylim(0, max(rmse_vals) * 1.15)

    # Add value labels
    for bar, val in zip(bars1, rmse_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    # Add best marker
    best_idx = np.argmin(rmse_vals)
    axes[0].annotate(
        "BEST",
        xy=(best_idx, rmse_vals[best_idx]),
        xytext=(best_idx, rmse_vals[best_idx] + 0.4),
        ha="center",
        fontsize=10,
        color="green",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="green"),
    )

    # MASE comparison
    bars2 = axes[1].bar(models, mase_vals, color=colors, edgecolor="black", linewidth=1.5)
    axes[1].axhline(1.0, color="gray", linestyle="--", lw=2, label="Naive Baseline")
    axes[1].set_ylabel("MASE", fontsize=12)
    axes[1].set_title("MASE Comparison (Lower is Better)", fontsize=14)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()

    # Add value labels
    for bar, val in zip(bars2, mase_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    # Add best marker
    best_idx = np.argmin(mase_vals)
    axes[1].annotate(
        "BEST",
        xy=(best_idx, mase_vals[best_idx]),
        xytext=(best_idx, mase_vals[best_idx] + 0.08),
        ha="center",
        fontsize=10,
        color="green",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="green"),
    )

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "11_model_comparison_overall.png", dpi=150)
    plt.close()
    print("✓ 11_model_comparison_overall.png")


def plot_12_model_comparison_by_horizon():
    """Line plot comparing RMSE and MASE by horizon for Multi-Branch vs TFT."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    horizons = range(1, 25)

    # RMSE by horizon
    axes[0].plot(horizons, mb_horizon["rmse"], "g-o", label="Multi-Branch (Seed 2)", linewidth=2, markersize=6)
    axes[0].plot(horizons, tft_metrics["rmse_model"], "b-s", label="TFT Baseline", linewidth=2, markersize=6)
    axes[0].fill_between(
        horizons,
        mb_horizon["rmse"],
        tft_metrics["rmse_model"],
        alpha=0.2,
        color="green",
        where=mb_horizon["rmse"] < tft_metrics["rmse_model"].values,
    )
    axes[0].fill_between(
        horizons,
        mb_horizon["rmse"],
        tft_metrics["rmse_model"],
        alpha=0.2,
        color="blue",
        where=mb_horizon["rmse"] >= tft_metrics["rmse_model"].values,
    )

    axes[0].set_xlabel("Forecast Horizon (hours)", fontsize=12)
    axes[0].set_ylabel("RMSE (kW)", fontsize=12)
    axes[0].set_title("RMSE by Horizon: Multi-Branch vs TFT", fontsize=14)
    axes[0].legend()
    axes[0].set_xticks(range(1, 25, 2))
    axes[0].grid(True, alpha=0.3)

    # MASE by horizon
    axes[1].plot(horizons, mb_horizon["mase"], "g-o", label="Multi-Branch (Seed 2)", linewidth=2, markersize=6)
    axes[1].plot(horizons, tft_metrics["mase_model"], "b-s", label="TFT Baseline", linewidth=2, markersize=6)
    axes[1].axhline(1.0, color="gray", linestyle="--", lw=1.5, label="Naive Baseline")
    axes[1].fill_between(
        horizons,
        mb_horizon["mase"],
        tft_metrics["mase_model"],
        alpha=0.2,
        color="green",
        where=mb_horizon["mase"] < tft_metrics["mase_model"].values,
    )
    axes[1].fill_between(
        horizons,
        mb_horizon["mase"],
        tft_metrics["mase_model"],
        alpha=0.2,
        color="blue",
        where=mb_horizon["mase"] >= tft_metrics["mase_model"].values,
    )

    axes[1].set_xlabel("Forecast Horizon (hours)", fontsize=12)
    axes[1].set_ylabel("MASE", fontsize=12)
    axes[1].set_title("MASE by Horizon: Multi-Branch vs TFT", fontsize=14)
    axes[1].legend()
    axes[1].set_xticks(range(1, 25, 2))
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "12_model_comparison_by_horizon.png", dpi=150)
    plt.close()
    print("✓ 12_model_comparison_by_horizon.png")


def plot_13_improvement_heatmap():
    """Heatmap showing % improvement of Multi-Branch over TFT."""
    # Calculate % improvement (positive = Multi-Branch is better)
    rmse_improvement = (
        (tft_metrics["rmse_model"].values - mb_horizon["rmse"].values) / tft_metrics["rmse_model"].values * 100
    )
    mase_improvement = (
        (tft_metrics["mase_model"].values - mb_horizon["mase"].values) / tft_metrics["mase_model"].values * 100
    )

    data = pd.DataFrame(
        {"Horizon": range(1, 25), "RMSE Improvement (%)": rmse_improvement, "MASE Improvement (%)": mase_improvement}
    )

    fig, ax = plt.subplots(figsize=(16, 4))

    # Create a diverging colormap (green = better, red = worse)
    improvement_matrix = np.array([rmse_improvement, mase_improvement])

    im = ax.imshow(improvement_matrix, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=20)

    ax.set_xticks(range(24))
    ax.set_xticklabels(range(1, 25))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["RMSE", "MASE"])
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_title("% Improvement: Multi-Branch vs TFT (Green = MB Better, Red = TFT Better)", fontsize=14)

    # Add text annotations
    for i in range(2):
        for j in range(24):
            val = improvement_matrix[i, j]
            color = "white" if abs(val) > 10 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="% Improvement", shrink=0.8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "13_improvement_heatmap.png", dpi=150)
    plt.close()
    print("✓ 13_improvement_heatmap.png")


def plot_14_architecture_diagram():
    """Detailed architecture diagram of Multi-Branch Transformer."""
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Colors
    colors = {
        "pv": "#3498db",  # Blue
        "weather_hist": "#2ecc71",  # Green
        "weather_fcst": "#e74c3c",  # Red
        "fusion": "#9b59b6",  # Purple
        "output": "#f39c12",  # Orange
        "attention": "#1abc9c",  # Teal
    }

    def draw_box(x, y, w, h, label, color, fontsize=10, sublabel=None):
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05", facecolor=color, edgecolor="black", linewidth=2, alpha=0.8
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2 + (0.1 if sublabel else 0),
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="white",
        )
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25, sublabel, ha="center", va="center", fontsize=8, color="white")

    def draw_arrow(start, end, color="black"):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=2))

    # Title
    ax.text(10, 13.5, "Multi-Branch Transformer Architecture", ha="center", fontsize=18, fontweight="bold")
    ax.text(10, 13, "d_model=256, num_heads=4, num_layers=2, dropout=0.2", ha="center", fontsize=12, style="italic")

    # ============ INPUT LAYER ============
    ax.text(10, 12.3, "═══════════════ INPUT LAYER ═══════════════", ha="center", fontsize=11, color="gray")

    # Input boxes
    draw_box(1, 11, 4, 1, "PV History", colors["pv"], 11, "168h × 1 feature")
    draw_box(8, 11, 4, 1, "Weather History", colors["weather_hist"], 11, "168h × 14 features")
    draw_box(15, 11, 4, 1, "Weather Forecast", colors["weather_fcst"], 11, "24h × 14 features")

    # ============ EMBEDDING LAYER ============
    ax.text(10, 10.3, "═══════════════ INPUT EMBEDDING ═══════════════", ha="center", fontsize=11, color="gray")

    draw_box(1, 9.2, 4, 0.8, "Linear(1→256)", colors["pv"], 10)
    draw_box(8, 9.2, 4, 0.8, "Linear(14→256)", colors["weather_hist"], 10)
    draw_box(15, 9.2, 4, 0.8, "Linear(14→256)", colors["weather_fcst"], 10)

    # Arrows from input to embedding
    draw_arrow((3, 11), (3, 10), colors["pv"])
    draw_arrow((10, 11), (10, 10), colors["weather_hist"])
    draw_arrow((17, 11), (17, 10), colors["weather_fcst"])

    # ============ TRANSFORMER BRANCHES ============
    ax.text(10, 8.6, "═══════════════ TRANSFORMER BRANCHES ═══════════════", ha="center", fontsize=11, color="gray")

    # Branch boxes (larger, with details)
    draw_box(0.5, 5.5, 5, 2.8, "PV Branch", colors["pv"], 12)
    draw_box(7.5, 5.5, 5, 2.8, "Weather Hist Branch", colors["weather_hist"], 12)
    draw_box(14.5, 5.5, 5, 2.8, "Weather Fcst Branch", colors["weather_fcst"], 12)

    # Add branch details
    for x_pos in [3, 10, 17]:
        ax.text(x_pos, 7.5, "TransformerEncoder", ha="center", fontsize=9, color="white")
        ax.text(x_pos, 7.0, "× 2 layers", ha="center", fontsize=8, color="white")
        ax.text(x_pos, 6.5, "MultiHead Attention (4 heads)", ha="center", fontsize=8, color="white")
        ax.text(x_pos, 6.0, "FFN (256→1024→256)", ha="center", fontsize=8, color="white")

    # Arrows from embedding to branches
    draw_arrow((3, 9.2), (3, 8.3), colors["pv"])
    draw_arrow((10, 9.2), (10, 8.3), colors["weather_hist"])
    draw_arrow((17, 9.2), (17, 8.3), colors["weather_fcst"])

    # ============ FUSION LAYER ============
    ax.text(10, 5, "═══════════════ HIERARCHICAL FUSION ═══════════════", ha="center", fontsize=11, color="gray")

    # Stage 1: Fuse PV + Weather History
    draw_box(3.5, 3.2, 6, 1.2, "Fusion Stage 1", colors["fusion"], 11, "Soft Attention: PV + Weather Hist")

    # Arrows to Stage 1
    draw_arrow((3, 5.5), (5, 4.4), colors["pv"])
    draw_arrow((10, 5.5), (8, 4.4), colors["weather_hist"])

    # Stage 2: Fuse History + Forecast
    draw_box(10.5, 3.2, 6, 1.2, "Fusion Stage 2", colors["fusion"], 11, "Soft Attention: Fused Hist + Forecast")

    # Arrows to Stage 2
    draw_arrow((6.5, 3.8), (10.5, 3.8), colors["fusion"])
    draw_arrow((17, 5.5), (14.5, 4.4), colors["weather_fcst"])

    # ============ OUTPUT LAYER ============
    ax.text(10, 2.6, "═══════════════ OUTPUT LAYER ═══════════════", ha="center", fontsize=11, color="gray")

    # Decoder
    draw_box(7, 1.2, 6, 1, "Linear Decoder", colors["output"], 11, "256 → 24 horizons")

    # Arrow to output
    draw_arrow((13.5, 3.2), (11.5, 2.2), colors["fusion"])

    # Output
    draw_box(7, 0, 6, 0.8, "PV Forecast (24h)", colors["output"], 12)
    draw_arrow((10, 1.2), (10, 0.8), colors["output"])

    # ============ LEGEND ============
    legend_items = [
        ("PV Processing", colors["pv"]),
        ("Weather History", colors["weather_hist"]),
        ("Weather Forecast", colors["weather_fcst"]),
        ("Fusion Layers", colors["fusion"]),
        ("Output", colors["output"]),
    ]

    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((0.3, 0.3 + i * 0.35), 0.4, 0.25, facecolor=color, edgecolor="black"))
        ax.text(0.85, 0.425 + i * 0.35, label, fontsize=9, va="center")

    # Model stats
    stats_text = """Model Statistics:
    • Total Parameters: 5.0M
    • d_model: 256
    • Attention Heads: 4
    • Transformer Layers: 2/branch
    • Dropout: 0.2
    • Input: 168h history
    • Output: 24h forecast"""

    ax.text(
        17,
        2.2,
        stats_text,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "14_architecture_diagram.png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()
    print("✓ 14_architecture_diagram.png")


def plot_15_model_summary_table():
    """Create a visual summary table of all models."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    # Table data
    columns = ["Model", "RMSE (kW)", "MASE", "Parameters", "Training Time", "Status"]
    data = [
        ["Multi-Branch Transformer (Seed 2)", "3.24", "0.51", "5.0M", "~20 min", "✅ BEST RMSE"],
        ["TFT Baseline", "3.71", "0.43", "2.3M", "~4 hours", "✅ Best MASE"],
        ["CNN-BiLSTM Baseline", "3.73", "0.47", "597K", "~2.5 hours", "✅ Completed"],
        ["CNN-BiLSTM Fusion", "4.36", "0.55", "1.2M", "~3 hours", "⚠️ Overfitted"],
        ["LightGBM (24 models)", "-", ">1.0 (h18+)", "~50K", "~10 min", "❌ Degraded at h18-24"],
    ]

    # Create table
    table = ax.table(cellText=data, colLabels=columns, loc="center", cellLoc="center", colColours=["#34495e"] * 6)

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color="white", fontweight="bold")
        table[(0, i)].set_facecolor("#34495e")

    # Style rows
    colors_rows = ["#d5f5e3", "#d6eaf8", "#d6eaf8", "#fadbd8", "#f9ebea"]
    for i, color in enumerate(colors_rows, start=1):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)

    # Highlight best
    table[(1, 1)].set_text_props(fontweight="bold", color="green")
    table[(2, 2)].set_text_props(fontweight="bold", color="green")

    ax.set_title("Model Comparison Summary\n24-Hour Ahead PV Power Forecasting", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "15_model_summary_table.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ 15_model_summary_table.png")


def main():
    print("=" * 60)
    print("Generating model comparison and architecture plots...")
    print("=" * 60)

    plot_11_model_comparison_overall()
    plot_12_model_comparison_by_horizon()
    plot_13_improvement_heatmap()
    plot_14_architecture_diagram()
    plot_15_model_summary_table()

    print("=" * 60)
    print(f"✅ All comparison plots saved to: {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
