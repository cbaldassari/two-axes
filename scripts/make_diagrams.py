"""
make_diagrams.py
================
Generate two conceptual diagrams for the paper:
1. Two-axis regime grid (economic x dynamic)
2. Sliding window and representation extraction
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

PAPER_DIR = Path("paper")

# ══════════════════════════════════════════════════════════════════════════════
#  DIAGRAM 1: Two-axis conceptual grid
# ══════════════════════════════════════════════════════════════════════════════

def make_two_axis_diagram():
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="white")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    # Quadrant colors
    colors = {
        "BL": "#d4e6f1",  # low price, fast-reverting (bottom-left)
        "BR": "#a9cce3",  # low price, sticky (bottom-right)
        "TL": "#f5cba7",  # high price, fast-reverting (top-left)
        "TR": "#e74c3c",  # high price, sticky (top-right)
    }
    alpha = 0.35

    # Draw quadrants
    ax.add_patch(FancyBboxPatch((0, 0), 0.48, 0.48, boxstyle="round,pad=0.02",
                                facecolor=colors["BL"], alpha=alpha, edgecolor="gray", linewidth=0.5))
    ax.add_patch(FancyBboxPatch((0.52, 0), 0.48, 0.48, boxstyle="round,pad=0.02",
                                facecolor=colors["BR"], alpha=alpha, edgecolor="gray", linewidth=0.5))
    ax.add_patch(FancyBboxPatch((0, 0.52), 0.48, 0.48, boxstyle="round,pad=0.02",
                                facecolor=colors["TL"], alpha=alpha, edgecolor="gray", linewidth=0.5))
    ax.add_patch(FancyBboxPatch((0.52, 0.52), 0.48, 0.48, boxstyle="round,pad=0.02",
                                facecolor=colors["TR"], alpha=alpha, edgecolor="gray", linewidth=0.5))

    # Quadrant labels
    fs_title = 11
    fs_detail = 8.5
    fs_pct = 13

    # Bottom-left: low price, fast-reverting
    ax.text(0.24, 0.38, "Low price", ha="center", fontsize=fs_title, fontweight="bold", color="#1a5276")
    ax.text(0.24, 0.30, "Fast mean-reversion", ha="center", fontsize=fs_detail, color="#1a5276")
    ax.text(0.24, 0.18, "34%", ha="center", fontsize=fs_pct, fontweight="bold", color="#1a5276")
    ax.text(0.24, 0.10, "Off-peak spring/autumn\nquick price correction", ha="center",
            fontsize=7.5, color="#2c3e50", style="italic")

    # Bottom-right: low price, sticky
    ax.text(0.76, 0.38, "Low price", ha="center", fontsize=fs_title, fontweight="bold", color="#1a5276")
    ax.text(0.76, 0.30, "Persistent / sticky", ha="center", fontsize=fs_detail, color="#1a5276")
    ax.text(0.76, 0.18, "38%", ha="center", fontsize=fs_pct, fontweight="bold", color="#1a5276")
    ax.text(0.76, 0.10, "Stable baseload periods\nslow drift", ha="center",
            fontsize=7.5, color="#2c3e50", style="italic")

    # Top-left: high price, fast-reverting
    ax.text(0.24, 0.90, "High price", ha="center", fontsize=fs_title, fontweight="bold", color="#922b21")
    ax.text(0.24, 0.82, "Fast mean-reversion", ha="center", fontsize=fs_detail, color="#922b21")
    ax.text(0.24, 0.70, "8%", ha="center", fontsize=fs_pct, fontweight="bold", color="#922b21")
    ax.text(0.24, 0.62, "Transient spikes\nrapid correction", ha="center",
            fontsize=7.5, color="#2c3e50", style="italic")

    # Top-right: high price, sticky
    ax.text(0.76, 0.90, "High price", ha="center", fontsize=fs_title, fontweight="bold", color="#922b21")
    ax.text(0.76, 0.82, "Persistent / sticky", ha="center", fontsize=fs_detail, color="#922b21")
    ax.text(0.76, 0.70, "20%", ha="center", fontsize=fs_pct, fontweight="bold", color="#922b21")
    ax.text(0.76, 0.62, "Sustained winter stress\nprolonged crisis", ha="center",
            fontsize=7.5, color="#2c3e50", style="italic")

    # Axes arrows
    ax.annotate("", xy=(1.12, -0.02), xytext=(-0.02, -0.02),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.annotate("", xy=(-0.02, 1.12), xytext=(-0.02, -0.02),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

    # Axis labels
    ax.text(0.50, -0.10, "Dynamic Axis (MOMENT)  $\\longrightarrow$  persistence",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(-0.10, 0.50, "Economic Axis (FE)  $\\longrightarrow$  price level",
            ha="center", fontsize=11, fontweight="bold", rotation=90)

    # Axis endpoint labels
    ax.text(0.0, -0.06, "fast-reverting", ha="center", fontsize=8, color="gray")
    ax.text(1.0, -0.06, "sticky", ha="center", fontsize=8, color="gray")
    ax.text(-0.07, 0.0, "low", ha="center", fontsize=8, color="gray", rotation=90)
    ax.text(-0.07, 1.0, "high", ha="center", fontsize=8, color="gray", rotation=90)

    # ARI annotation
    ax.text(0.50, 1.10, "ARI $\\approx$ 0.09  (nearly independent axes)",
            ha="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9e79f", edgecolor="#d4ac0d", alpha=0.8))

    fig.tight_layout(pad=1.5)
    out = PAPER_DIR / "two_axis_grid.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGRAM 2: Sliding window and representation extraction
# ══════════════════════════════════════════════════════════════════════════════

def make_windowing_diagram():
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.8, 5.5)
    ax.axis("off")

    # ── Stage 0: Time series ──
    # Draw a simplified time series
    np.random.seed(42)
    t = np.linspace(0, 10, 300)
    y = 3.5 + 0.8 * np.sin(t * 2) + 0.3 * np.random.randn(300) + 0.5 * np.sin(t * 0.5)
    ax.plot(t, y, color="#2c3e50", linewidth=0.8, alpha=0.7)
    ax.text(5.0, 5.2, "MSTL residual  $r_t$  (43,814 hours)", ha="center",
            fontsize=11, fontweight="bold")

    # ── Windows ──
    window_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]
    window_starts = [0.0, 0.15, 0.30, 0.45, 0.60]  # in axis units of t
    window_width = 2.56  # 512h / ~200 scale

    for i, (ws, color) in enumerate(zip(window_starts, window_colors)):
        ybot = 2.3 - i * 0.08
        ytop = 4.7 + i * 0.08
        alpha = 0.15 if i > 0 else 0.25
        ax.axvspan(ws, ws + window_width, ymin=0.55, ymax=0.95,
                   alpha=alpha, color=color, zorder=0)
        if i == 0:
            # Bracket for window 1
            ax.annotate("", xy=(ws, 2.4), xytext=(ws + window_width, 2.4),
                        arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))
            ax.text(ws + window_width / 2, 2.15, "$W = 512$ h ($\\approx$ 21 days)",
                    ha="center", fontsize=8.5, color=color, fontweight="bold")

    # Stride annotation
    ax.annotate("", xy=(0.0, 2.65), xytext=(0.15, 2.65),
                arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=1.2))
    ax.text(0.075, 2.80, "$S$=6h", ha="center", fontsize=7.5, color="#7f8c8d")

    # N windows annotation
    ax.text(8.0, 4.0, "$N = 7{,}217$ windows", fontsize=10, fontweight="bold",
            color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2f8", edgecolor="#aed6f1"))

    # ── Stage 1: Arrow down to representations ──
    ax.annotate("", xy=(2.5, 1.0), xytext=(2.5, 1.8),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2))
    ax.text(2.5, 1.4, "per window", ha="center", fontsize=8, color="#7f8c8d")

    # ── Representation A: FE ──
    fe_box = FancyBboxPatch((0.3, -0.2), 4.0, 1.1, boxstyle="round,pad=0.1",
                            facecolor="#fadbd8", edgecolor="#e74c3c", linewidth=1.5)
    ax.add_patch(fe_box)
    ax.text(2.3, 0.55, "Feature Engineering", ha="center", fontsize=10, fontweight="bold",
            color="#c0392b")
    ax.text(2.3, 0.15, "19 features: statistics, ACF,\nvolatility, raw LMP level",
            ha="center", fontsize=8, color="#2c3e50")
    ax.text(4.0, 0.75, "19D", ha="center", fontsize=9, fontweight="bold",
            color="#c0392b",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#e74c3c"))

    # ── Representation B: MOMENT ──
    mom_box = FancyBboxPatch((5.5, -0.2), 4.5, 1.1, boxstyle="round,pad=0.1",
                             facecolor="#d4e6f1", edgecolor="#2980b9", linewidth=1.5)
    ax.add_patch(mom_box)
    ax.text(7.75, 0.55, "MOMENT Encoder", ha="center", fontsize=10, fontweight="bold",
            color="#2471a3")
    ax.text(7.75, 0.15, "zero-shot embedding\n(340M params, masked reconstruction)",
            ha="center", fontsize=8, color="#2c3e50")
    ax.text(9.7, 0.75, "1,024D", ha="center", fontsize=9, fontweight="bold",
            color="#2471a3",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#2980b9"))

    # Arrow from time series to MOMENT
    ax.annotate("", xy=(7.75, 1.0), xytext=(7.75, 1.8),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2))

    # ── Stage 2: Arrows down to pipeline ──
    ax.annotate("", xy=(2.3, -0.9), xytext=(2.3, -0.35),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.5))
    ax.annotate("", xy=(7.75, -0.9), xytext=(7.75, -0.35),
                arrowprops=dict(arrowstyle="-|>", color="#2471a3", lw=1.5))

    # Pipeline boxes
    pipe_y = -1.55
    pipe_h = 0.55

    # DiffMaps
    dm1 = FancyBboxPatch((0.8, pipe_y), 2.8, pipe_h, boxstyle="round,pad=0.08",
                          facecolor="#fdebd0", edgecolor="#e67e22", linewidth=1.2)
    ax.add_patch(dm1)
    ax.text(2.2, pipe_y + pipe_h / 2, "DiffMaps $\\to$ 11D", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#ca6f1e")

    dm2 = FancyBboxPatch((6.0, pipe_y), 3.2, pipe_h, boxstyle="round,pad=0.08",
                          facecolor="#fdebd0", edgecolor="#e67e22", linewidth=1.2)
    ax.add_patch(dm2)
    ax.text(7.6, pipe_y + pipe_h / 2, "PCA + DiffMaps $\\to$ 2D", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#ca6f1e")

    # Arrows to results
    ax.annotate("", xy=(2.2, pipe_y - 0.25), xytext=(2.2, pipe_y),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.5))
    ax.annotate("", xy=(7.6, pipe_y - 0.25), xytext=(7.6, pipe_y),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.5))

    # Merge at bottom: convergent arrows to single label
    # But actually keep them separate to show the two axes

    fig.tight_layout(pad=0.5)
    out = PAPER_DIR / "windowing_diagram.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_two_axis_diagram()
    make_windowing_diagram()
