"""
make_diagrams_v3.py — Cleaner, more readable versions of Figures 3 and 4.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
PAPER_DIR = Path("paper")


def make_windowing_diagram():
    """Figure 3: Sliding-window overview — series, windows, two paths."""

    prep = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    resid = prep["mstl_resid_arcsinh"].values

    fig = plt.figure(figsize=(11, 7.5), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.30,
                          left=0.05, right=0.95, top=0.94, bottom=0.04)

    # ── TOP PANEL: time series with windows ──
    ax = fig.add_subplot(gs[0])
    # Show ~3000 hours to make windows visible
    n_show = 3000
    t = np.arange(n_show)
    ax.plot(t, resid[:n_show], color="#2c3e50", linewidth=0.5, alpha=0.7)

    # Draw 4 example windows
    w_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    w_starts = [200, 206, 212, 218]  # stride=6 between them
    for i, (ws, c) in enumerate(zip(w_starts, w_colors)):
        ax.axvspan(ws, ws + 512, alpha=0.12 if i > 0 else 0.22, color=c, zorder=0)

    # Bracket for W=512
    bh = resid[:n_show].min() - 0.15
    ax.annotate("", xy=(200, bh), xytext=(712, bh),
                arrowprops=dict(arrowstyle="<->", color="#3498db", lw=2))
    ax.text(456, bh - 0.12, "$W = 512$ h  ($\\approx$ 21 days)",
            ha="center", fontsize=10, fontweight="bold", color="#3498db")

    # Stride bracket
    ax.annotate("", xy=(200, resid[:n_show].max() + 0.1),
                xytext=(206, resid[:n_show].max() + 0.1),
                arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=1.5))
    ax.text(203, resid[:n_show].max() + 0.2, "$S = 6$ h",
            ha="center", fontsize=9, fontweight="bold", color="#e74c3c")

    # N windows box
    ax.text(2200, resid[:n_show].max() - 0.1,
            "$N = 7{,}217$ windows",
            fontsize=12, fontweight="bold", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf2f8",
                      edgecolor="#aed6f1", alpha=0.95))

    ax.set_xlim(0, n_show)
    ax.set_ylabel("MSTL residual", fontsize=10)
    ax.set_xlabel("Hour", fontsize=10)
    ax.set_title("MSTL residual $r_t$ with overlapping sliding windows",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.1)

    # ── BOTTOM PANEL: two paths (as diagram) ──
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)

    # Central arrow label
    ax2.text(5, 4.6, "each window $\\longrightarrow$ two parallel representations",
             ha="center", fontsize=11, fontweight="bold", color="#555555")

    # ── LEFT: FE box ──
    fe = FancyBboxPatch((0.2, 0.8), 4.3, 3.5, boxstyle="round,pad=0.15",
                         facecolor="#fadbd8", edgecolor="#c0392b", linewidth=2)
    ax2.add_patch(fe)
    ax2.text(2.35, 3.9, "A: Feature Engineering", ha="center",
             fontsize=12, fontweight="bold", color="#922b21")
    ax2.text(2.35, 3.4, "19 features  $\\rightarrow$  19D vector", ha="center",
             fontsize=10, fontweight="bold", color="#c0392b")

    lines_fe = [
        "11 distributional statistics",
        "(mean, std, skew, kurtosis, min, max,",
        " range, median, P5, P95, IQR)",
        "",
        "4 temporal persistence",
        "(ACF at lag 1h, 6h, 24h, 168h)",
        "",
        "1 intra-day volatility",
        "3 raw LMP statistics",
    ]
    y = 2.9
    for line in lines_fe:
        if line == "":
            y -= 0.15
            continue
        ax2.text(0.5, y, line, fontsize=8, color="#333333", va="top")
        y -= 0.22

    # Arrow FE → DiffMaps
    ax2.annotate("DiffMaps $\\rightarrow$ 11D", xy=(2.35, 0.55), xytext=(2.35, 0.85),
                 fontsize=9, fontweight="bold", color="#c0392b", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#fdebd0",
                           edgecolor="#e67e22"))

    # ── RIGHT: MOMENT box ──
    mom = FancyBboxPatch((5.5, 0.8), 4.3, 3.5, boxstyle="round,pad=0.15",
                          facecolor="#d4e6f1", edgecolor="#2471a3", linewidth=2)
    ax2.add_patch(mom)
    ax2.text(7.65, 3.9, "B: MOMENT Encoder", ha="center",
             fontsize=12, fontweight="bold", color="#1a5276")
    ax2.text(7.65, 3.4, "512 values  $\\rightarrow$  1,024D vector", ha="center",
             fontsize=10, fontweight="bold", color="#2471a3")

    lines_mom = [
        "MOMENT-1-large (340M params)",
        "",
        "Pre-trained on 385K+ time series",
        "via masked reconstruction",
        "",
        "Zero-shot inference:",
        "window $\\rightarrow$ transformer encoder",
        "$\\rightarrow$ mean-pool hidden states",
        "",
        "Domain-agnostic embedding",
    ]
    y = 2.9
    for line in lines_mom:
        if line == "":
            y -= 0.15
            continue
        ax2.text(5.8, y, line, fontsize=8, color="#333333", va="top")
        y -= 0.22

    # Arrow MOMENT → DiffMaps
    ax2.annotate("PCA + DiffMaps $\\rightarrow$ 2D", xy=(7.65, 0.55), xytext=(7.65, 0.85),
                 fontsize=9, fontweight="bold", color="#2471a3", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#fdebd0",
                           edgecolor="#e67e22"))

    out = PAPER_DIR / "windowing_diagram.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def make_single_window_diagram():
    """Figure 4: Zoom on one 512h window with feature detail."""

    prep = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    # Pick a representative window
    start_idx = 18000
    window = prep["mstl_resid_arcsinh"].values[start_idx:start_idx + 512]

    fig = plt.figure(figsize=(12, 6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.05,
                          left=0.06, right=0.94, top=0.90, bottom=0.06)

    # ── LEFT: Time series of the window ──
    ax = fig.add_subplot(gs[0])
    hours = np.arange(512)
    ax.plot(hours, window, color="#2c3e50", linewidth=0.8)
    ax.fill_between(hours, window, alpha=0.06, color="#2c3e50")

    mean_v = np.mean(window)
    std_v = np.std(window)
    ax.axhline(mean_v, color="#e74c3c", ls="--", lw=1, alpha=0.7)
    ax.axhspan(mean_v - std_v, mean_v + std_v, alpha=0.06, color="#e74c3c")

    # Mark key features visually
    # ACF lag-6h
    for lag_start in [0, 100, 200, 300]:
        ax.annotate("", xy=(lag_start + 6, window[lag_start + 6]),
                    xytext=(lag_start, window[lag_start]),
                    arrowprops=dict(arrowstyle="-", color="#2980b9", lw=0.8, alpha=0.3))

    # Max spike
    max_idx = np.argmax(window)
    ax.plot(max_idx, window[max_idx], "v", color="#e74c3c", markersize=8)
    ax.annotate("max", xy=(max_idx, window[max_idx]),
                xytext=(max_idx + 30, window[max_idx] + 0.1),
                fontsize=8, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8))

    # Min
    min_idx = np.argmin(window)
    ax.plot(min_idx, window[min_idx], "^", color="#27ae60", markersize=8)

    ax.set_xlim(0, 511)
    ax.set_xlabel("Hour within window", fontsize=10)
    ax.set_ylabel("MSTL residual", fontsize=10)
    ax.set_title("One window: $W = 512$ h ($\\approx$ 21 days)",
                 fontsize=12, fontweight="bold")

    # Legend with computed values
    ax.text(0.98, 0.97,
            f"mean = {mean_v:.2f}\n"
            f"std = {std_v:.2f}\n"
            f"skew = {float(pd.Series(window).skew()):.2f}\n"
            f"ACF(1h) = {np.corrcoef(window[:-1], window[1:])[0,1]:.3f}\n"
            f"ACF(6h) = {np.corrcoef(window[:-6], window[6:])[0,1]:.3f}",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
            family="monospace")
    ax.grid(alpha=0.12)

    # ── RIGHT: Feature extraction summary ──
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Title
    ax2.text(5, 9.5, "Extracted from this window:", ha="center",
             fontsize=11, fontweight="bold", color="#333333")

    # FE section
    y = 8.8
    ax2.text(0.5, y, "Repr. A: Feature Engineering (19D)",
             fontsize=10, fontweight="bold", color="#c0392b")
    items_fe = [
        ("Distributional (11)", "shape of the price distribution"),
        ("Persistence (4)", "autocorrelation at multiple lags"),
        ("Volatility (1)", "intra-day price movement"),
        ("Raw LMP (3)", "absolute price level"),
    ]
    y -= 0.5
    for name, desc in items_fe:
        ax2.text(0.8, y, f"{name}", fontsize=9, fontweight="bold", color="#555")
        ax2.text(4.5, y, f"{desc}", fontsize=8.5, color="#777", style="italic")
        y -= 0.5

    # Separator
    y -= 0.2
    ax2.plot([0.5, 9.5], [y + 0.15, y + 0.15], color="#ddd", lw=1)

    # MOMENT section
    y -= 0.3
    ax2.text(0.5, y, "Repr. B: MOMENT Encoder (1,024D)",
             fontsize=10, fontweight="bold", color="#1a5276")
    items_mom = [
        ("Input", "raw 512 residual values"),
        ("Model", "MOMENT-1-large, 340M params"),
        ("Method", "zero-shot, no fine-tuning"),
        ("Output", "1,024D domain-agnostic embedding"),
        ("Captures", "temporal dynamics, not price level"),
    ]
    y -= 0.5
    for name, desc in items_mom:
        ax2.text(0.8, y, f"{name}:", fontsize=9, fontweight="bold", color="#555")
        ax2.text(3.0, y, f"{desc}", fontsize=8.5, color="#777", style="italic")
        y -= 0.5

    # Separator
    y -= 0.2
    ax2.plot([0.5, 9.5], [y + 0.15, y + 0.15], color="#ddd", lw=1)

    # Downstream
    y -= 0.3
    ax2.text(0.5, y, "Downstream:", fontsize=10, fontweight="bold", color="#e67e22")
    y -= 0.5
    ax2.text(0.8, y, "FE $\\rightarrow$ DiffMaps $\\rightarrow$ 11D  |  "
             "MOMENT $\\rightarrow$ PCA + DiffMaps $\\rightarrow$ 2D",
             fontsize=9, color="#555")
    y -= 0.5
    ax2.text(0.8, y, "$\\rightarrow$ ToMATo clustering $\\rightarrow$ Tukey HSD merge",
             fontsize=9, color="#555")

    fig.suptitle("", fontsize=1)  # clear any default
    out = PAPER_DIR / "single_window_detail.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_windowing_diagram()
    make_single_window_diagram()
