"""
make_diagrams_v2.py
===================
1. Single-window detail diagram (what happens inside one 512h window)
2. Regenerated TDA barcode (standalone, no "(b)" label)
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
from gudhi.clustering.tomato import Tomato

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
PAPER_DIR = Path("paper")


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGRAM: Single window detail
# ══════════════════════════════════════════════════════════════════════════════

def make_single_window_diagram():
    # Load a real window for the time series subplot
    prep = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    prep["datetime"] = pd.to_datetime(prep["datetime"])

    # Pick a representative window (e.g., starting around Feb 2023 — winter, interesting)
    start_idx = 18000  # roughly Feb 2023
    end_idx = start_idx + 512
    window_resid = prep["mstl_resid_arcsinh"].values[start_idx:end_idx]
    window_lmp = prep["lmp"].values[start_idx:end_idx]
    window_dt = prep["datetime"].values[start_idx:end_idx]

    fig = plt.figure(figsize=(13, 8), facecolor="white")

    # Layout: top row = time series, bottom row = two representation boxes
    # Use gridspec for precise control
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.15, 1.6],
                          hspace=0.35, wspace=0.35,
                          left=0.06, right=0.94, top=0.92, bottom=0.04)

    # ── Top: Time series of the window ──
    ax_ts = fig.add_subplot(gs[0, :])
    hours = np.arange(512)

    ax_ts.plot(hours, window_resid, color="#2c3e50", linewidth=0.8, alpha=0.9)
    ax_ts.fill_between(hours, window_resid, alpha=0.08, color="#2c3e50")
    ax_ts.set_xlim(0, 511)
    ax_ts.set_xlabel("Hour within window", fontsize=9)
    ax_ts.set_ylabel("MSTL residual", fontsize=9)
    ax_ts.set_title("One sliding window:  $W = 512$ hours  ($\\approx$ 21 days)",
                     fontsize=12, fontweight="bold", pad=10)

    # Mark some features visually
    mean_val = np.mean(window_resid)
    std_val = np.std(window_resid)
    ax_ts.axhline(mean_val, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label=f"mean = {mean_val:.2f}")
    ax_ts.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.08, color="#e74c3c", label=f"$\\pm$1 std = {std_val:.2f}")

    # Mark ACF concept: highlight lag-6h correlation
    ax_ts.annotate("", xy=(6, window_resid[6]), xytext=(0, window_resid[0]),
                    arrowprops=dict(arrowstyle="<->", color="#2980b9", lw=1.5, alpha=0.6))
    ax_ts.text(3, max(window_resid[0], window_resid[6]) + 0.08, "lag 6h",
               ha="center", fontsize=7.5, color="#2980b9", fontweight="bold")

    ax_ts.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax_ts.grid(alpha=0.15)

    # ── Arrow row (just text) ──
    ax_arrow = fig.add_subplot(gs[1, :])
    ax_arrow.axis("off")
    ax_arrow.text(0.5, 0.5, "$\\Downarrow$   extract two parallel representations   $\\Downarrow$",
                  ha="center", va="center", fontsize=11, color="#7f8c8d", fontweight="bold",
                  transform=ax_arrow.transAxes)

    # ── Bottom left: Feature Engineering ──
    ax_fe = fig.add_subplot(gs[2, 0])
    ax_fe.axis("off")
    ax_fe.set_xlim(0, 10)
    ax_fe.set_ylim(0, 10)

    # Box
    fe_box = FancyBboxPatch((0.3, 0.3), 9.4, 9.2, boxstyle="round,pad=0.2",
                             facecolor="#fadbd8", edgecolor="#c0392b", linewidth=2)
    ax_fe.add_patch(fe_box)

    ax_fe.text(5, 9.0, "Representation A: Feature Engineering",
               ha="center", fontsize=11, fontweight="bold", color="#922b21")
    ax_fe.text(5, 8.2, "19 features  $\\rightarrow$  19D vector",
               ha="center", fontsize=10, color="#c0392b", fontweight="bold")

    # Feature groups
    y = 7.2
    ax_fe.text(1, y, "Distributional (11):", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax_fe.text(1, y - 0.6, "mean, std, skew, kurtosis, min, max,\nrange, median, P5, P95, IQR",
               fontsize=7.5, color="#566573", linespacing=1.4)

    y = 5.0
    ax_fe.text(1, y, "Temporal persistence (4):", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax_fe.text(1, y - 0.6, "ACF at lag 1h, 6h, 24h, 168h",
               fontsize=7.5, color="#566573")

    y = 3.6
    ax_fe.text(1, y, "Intra-day volatility (1):", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax_fe.text(1, y - 0.6, "mean |$\\Delta r$| per 24h block",
               fontsize=7.5, color="#566573")

    y = 2.2
    ax_fe.text(1, y, "Raw LMP statistics (3):", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax_fe.text(1, y - 0.6, "LMP mean, P95, std  (absolute level)",
               fontsize=7.5, color="#566573")

    ax_fe.text(5, 0.7, "computed on MSTL residual",
               ha="center", fontsize=8, color="#922b21", style="italic")

    # ── Bottom right: MOMENT ──
    ax_mom = fig.add_subplot(gs[2, 1])
    ax_mom.axis("off")
    ax_mom.set_xlim(0, 10)
    ax_mom.set_ylim(0, 10)

    mom_box = FancyBboxPatch((0.3, 0.3), 9.4, 9.2, boxstyle="round,pad=0.2",
                              facecolor="#d4e6f1", edgecolor="#2471a3", linewidth=2)
    ax_mom.add_patch(mom_box)

    ax_mom.text(5, 9.0, "Representation B: MOMENT Encoder",
                ha="center", fontsize=11, fontweight="bold", color="#1a5276")
    ax_mom.text(5, 8.2, "512 values  $\\rightarrow$  1,024D vector",
                ha="center", fontsize=10, color="#2471a3", fontweight="bold")

    y = 7.0
    ax_mom.text(1, y, "Model:", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax_mom.text(1, y - 0.6, "MOMENT-1-large (340M parameters)",
                fontsize=7.5, color="#566573")

    y = 5.6
    ax_mom.text(1, y, "Pre-training:", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax_mom.text(1, y - 0.6, "masked reconstruction on 385K+\ndiverse time series (Time-Series Pile)",
                fontsize=7.5, color="#566573", linespacing=1.4)

    y = 4.0
    ax_mom.text(1, y, "Inference:", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax_mom.text(1, y - 0.6, "zero-shot (no fine-tuning)\nraw 512h window $\\rightarrow$ transformer encoder\n$\\rightarrow$ mean-pool hidden states",
                fontsize=7.5, color="#566573", linespacing=1.4)

    y = 2.0
    ax_mom.text(1, y, "Output:", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax_mom.text(1, y - 0.6, "domain-agnostic embedding capturing\ntemporal dynamics, not price level",
                fontsize=7.5, color="#566573", linespacing=1.4)

    ax_mom.text(5, 0.7, "computed on MSTL residual",
                ha="center", fontsize=8, color="#1a5276", style="italic")

    out = PAPER_DIR / "single_window_detail.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  TDA BARCODE (regenerated, standalone)
# ══════════════════════════════════════════════════════════════════════════════

def make_tda_barcode():
    # Load the ORIGINAL saved DiffMaps coordinates
    df = pd.read_parquet(RESULTS_DIR / "exp_FE" / "step03" / "pca_embeddings.parquet")
    E = df.drop(columns=["datetime"]).values.astype(np.float32)

    t = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1, k=100)
    t.fit(E)
    diagram = np.array(t.diagram_)

    # Persistence sorted by absolute value (most persistent at top)
    raw_pers = diagram[:, 1] - diagram[:, 0]
    abs_pers = np.sort(np.abs(raw_pers))[::-1]
    n_modes = len(abs_pers)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    y_positions = np.arange(n_modes)

    # Color gradient: most persistent in red, fading to gray
    colors = []
    for i in range(n_modes):
        frac = i / n_modes
        if frac < 0.3:
            colors.append("#c0392b")
        elif frac < 0.6:
            colors.append("#e67e22")
        else:
            colors.append("#bdc3c7")

    ax.barh(y_positions, abs_pers, color=colors, edgecolor="white",
            linewidth=0.5, height=0.8)

    ax.set_xlabel("Persistence  $|$death $-$ birth$|$", fontsize=11)
    ax.set_title("Persistence barcode  ---  FE representation (logDTM, $k$-NN = 100)",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    ax.set_yticks([])

    # Annotations
    ax.text(abs_pers[0] * 0.55, 2,
            f"{n_modes} topological modes",
            fontsize=11, fontweight="bold", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdebd0",
                      edgecolor="#e67e22", alpha=0.9))

    ax.annotate(
        "Tukey HSD merge\n(iterative pairwise test)\n"
        r"$\Downarrow$" + "\n"
        "$K = 9$ final regimes",
        xy=(abs_pers[-1], n_modes - 1), xytext=(abs_pers[0] * 0.50, n_modes - 5),
        fontsize=10, color="#27ae60", fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#d5f5e3",
                  edgecolor="#27ae60", alpha=0.9),
        arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.5))

    ax.text(abs_pers[0] * 0.55, n_modes * 0.45,
            "Continuous density gradient\nin price space: no sharp gap\n"
            r"$\rightarrow$ Tukey HSD selects $K$",
            fontsize=9, color="#7f8c8d", style="italic", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#bdc3c7", alpha=0.8))

    fig.tight_layout()
    out = PAPER_DIR / "tda_barcode.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  Total modes: {n_modes}")


if __name__ == "__main__":
    make_single_window_diagram()
    make_tda_barcode()
