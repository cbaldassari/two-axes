"""
fig_two_axis_grid.py — Regenerate the two-axis scatter plot (Figure 12).
Scatter of LMP mean vs ACF 6h, colored by economic regime,
with quadrant labels positioned well inside the plot area.
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("results_darcsinh/split_W512_S6")
PAPER = Path("paper")

lab = pd.read_parquet(OUT / "labels.parquet")
lmp_mean = lab["lmp_mean"].values
acf6 = lab["acf_6h"].values
lab_E = lab["regime_E"].values

# Sort economic regimes by LMP mean for consistent coloring
e_order = np.argsort([lmp_mean[lab_E == e].mean() for e in range(9)])
color_map = {}
cmap = plt.cm.tab10
for rank, e in enumerate(e_order):
    color_map[e] = cmap(rank)

colors = [color_map[e] for e in lab_E]

# Quadrant thresholds (medians)
med_lmp = np.median(lmp_mean)
med_acf = np.median(acf6)

def make_figure(lang="it"):
    is_en = lang == "en"

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter with legend entries per E regime (ordered E0..E8)
    e_labels = {e: f"E{e}" for e in range(9)}
    for e in range(9):
        mask = lab_E == e
        ax.scatter(lmp_mean[mask], acf6[mask], c=[color_map[e]], s=8, alpha=0.4,
                   edgecolors="none", label=e_labels[e])

    ax.axhline(med_acf, color="black", ls="--", lw=1.2, alpha=0.25, zorder=5)
    ax.axvline(med_lmp, color="black", ls="--", lw=1.2, alpha=0.25, zorder=5)

    # Quadrant populations
    q_ll = ((lmp_mean < med_lmp) & (acf6 < med_acf)).sum()
    q_lh = ((lmp_mean < med_lmp) & (acf6 >= med_acf)).sum()
    q_rl = ((lmp_mean >= med_lmp) & (acf6 < med_acf)).sum()
    q_rh = ((lmp_mean >= med_lmp) & (acf6 >= med_acf)).sum()
    N = len(lmp_mean)

    if is_en:
        labels = ["Calm persistent", "Persistent stress", "Calm transitory", "Fast spike"]
        xlabel = "LMP mean ($/MWh)"
        ylabel = "ACF at lag 6h"
        title = "Two-axis regime space"
    else:
        labels = ["Calmo persistente", "Stress persistente", "Calmo transitorio", "Spike rapido"]
        xlabel = "LMP medio ($/MWh)"
        ylabel = "ACF lag 6h"
        title = "Spazio dei regimi a due assi"

    # Quadrant labels — snapped to inner edges of the plot
    fs = 13
    # Top-left (low price, high persistence)
    ax.text(0.02, 0.98, f"{labels[0]} ({100*q_lh/N:.0f}%)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=fs, color="#2c3e50", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"))

    # Top-right (high price, high persistence)
    ax.text(0.98, 0.98, f"{labels[1]} ({100*q_rh/N:.0f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=fs, color="#922b21", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"))

    # Bottom-left (low price, fast reversion)
    ax.text(0.02, 0.02, f"{labels[2]} ({100*q_ll/N:.0f}%)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=fs, color="#1a5276", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"))

    # Bottom-right (high price, fast reversion)
    ax.text(0.98, 0.02, f"{labels[3]} ({100*q_rl/N:.0f}%)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=fs, color="#c0392b", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.1)

    # Legend below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=9,
              fontsize=10, markerscale=3, frameon=False)

    fig.tight_layout()
    return fig

# Generate IT
fig_it = make_figure("it")
out_it = PAPER / "two_axis_grid.png"
fig_it.savefig(out_it, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_it}")
plt.close(fig_it)

# Generate EN
fig_en = make_figure("en")
out_en = Path("paper/en") / "two_axis_grid.png"
fig_en.savefig(out_en, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_en}")

# Also save EN to figures/ for README
out_fig = Path("figures") / "two_axis_grid.png"
fig_en.savefig(out_fig, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_fig}")
plt.close(fig_en)
