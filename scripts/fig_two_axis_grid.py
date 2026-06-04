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

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(lmp_mean, acf6, c=colors, s=4, alpha=0.35, edgecolors="none")

ax.axhline(med_acf, color="black", ls="--", lw=1.2, alpha=0.5, zorder=5)
ax.axvline(med_lmp, color="black", ls="--", lw=1.2, alpha=0.5, zorder=5)

# Quadrant populations
q_ll = ((lmp_mean < med_lmp) & (acf6 < med_acf)).sum()
q_lh = ((lmp_mean < med_lmp) & (acf6 >= med_acf)).sum()
q_rl = ((lmp_mean >= med_lmp) & (acf6 < med_acf)).sum()
q_rh = ((lmp_mean >= med_lmp) & (acf6 >= med_acf)).sum()
N = len(lmp_mean)

# Label positions: well inside the plot, using axes fraction
# Top-left (low price, high persistence)
ax.text(0.08, 0.88, f"Calmo persistente\n{100*q_lh/N:.0f}%",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, color="#2c3e50", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))

# Top-right (high price, high persistence)
ax.text(0.92, 0.88, f"Stress persistente\n{100*q_rh/N:.0f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#922b21", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))

# Bottom-left (low price, fast reversion)
ax.text(0.08, 0.12, f"Calmo transitorio\n{100*q_ll/N:.0f}%",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=9, color="#1a5276", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))

# Bottom-right (high price, fast reversion)
ax.text(0.92, 0.12, f"Spike rapido\n{100*q_rl/N:.0f}%",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#c0392b", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))

ax.set_xlabel("LMP medio ($/MWh)", fontsize=11)
ax.set_ylabel("ACF lag 6h", fontsize=11)
ax.set_title("Spazio dei regimi a due assi", fontsize=12, fontweight="bold")
ax.grid(alpha=0.15)

fig.tight_layout()
out = PAPER / "two_axis_grid.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")

# Also save to figures/ for README
out2 = Path("figures") / "two_axis_grid.png"
fig.savefig(out2, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out2}")
plt.close(fig)
