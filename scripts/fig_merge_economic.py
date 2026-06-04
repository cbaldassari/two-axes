"""
fig_merge_economic.py — Regenerate Figure 6: economic axis merge boxplot.
Same style as fig_merge_dynamic.py: labels as multi-line xtick labels.
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
lab_E = lab["regime_E"].values

# Sort regimes by LMP mean
e_ids = sorted(np.unique(lab_E))
e_means = {e: lmp_mean[lab_E == e].mean() for e in e_ids}
e_order = sorted(e_ids, key=lambda e: e_means[e])

# Color gradient: green (low price) to red (high price)
cmap = plt.cm.RdYlGn_r
colors = [cmap(i / (len(e_order) - 1)) for i in range(len(e_order))]

fig, ax = plt.subplots(figsize=(12, 5))

data = [lmp_mean[lab_E == e] for e in e_order]
bp = ax.boxplot(data, positions=range(len(e_order)), widths=0.6, patch_artist=True,
                medianprops=dict(color="red", lw=2))

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Multi-line xtick labels (same style as fig 7)
ax.set_xticks(range(len(e_order)))
tick_labels = []
for e in e_order:
    mask = lab_E == e
    n = mask.sum()
    mu = lmp_mean[mask].mean()
    tick_labels.append(f"E{e}\nn={n:,}\n${mu:.0f}")
ax.set_xticklabels(tick_labels, fontsize=8, linespacing=1.3)

ax.set_ylabel("LMP medio ($/MWh)", fontsize=11)
ax.set_xlabel("Regime economico (ordinato per LMP crescente)", fontsize=11, labelpad=35)
ax.set_title("9 regimi economici dopo merge Tukey HSD (da 48 modi iniziali)", fontsize=12)
ax.grid(alpha=0.15, axis="y")

fig.subplots_adjust(bottom=0.20)

out = PAPER / "merge_process.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close(fig)
