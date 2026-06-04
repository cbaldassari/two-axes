"""
fig_merge_dynamic.py — Regenerate Figure 7: dynamic axis merge boxplot.
Labels placed below each boxplot to avoid overlap.
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
pre = pd.read_parquet(OUT / "preprocessed.parquet")
r = pre["r"].values
W, S = 512, 6
starts = list(range(0, len(r) - W + 1, S))
N = len(starts)

acf6 = lab["acf_6h"].values
lab_D = lab["regime_D"].values

# Sort regimes by ACF mean
d_ids = sorted(np.unique(lab_D))
d_means = {d: acf6[lab_D == d].mean() for d in d_ids}
d_order = sorted(d_ids, key=lambda d: d_means[d])

# Color gradient: blue (low ACF) to red (high ACF)
cmap = plt.cm.coolwarm
colors = [cmap(i / (len(d_order) - 1)) for i in range(len(d_order))]

fig, ax = plt.subplots(figsize=(12, 5))

data = [acf6[lab_D == d] for d in d_order]
bp = ax.boxplot(data, positions=range(len(d_order)), widths=0.6, patch_artist=True,
                medianprops=dict(color="red", lw=2))

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Annotations: regime name + stats as xtick labels (multi-line)
ax.set_xticks(range(len(d_order)))
tick_labels = []
for d in d_order:
    mask = lab_D == d
    n = mask.sum()
    acf_mean = acf6[mask].mean()
    tick_labels.append(f"D{d}\nn={n:,}\nACF={acf_mean:.2f}")
ax.set_xticklabels(tick_labels, fontsize=8, linespacing=1.3)

ax.set_ylabel("ACF al lag 6h", fontsize=11)
ax.set_xlabel("Regime dinamico (persistenza crescente)", fontsize=11, labelpad=35)
ax.set_title("9 regimi dinamici dopo merge Tukey HSD (da 47 modi iniziali)", fontsize=12)
ax.grid(alpha=0.15, axis="y")

fig.subplots_adjust(bottom=0.20)

out = PAPER / "merge_process_dynamic.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close(fig)
