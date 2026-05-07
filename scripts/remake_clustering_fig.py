"""
Remake tda_clustering.png with better cluster visibility.
"""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from gudhi.clustering.tomato import Tomato

warnings.filterwarnings("ignore")

RESULTS = Path("results")

# Load
df_pca = pd.read_parquet(RESULTS / "exp_FE/step03/pca_embeddings.parquet")
E = df_pca.drop(columns=["datetime"]).values.astype(np.float32)

df_lab = pd.read_parquet(RESULTS / "exp_FE/step04/labels.parquet")
labels_final = df_lab["cluster"].values

pre = pd.read_parquet(RESULTS / "preprocessed.parquet")
pre["datetime"] = pd.to_datetime(pre["datetime"])
df_lab["datetime"] = pd.to_datetime(df_lab["datetime"])
merged = df_lab.merge(pre[["datetime", "lmp"]], on="datetime")
lmp = merged["lmp"].values

# Order
fe_order = merged.groupby("cluster")["lmp"].mean().sort_values().index.tolist()
fe_rank = {r: i for i, r in enumerate(fe_order)}
labels_ranked = np.array([fe_rank.get(l, -1) for l in labels_final])
K = len(fe_order)

# Density
t = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1, k=60)
t.fit(E)
density = t.weights_

# Use UMAP-style colors (tab10 for categorical)
cmap = plt.cm.tab10

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) Pre-merge: density landscape
ax = axes[0]
ax.set_facecolor("#1a1a2e")
sc = ax.scatter(E[:, 0], E[:, 1], c=density, cmap="inferno", s=8, alpha=0.7,
                edgecolors="none", rasterized=True)
cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("Log-density (logDTM)", fontsize=11, color="white")
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
ax.set_xlabel("DC 1", color="white", fontsize=12)
ax.set_ylabel("DC 2", color="white", fontsize=12)
ax.set_title("(a) Density landscape: 53 topological modes\n"
             "Bright = high density (cluster centers)",
             fontweight="bold", fontsize=13, color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")
ax.text(0.05, 0.95, "53 modes\npre-merge", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E65100", alpha=0.9))

# (b) Post-merge: 10 regimes with strong colors
ax = axes[1]
ax.set_facecolor("#f5f5f5")

regime_names = []
for i in range(K):
    mask = labels_ranked == i
    mean_lmp = lmp[mask].mean()
    regime_names.append(f"R{i} (${mean_lmp:.0f})")

# Plot each regime with large points
for i in range(K):
    mask = labels_ranked == i
    if mask.sum() > 0:
        color = cmap(i % 10)
        ax.scatter(E[mask, 0], E[mask, 1], s=12, alpha=0.6,
                   color=color, edgecolors="none",
                   label=regime_names[i], rasterized=True)

# Add centroid labels
for i in range(K):
    mask = labels_ranked == i
    if mask.sum() > 0:
        cx, cy = E[mask, 0].mean(), E[mask, 1].mean()
        ax.annotate(f"R{i}", (cx, cy), fontsize=9, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=cmap(i % 10), alpha=0.9, linewidth=1.5))

ax.set_xlabel("DC 1", fontsize=12)
ax.set_ylabel("DC 2", fontsize=12)
ax.set_title("(b) After Tukey merge: 10 regimes\n"
             "All pairwise distinct on LMP (ordered by price)",
             fontweight="bold", fontsize=13)
ax.legend(fontsize=8, ncol=2, loc="lower left", markerscale=2.5,
          framealpha=0.95, edgecolor="gray")
ax.text(0.05, 0.95, "10 regimes\npost-merge", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2E7D32", alpha=0.9))

fig.tight_layout(w_pad=3)
fig.savefig("paper/tda_clustering.png", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved tda_clustering.png")
