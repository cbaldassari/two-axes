"""
remake_tda_figures.py — Regenerate TDA figures with better readability.
"""
import sys, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from gudhi.clustering.tomato import Tomato

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

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

# Re-run ToMATo for persistence data
k = 60
t = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1, k=k)
t.fit(E)
diagram = np.array(t.diagram_)
pers = np.sort(diagram[:, 1] - diagram[:, 0])[::-1]
density = t.weights_

# Order final regimes by LMP
fe_order = merged.groupby("cluster")["lmp"].mean().sort_values().index.tolist()
fe_rank = {r: i for i, r in enumerate(fe_order)}
labels_ranked = np.array([fe_rank.get(l, -1) for l in labels_final])
K = len(fe_order)

# ===================================================================
# FIGURE 2: TDA explanation (persistence + barcode, no density panel)
# ===================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) Persistence diagram — bigger points, annotations
ax = axes[0]
births = diagram[:, 0]
deaths = diagram[:, 1]
persistence = deaths - births
# Color by persistence
colors = np.where(np.isin(np.argsort(np.argsort(-persistence)),
                           range(10)), "#E65100", "#90CAF9")
sizes = np.where(np.isin(np.argsort(np.argsort(-persistence)),
                          range(10)), 60, 15)
ax.scatter(births, deaths, s=sizes, c=colors, alpha=0.7, edgecolors="black",
           linewidth=0.3, zorder=3)
mn = min(births.min(), deaths.min()) - 0.5
mx = max(births.max(), deaths.max()) + 0.5
ax.plot([mn, mx], [mn, mx], "--", color="gray", lw=0.8, zorder=1)
ax.set_xlabel("Birth (density level)")
ax.set_ylabel("Death (density level)")
ax.set_title("(a) Persistence diagram", fontweight="bold")
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
ax.set_aspect("equal")
# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E65100",
           markersize=10, label="10 persistent modes (regimes)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#90CAF9",
           markersize=7, label="Noise (merged)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
# Arrow to show "far from diagonal = persistent"
ax.annotate("High persistence\n(genuine regimes)",
            xy=(births[np.argmax(persistence)], deaths[np.argmax(persistence)]),
            xytext=(births.min() + 2, deaths.max() - 2),
            fontsize=8, fontstyle="italic", color="#E65100",
            arrowprops=dict(arrowstyle="->", color="#E65100", lw=1))

# (b) Persistence barcode — thicker bars, clearer gap
ax = axes[1]
n_show = min(30, len(pers))
colors_bar = ["#E65100" if i < 10 else "#BBBBBB" for i in range(n_show)]
bars = ax.barh(range(n_show), pers[:n_show], color=colors_bar, height=0.7,
               edgecolor="white", linewidth=0.3)
# Gap line
ax.axhline(9.5, color="red", ls="--", lw=2, alpha=0.8)
ax.text(0.5, 9.5 - 0.6, "GAP → K = 10 regimes",
        fontsize=10, color="red", fontweight="bold", ha="center",
        transform=ax.get_yaxis_transform())
ax.set_xlabel("Persistence (death − birth)")
ax.set_ylabel("Mode rank")
ax.set_title("(b) Persistence barcode", fontweight="bold")
ax.invert_yaxis()
ax.set_yticks(range(0, n_show, 5))

fig.tight_layout()
fig.savefig("paper/tda_explanation.png", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved tda_explanation.png")

# ===================================================================
# FIGURE 3: Before/after Tukey merge
# ===================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# (a) Before merge: color by DENSITY (not 53 clusters — unreadable)
ax = axes2[0]
sc = ax.scatter(E[:, 0], E[:, 1], c=density, cmap="plasma", s=4, alpha=0.5,
                edgecolors="none", rasterized=True)
cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Log-density", fontsize=10)
ax.set_xlabel("DC 1")
ax.set_ylabel("DC 2")
ax.set_title("(a) Density landscape with 53 topological modes\n"
             "(many modes, similar prices → need merging)", fontweight="bold")
# Add text showing N modes
ax.text(0.05, 0.95, "53 modes\n(pre-merge)", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", color="white",
        bbox=dict(boxstyle="round", facecolor="#E65100", alpha=0.8))

# (b) After merge: 10 regimes, large points, ordered colors
ax = axes2[1]
cmap = plt.cm.RdYlGn_r
colors_regime = cmap(np.linspace(0.1, 0.9, K))

for i in range(K):
    mask = labels_ranked == i
    if mask.sum() > 0:
        mean_lmp = lmp[mask].mean()
        ax.scatter(E[mask, 0], E[mask, 1], s=6, alpha=0.5,
                   color=colors_regime[i], edgecolors="none",
                   label=f"R{i} (${mean_lmp:.0f})", rasterized=True)

ax.set_xlabel("DC 1")
ax.set_ylabel("DC 2")
ax.set_title("(b) After Tukey merge: 10 regimes\n"
             "(all pairwise distinct on LMP, ordered by price)",
             fontweight="bold")
ax.legend(fontsize=7, ncol=2, loc="lower left", markerscale=3,
          framealpha=0.9)
ax.text(0.05, 0.95, "10 regimes\n(post-merge)", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", color="white",
        bbox=dict(boxstyle="round", facecolor="#2E7D32", alpha=0.8))

fig2.tight_layout()
fig2.savefig("paper/tda_clustering.png", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved tda_clustering.png")
