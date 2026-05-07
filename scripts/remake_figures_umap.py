"""
Remake TDA figures using UMAP for 2D visualization.
UMAP is used ONLY for plotting — clustering was done on Diffusion Maps 11D.
"""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from gudhi.clustering.tomato import Tomato
import umap

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

RESULTS = Path("results")
SEED = 42

# ── Load ──
df_pca = pd.read_parquet(RESULTS / "exp_FE/step03/pca_embeddings.parquet")
E_dm = df_pca.drop(columns=["datetime"]).values.astype(np.float32)  # 11D DiffMaps

df_lab = pd.read_parquet(RESULTS / "exp_FE/step04/labels.parquet")
labels_final = df_lab["cluster"].values

pre = pd.read_parquet(RESULTS / "preprocessed.parquet")
pre["datetime"] = pd.to_datetime(pre["datetime"])
df_lab["datetime"] = pd.to_datetime(df_lab["datetime"])
merged = df_lab.merge(pre[["datetime", "lmp"]], on="datetime")
lmp = merged["lmp"].values

# Order regimes by LMP
fe_order = merged.groupby("cluster")["lmp"].mean().sort_values().index.tolist()
fe_rank = {r: i for i, r in enumerate(fe_order)}
labels_ranked = np.array([fe_rank.get(l, -1) for l in labels_final])
K = len(fe_order)

# Density from ToMATo
print("Running ToMATo for density...")
t = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1, k=60)
t.fit(E_dm)
density = t.weights_
diagram = np.array(t.diagram_)
pers = np.sort(diagram[:, 1] - diagram[:, 0])[::-1]

# ── UMAP projection (for visualization only) ──
print("Computing UMAP projection...")
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                     metric="euclidean", random_state=SEED)
E_umap = reducer.fit_transform(E_dm)
print(f"UMAP done: {E_umap.shape}")

# ═══════════════════════════════════════════════════════
# FIGURE 2: TDA explanation (density + persistence + barcode)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# (a) Density landscape on UMAP
ax = axes[0]
ax.set_facecolor("#1a1a2e")
sc = ax.scatter(E_umap[:, 0], E_umap[:, 1], c=density, cmap="inferno",
                s=5, alpha=0.7, edgecolors="none", rasterized=True)
cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Log-density (logDTM)", fontsize=10, color="white")
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
ax.set_xlabel("UMAP 1", color="white", fontsize=11)
ax.set_ylabel("UMAP 2", color="white", fontsize=11)
ax.set_title("(a) Density landscape\n(UMAP projection for visualization)",
             fontweight="bold", color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

# (b) Persistence diagram
ax = axes[1]
births = diagram[:, 0]
deaths = diagram[:, 1]
persistence = deaths - births
rank_order = np.argsort(-persistence)
is_top10 = np.zeros(len(persistence), dtype=bool)
is_top10[rank_order[:10]] = True

colors_pts = np.where(is_top10, "#E65100", "#90CAF9")
sizes_pts = np.where(is_top10, 80, 20)

ax.scatter(births, deaths, s=sizes_pts, c=colors_pts, alpha=0.7,
           edgecolors="black", linewidth=0.3, zorder=3)
mn = min(births.min(), deaths.min()) - 0.5
mx = max(births.max(), deaths.max()) + 0.5
ax.plot([mn, mx], [mn, mx], "--", color="gray", lw=0.8, zorder=1)
ax.set_xlabel("Birth (density level)")
ax.set_ylabel("Death (density level)")
ax.set_title("(b) Persistence diagram", fontweight="bold")
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
ax.set_aspect("equal")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E65100",
           markersize=10, label="10 persistent modes (regimes)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#90CAF9",
           markersize=7, label="Noise (merged)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
ax.annotate("High persistence\n(genuine regimes)",
            xy=(births[rank_order[0]], deaths[rank_order[0]]),
            xytext=(births.min() + 2, deaths.max() - 2),
            fontsize=8, fontstyle="italic", color="#E65100",
            arrowprops=dict(arrowstyle="->", color="#E65100", lw=1))

# (c) Persistence barcode
ax = axes[2]
n_show = min(30, len(pers))
colors_bar = ["#E65100" if i < 10 else "#BBBBBB" for i in range(n_show)]
ax.barh(range(n_show), pers[:n_show], color=colors_bar, height=0.7,
        edgecolor="white", linewidth=0.3)
ax.axhline(9.5, color="red", ls="--", lw=2, alpha=0.8)
ax.text(pers[0] * 0.5, 9.5 + 0.6, "GAP \u2192 K = 10 regimes",
        fontsize=10, color="red", fontweight="bold")
ax.set_xlabel("Persistence (death \u2212 birth)")
ax.set_ylabel("Mode rank")
ax.set_title("(c) Persistence barcode", fontweight="bold")
ax.invert_yaxis()
ax.set_yticks(range(0, n_show, 5))

fig.tight_layout()
fig.savefig("paper/tda_explanation.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("Saved tda_explanation.png")


# ═══════════════════════════════════════════════════════
# FIGURE 3: Before/after Tukey merge (UMAP)
# ═══════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

# (a) Pre-merge: density on UMAP
ax = axes2[0]
ax.set_facecolor("#1a1a2e")
sc = ax.scatter(E_umap[:, 0], E_umap[:, 1], c=density, cmap="inferno",
                s=8, alpha=0.7, edgecolors="none", rasterized=True)
cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("Log-density", fontsize=10, color="white")
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
ax.set_xlabel("UMAP 1", color="white", fontsize=12)
ax.set_ylabel("UMAP 2", color="white", fontsize=12)
ax.set_title("(a) 53 topological modes (pre-merge)\n"
             "Bright peaks = density modes, many share similar prices",
             fontweight="bold", fontsize=12, color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")
ax.text(0.03, 0.97, "53 modes", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E65100", alpha=0.9))

# (b) Post-merge: 10 regimes on UMAP
ax = axes2[1]
ax.set_facecolor("#f0f0f0")
cmap10 = plt.cm.tab10

for i in range(K):
    mask = labels_ranked == i
    if mask.sum() > 0:
        mean_lmp = lmp[mask].mean()
        ax.scatter(E_umap[mask, 0], E_umap[mask, 1], s=10, alpha=0.6,
                   color=cmap10(i), edgecolors="none",
                   label=f"R{i} (${mean_lmp:.0f})", rasterized=True)

# Centroid labels
for i in range(K):
    mask = labels_ranked == i
    if mask.sum() > 0:
        cx = E_umap[mask, 0].mean()
        cy = E_umap[mask, 1].mean()
        ax.annotate(f"R{i}", (cx, cy), fontsize=10, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=cmap10(i), alpha=0.9, linewidth=2))

ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title("(b) 10 regimes after Tukey merge\n"
             "All pairwise distinct on LMP, colored by price",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=8, ncol=2, loc="lower left", markerscale=2.5,
          framealpha=0.95, edgecolor="gray")
ax.text(0.03, 0.97, "10 regimes", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2E7D32", alpha=0.9))

fig2.tight_layout(w_pad=3)
fig2.savefig("paper/tda_clustering.png", dpi=300, bbox_inches="tight",
             facecolor="white")
print("Saved tda_clustering.png")
