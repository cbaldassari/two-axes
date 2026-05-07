"""
Compare UMAP, t-SNE and PaCMAP projections to find the most cluster-friendly.
"""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import umap

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10})

RESULTS = Path("results")
SEED = 42

# Load
df_pca = pd.read_parquet(RESULTS / "exp_FE/step03/pca_embeddings.parquet")
E_dm = df_pca.drop(columns=["datetime"]).values.astype(np.float32)

df_lab = pd.read_parquet(RESULTS / "exp_FE/step04/labels.parquet")
labels = df_lab["cluster"].values

pre = pd.read_parquet(RESULTS / "preprocessed.parquet")
pre["datetime"] = pd.to_datetime(pre["datetime"])
df_lab["datetime"] = pd.to_datetime(df_lab["datetime"])
merged = df_lab.merge(pre[["datetime", "lmp"]], on="datetime")
lmp = merged["lmp"].values

fe_order = merged.groupby("cluster")["lmp"].mean().sort_values().index.tolist()
fe_rank = {r: i for i, r in enumerate(fe_order)}
labels_ranked = np.array([fe_rank.get(l, -1) for l in labels])
K = len(fe_order)

cmap = plt.cm.tab10

# Projections
print("UMAP (n_neighbors=30, min_dist=0.3)...")
proj_umap = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        random_state=SEED).fit_transform(E_dm)

print("UMAP (n_neighbors=50, min_dist=0.5) — more globular...")
proj_umap2 = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.5,
                         random_state=SEED).fit_transform(E_dm)

print("t-SNE (perplexity=30)...")
proj_tsne = TSNE(n_components=2, perplexity=30, random_state=SEED,
                  max_iter=1000, init="pca").fit_transform(E_dm)

print("t-SNE (perplexity=50) — more global...")
proj_tsne2 = TSNE(n_components=2, perplexity=50, random_state=SEED,
                   max_iter=1000, init="pca").fit_transform(E_dm)

projections = [
    ("UMAP (nn=30, md=0.3)", proj_umap),
    ("UMAP (nn=50, md=0.5)", proj_umap2),
    ("t-SNE (perp=30)", proj_tsne),
    ("t-SNE (perp=50)", proj_tsne2),
]

# Plot comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for col, (name, proj) in enumerate(projections):
    # Top row: colored by regime
    ax = axes[0, col]
    for i in range(K):
        mask = labels_ranked == i
        mean_lmp = lmp[mask].mean()
        ax.scatter(proj[mask, 0], proj[mask, 1], s=5, alpha=0.5,
                   color=cmap(i), edgecolors="none", rasterized=True)
    # Centroids
    for i in range(K):
        mask = labels_ranked == i
        if mask.sum() > 0:
            cx, cy = proj[mask, 0].mean(), proj[mask, 1].mean()
            ax.annotate(f"R{i}", (cx, cy), fontsize=8, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                  ec=cmap(i), alpha=0.85, lw=1.5))
    ax.set_title(name, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Colored by regime")

    # Bottom row: colored by LMP
    ax = axes[1, col]
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=lmp, cmap="RdYlGn_r",
                    s=5, alpha=0.5, edgecolors="none", vmin=20, vmax=200,
                    rasterized=True)
    ax.set_title(name, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Colored by LMP")
    if col == 3:
        plt.colorbar(sc, ax=ax, shrink=0.8, label="LMP ($/MWh)")

fig.suptitle("Projection comparison: which shows clusters best?",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("paper/projection_comparison.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("Saved paper/projection_comparison.png")
