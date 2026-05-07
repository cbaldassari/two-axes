"""
bootstrap_ari.py
================
Bootstrap ARI stability using the REAL pipeline functions.
20 runs x 80% subsample for each of the 3 ToMATo configs.

Calls step03 (DiffMaps) and step04 (GUDHI ToMATo + Tukey) logic directly.

Output: results/bootstrap_ari.json
"""

import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from gudhi.clustering.tomato import Tomato
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
N_BOOT = 20
FRAC = 0.80
SEED = 42
KNN_GRID = [20, 40, 60, 80, 100, 150]


# ---- DiffMaps (from step03_pca.py) ----

def compute_diffusion_maps(E, n_comp, seed=42):
    X = StandardScaler().fit_transform(E).astype(np.float64)
    n = X.shape[0]

    k_nn = max(15, int(np.sqrt(n)))
    nn = NearestNeighbors(n_neighbors=k_nn, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    sigma = float(np.median(distances[:, -1]))
    epsilon = sigma ** 2

    rows = np.repeat(np.arange(n), k_nn)
    cols = indices.ravel()
    vals = np.exp(-(distances ** 2).ravel() / epsilon)
    K = csr_matrix((vals, (rows, cols)), shape=(n, n))
    K = (K + K.T) / 2.0

    d = np.array(K.sum(axis=1)).flatten()
    D_inv = diags(np.power(d, -1.0))
    K_a = D_inv @ K @ D_inv
    d2 = np.array(K_a.sum(axis=1)).flatten()
    P = diags(1.0 / d2) @ K_a

    n_eig = min(max(n_comp + 5, 30), n - 2)
    np.random.seed(seed)
    v0 = np.random.randn(n)
    eigenvalues, eigenvectors = eigsh(P, k=n_eig + 1, which="LM", v0=v0)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][1:]
    eigenvectors = eigenvectors[:, idx][:, 1:]

    coords = eigenvectors[:, :n_comp] * eigenvalues[:n_comp][np.newaxis, :]
    return coords.astype(np.float32)


# ---- ToMATo grid search (from step04_tomato.py) ----

def run_tomato(E):
    n = E.shape[0]
    results = []
    for k in KNN_GRID:
        t = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1, k=k)
        t.fit(E)
        diagram = np.array(t.diagram_) if len(t.diagram_) > 0 else np.empty((0, 2))
        if diagram.shape[0] > 0:
            pers = np.sort(diagram[:, 1] - diagram[:, 0])[::-1]
        else:
            pers = np.array([])

        if len(pers) >= 2:
            gaps = pers[:-1] - pers[1:]
            gap_idx = int(np.argmax(gaps))
            k_auto = gap_idx + 2
            gap_size = float(gaps[gap_idx])
        else:
            k_auto = max(1, len(pers) + 1)
            gap_size = 0.0

        med = float(np.median(np.abs(pers))) if len(pers) > 0 else 1.0
        gap_norm = gap_size / max(med, 1e-12)
        results.append({"k": k, "k_auto": k_auto, "gap_norm": gap_norm, "tomato": t})

    best = max(results, key=lambda r: r["gap_norm"])
    tomato = best["tomato"]
    tomato.n_clusters_ = best["k_auto"]
    labels = np.asarray(tomato.labels_)
    return labels


# ---- Tukey HSD merge (from step04_tomato.py) ----

def tukey_merge(labels, lmp, alpha=0.05):
    valid = ~np.isnan(lmp)
    labels_v = labels[valid].copy()
    lmp_v = lmp[valid]

    while True:
        unique_labels = np.unique(labels_v)
        if len(unique_labels) <= 2:
            break

        result = pairwise_tukeyhsd(lmp_v, labels_v, alpha=alpha)
        pairs = list(combinations(sorted(unique_labels), 2))

        best_p = -1.0
        best_pair = None
        for i in range(min(len(result.reject), len(pairs))):
            if not result.reject[i]:
                if result.pvalues[i] > best_p:
                    best_p = result.pvalues[i]
                    best_pair = pairs[i]

        if best_pair is None:
            break
        else:
            a, b = best_pair
            labels_v[labels_v == b] = a

    unique_final = np.unique(labels_v)
    remap = {old: new for new, old in enumerate(unique_final)}
    labels_out = np.full(len(labels), -1, dtype=int)
    labels_out[valid] = np.array([remap[l] for l in labels_v])
    return labels_out


# ---- PCA denoise for high-D inputs (MOMENT) ----

def pca_denoise(E, seed=42):
    from sklearn.decomposition import PCA
    from utils.pca_dim_selector import marchenko_pastur_n_components

    X = StandardScaler().fit_transform(E)
    n, p = X.shape
    pca_full = PCA(random_state=seed).fit(X)
    n_comp_mp, _ = marchenko_pastur_n_components(pca_full.explained_variance_, n, p)
    pca = PCA(n_components=n_comp_mp, random_state=seed)
    return pca.fit_transform(X)


# ---- Main ----

def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    # Precompute window LMP
    prep = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    prep["datetime"] = pd.to_datetime(prep["datetime"])
    lmp_lookup = prep.set_index("datetime")["lmp"]

    CONFIGS = [
        {"name": "A:MOMENT+ToMATo", "exp": "exp_C", "n_comp": 2, "denoise": True},
        {"name": "B:FE+ToMATo", "exp": "exp_FE", "n_comp": 11, "denoise": False},
        {"name": "C:COMBO+ToMATo", "exp": "exp_COMBO", "n_comp": 3, "denoise": False},
    ]

    results = {}
    all_runs = []

    for cfg in CONFIGS:
        print(f"\n{'=' * 55}")
        print(f"  {cfg['name']}  (n_comp={cfg['n_comp']})")
        print(f"{'=' * 55}", flush=True)

        # Load embeddings + timestamps
        emb = pd.read_parquet(RESULTS_DIR / cfg["exp"] / "embeddings.parquet")
        ts_col = [c for c in emb.columns if "time" in c.lower() or "date" in c.lower()][0]
        timestamps = pd.to_datetime(emb[ts_col])
        emb_cols = [c for c in emb.columns if c != ts_col]
        emb_vals = emb[emb_cols].values.astype(np.float32)

        # Load full-sample labels
        full_labels_df = pd.read_parquet(
            RESULTS_DIR / cfg["exp"] / "step04" / "labels.parquet"
        )
        full_labels = full_labels_df["cluster"].values
        n = len(emb_vals)

        # Window LMP
        lmp_per_window = lmp_lookup.reindex(timestamps).values

        aris = []
        ks = []
        for b in range(N_BOOT):
            seed_b = SEED + b
            idx = rng.choice(n, size=int(n * FRAC), replace=False)
            idx.sort()

            X_sub = emb_vals[idx]

            # PCA denoise for high-D
            if cfg["denoise"] and X_sub.shape[1] > 100:
                X_sub = pca_denoise(X_sub, seed=seed_b)

            # Diffusion Maps
            coords = compute_diffusion_maps(X_sub, n_comp=cfg["n_comp"], seed=seed_b)

            # ToMATo
            sub_labels = run_tomato(coords)

            # Tukey merge
            lmp_sub = lmp_per_window[idx]
            merged_labels = tukey_merge(sub_labels, lmp_sub)

            # ARI vs full-sample labels (subsample only)
            full_sub = full_labels[idx]
            valid = merged_labels >= 0
            ari = adjusted_rand_score(full_sub[valid], merged_labels[valid])
            K_final = len(np.unique(merged_labels[valid]))

            aris.append(ari)
            ks.append(K_final)
            print(f"  boot {b + 1:2d}/{N_BOOT}: K={K_final:2d}, ARI={ari:.3f}", flush=True)

        mean_ari = float(np.mean(aris))
        std_ari = float(np.std(aris))
        mean_k = float(np.mean(ks))
        results[cfg["name"]] = {
            "mean": round(mean_ari, 4),
            "std": round(std_ari, 4),
        }
        # Save per-run data
        all_runs.append(pd.DataFrame({
            "config": cfg["name"],
            "boot": range(N_BOOT),
            "ARI": aris,
            "K": ks,
        }))
        print(f"  => ARI = {mean_ari:.4f} +/- {std_ari:.4f}, K_mean = {mean_k:.1f}")

    # Save summary JSON
    out_path = RESULTS_DIR / "bootstrap_ari.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Save per-run CSV
    df_runs = pd.concat(all_runs, ignore_index=True)
    runs_path = RESULTS_DIR / "bootstrap_ari_runs.csv"
    df_runs.to_csv(runs_path, index=False, float_format="%.4f")
    print(f"Saved: {runs_path}")

    # Generate figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config_names = ["Level\n(FE)", "Memory\n(MOMENT)", "Combo"]
    config_keys = ["B:FE+ToMATo", "A:MOMENT+ToMATo", "C:COMBO+ToMATo"]
    colors = ["#b2182b", "#2166ac", "#1b7837"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")

    # Panel (a): ARI strip plot + box
    ax = axes[0]
    for i, (key, name, color) in enumerate(zip(config_keys, config_names, colors)):
        sub = df_runs[df_runs["config"] == key]
        jitter = np.random.RandomState(42).normal(0, 0.05, len(sub))
        ax.scatter(np.full(len(sub), i) + jitter, sub["ARI"],
                   alpha=0.6, s=30, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.boxplot(sub["ARI"].values, positions=[i], widths=0.35,
                   patch_artist=True, boxprops=dict(facecolor=color, alpha=0.3),
                   medianprops=dict(color="black", linewidth=1.5),
                   whiskerprops=dict(color=color), capprops=dict(color=color),
                   flierprops=dict(marker=""), zorder=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(config_names, fontsize=9)
    ax.set_ylabel("Bootstrap ARI", fontsize=10)
    ax.set_title("(a) Bootstrap ARI (20 runs, 80% subsample)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    # Panel (b): K distribution
    ax2 = axes[1]
    for i, (key, name, color) in enumerate(zip(config_keys, config_names, colors)):
        sub = df_runs[df_runs["config"] == key]
        jitter = np.random.RandomState(42).normal(0, 0.05, len(sub))
        ax2.scatter(np.full(len(sub), i) + jitter, sub["K"],
                    alpha=0.6, s=30, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax2.boxplot(sub["K"].values, positions=[i], widths=0.35,
                    patch_artist=True, boxprops=dict(facecolor=color, alpha=0.3),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color=color), capprops=dict(color=color),
                    flierprops=dict(marker=""), zorder=2)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(config_names, fontsize=9)
    ax2.set_ylabel("K (number of regimes)", fontsize=10)
    ax2.set_title("(b) Number of regimes across bootstrap runs", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.2)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    fig_path = Path("paper") / "bootstrap_stability.png"
    fig.savefig(fig_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    elapsed = time.time() - t0
    print(f"Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
