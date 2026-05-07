"""
step04_ward.py
==============
NEPOOL TDA Pipeline -- Step 04 (Ward baseline): Hierarchical Clustering + Tukey HSD Merge

Ward linkage on diffusion coordinates from step03. K selected from dendrogram
via inconsistency criterion, then Tukey HSD merge to fuse non-distinct regimes.

Same input/output interface as step04_tomato.py for direct comparison.

Input  : results/exp_{X}/step03/pca_embeddings.parquet
Output : results/exp_{X}/step04_ward/
           labels.parquet
           ward_report.json
           01_dendrogram.png, 02_cluster_sizes.png, 03_timeline.png

Uso:
  python pipeline/step04_ward.py --exp C
  python pipeline/step04_ward.py --exp FE
  python pipeline/step04_ward.py --exp COMBO
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

_parser = argparse.ArgumentParser(description="Step 04 Ward baseline")
_parser.add_argument("--exp", default="C",
                     choices=["A", "B", "C", "D", "E", "FE", "FT", "COMBO", "D2", "ILR"])
_parser.add_argument("--n-clusters", type=int, default=None,
                     help="Force K (default: auto via inconsistency)")
_args = _parser.parse_args()
EXPERIMENT = _args.exp

RESULTS_DIR = Path("results")
PCA_PATH    = RESULTS_DIR / f"exp_{EXPERIMENT}" / "step03" / "pca_embeddings.parquet"
OUT_DIR     = RESULTS_DIR / f"exp_{EXPERIMENT}" / "step04_ward"
SEED        = 42
DPI         = 150
CMAP        = plt.get_cmap("tab20")


# =====================================================================
#  Load
# =====================================================================

def load_features():
    df = pd.read_parquet(PCA_PATH)
    ts_col = "datetime" if "datetime" in df.columns else "date"
    timestamps = pd.to_datetime(df[ts_col])
    E = df.drop(columns=[ts_col]).values.astype(np.float32)
    print(f"  Loaded: {E.shape}", flush=True)
    return E, timestamps


# =====================================================================
#  Ward clustering
# =====================================================================

def run_ward(E, n_clusters_forced=None):
    """
    Ward hierarchical clustering.
    K selected via inconsistency criterion if not forced.
    """
    n, d = E.shape
    print(f"\n  Ward linkage ({n} points, {d}D) ...", flush=True)

    t0 = time.time()
    Z = linkage(E, method="ward", metric="euclidean")
    elapsed = time.time() - t0
    print(f"  Linkage done ({elapsed:.1f}s)", flush=True)

    if n_clusters_forced:
        K = n_clusters_forced
        print(f"  K forced: {K}", flush=True)
    else:
        # Inconsistency criterion: find the cut with largest gap in merge distances
        # The merge distances are Z[:, 2]. Look for the biggest jump.
        merge_dists = Z[:, 2]
        diffs = np.diff(merge_dists)

        # Search in the last 5% of merges (where the big jumps happen)
        search_start = max(0, len(diffs) - int(0.05 * len(diffs)))
        best_idx = search_start + np.argmax(diffs[search_start:])

        # K = n - (best_idx + 1) because cutting after merge best_idx+1
        # leaves n - (best_idx + 1) clusters... actually:
        # Z has n-1 rows. Row i merges two clusters. After row i, there are n-1-i clusters.
        # We cut just before the big jump, so K = n - best_idx - 1
        K = n - best_idx - 1
        K = max(2, min(K, 30))  # reasonable bounds

        print(f"  Inconsistency -> K={K} (gap at merge {best_idx+1}/{len(diffs)},"
              f" dist jump {diffs[best_idx]:.4f})", flush=True)

    labels = fcluster(Z, t=K, criterion="maxclust") - 1  # 0-indexed

    unique, counts = np.unique(labels, return_counts=True)
    print(f"  K={len(unique)} clusters:", flush=True)
    for u, c in zip(unique, counts):
        print(f"    R{u}: {c} ({c/n*100:.1f}%)", flush=True)

    info = {
        "method": "ward",
        "n_clusters_raw": int(len(unique)),
        "input_shape": [int(n), int(d)],
        "cluster_sizes_raw": {int(u): int(c) for u, c in zip(unique, counts)},
    }

    return labels, Z, info


# =====================================================================
#  Tukey HSD Merge (same as step04_tomato)
# =====================================================================

def tukey_merge(labels, timestamps, alpha=0.05):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    print(f"\n  Tukey HSD merge (alpha={alpha}) ...", flush=True)

    pre = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    lmp_lookup = pre.set_index("datetime")["lmp"]

    ts = pd.to_datetime(timestamps)
    lmp = lmp_lookup.reindex(ts).values

    valid = ~np.isnan(lmp)
    labels_v = labels[valid].copy()
    lmp_v = lmp[valid]

    k_before = len(np.unique(labels_v))
    print(f"    K before: {k_before}", flush=True)

    merged = True
    iteration = 0
    while merged:
        iteration += 1
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
            merged = False
        else:
            a, b = best_pair
            labels_v[labels_v == b] = a
            k_now = len(np.unique(labels_v))
            print(f"    Iter {iteration}: merge {b}->{a} (p={best_p:.4f}) K={k_now}",
                  flush=True)

    unique_final = np.unique(labels_v)
    remap = {old: new for new, old in enumerate(unique_final)}
    labels_out = np.full(len(labels), -1, dtype=int)
    labels_out[valid] = np.array([remap[l] for l in labels_v])

    k_after = len(unique_final)
    print(f"    K: {k_before} -> {k_after}", flush=True)

    unique_m, counts_m = np.unique(labels_out[labels_out >= 0], return_counts=True)
    for u, c in zip(unique_m, counts_m):
        print(f"      R{u}: {c} ({c/valid.sum()*100:.1f}%)", flush=True)

    return labels_out, {
        "k_before_merge": k_before,
        "k_after_merge": k_after,
        "cluster_sizes_merged": {int(u): int(c) for u, c in zip(unique_m, counts_m)},
    }


# =====================================================================
#  Plots
# =====================================================================

def make_plots(E, labels, timestamps, Z):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": DPI, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    valid = labels >= 0
    K = len(np.unique(labels[valid]))
    colors = [CMAP(i % 20) for i in range(K)]

    # 01: Dendrogram (last 50 merges)
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode="lastp", p=50, ax=ax,
               leaf_rotation=90, leaf_font_size=8,
               color_threshold=0)
    ax.set_title(f"Ward dendrogram (last 50 merges) -- {EXPERIMENT}")
    ax.set_ylabel("Merge distance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_dendrogram.png")
    plt.close(fig)

    # 02: Cluster sizes
    unique, counts = np.unique(labels[valid], return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(unique)), counts, color=[colors[u] for u in unique])
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([f"R{u}" for u in unique])
    ax.set_ylabel("Windows"); ax.set_title(f"Ward cluster sizes (K={K})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_cluster_sizes.png")
    plt.close(fig)

    # 03: Timeline
    if E.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(14, 3))
        dt = timestamps.values
        for k in range(K):
            mask = labels == k
            ax.scatter(dt[mask], np.full(mask.sum(), k), s=2, color=colors[k], alpha=0.6)
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"R{k}" for k in range(K)])
        ax.set_title("Ward regime timeline")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "03_timeline.png")
        plt.close(fig)

    # 04: 2D scatter
    if E.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        for k in range(K):
            mask = labels == k
            ax.scatter(E[mask, 0], E[mask, 1], s=3, alpha=0.4, color=colors[k], label=f"R{k}")
        ax.legend(markerscale=5, fontsize=7, ncol=2)
        ax.set_title(f"Ward clusters (K={K})")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "04_cluster_scatter.png")
        plt.close(fig)

    print(f"  Plots saved -> {OUT_DIR}/", flush=True)


# =====================================================================
#  Main
# =====================================================================

def main():
    t0 = time.time()
    print("=" * 65)
    print(f"  NEPOOL TDA -- Step 04 Ward Baseline (exp {EXPERIMENT})")
    print("=" * 65, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    E, timestamps = load_features()
    labels_raw, Z, info = run_ward(E, n_clusters_forced=_args.n_clusters)

    labels, merge_info = tukey_merge(labels_raw, timestamps.values)
    info.update(merge_info)

    make_plots(E, labels, timestamps, Z)

    # Save
    df = pd.DataFrame({"datetime": timestamps.values, "cluster": labels})
    df.to_parquet(OUT_DIR / "labels.parquet", index=False)
    with open(OUT_DIR / "ward_report.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    print(f"\n  Done in {time.time()-t0:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
