"""
step04_tomato.py
================
NEPOOL TDA Pipeline -- Step 04: ToMATo Clustering + Tukey HSD Merge

Input  : results/exp_{X}/step03/pca_embeddings.parquet
Output : results/exp_{X}/step04/labels.parquet, tomato_report.json
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
from gudhi.clustering.tomato import Tomato

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

_parser = argparse.ArgumentParser()
_parser.add_argument("--exp", default="D", choices=["A", "B", "C", "D", "E", "FE", "FT", "COMBO", "D2"])
_args = _parser.parse_args()
EXPERIMENT = _args.exp

RESULTS_DIR = Path("results")
PCA_PATH    = RESULTS_DIR / f"exp_{EXPERIMENT}" / "step03" / "pca_embeddings.parquet"
OUT_DIR     = RESULTS_DIR / f"exp_{EXPERIMENT}" / "step04"
SEED        = 42
DPI         = 150
CMAP        = plt.get_cmap("tab20")

KNN_GRID = [20, 40, 60, 80, 100, 150]


def load_data():
    df = pd.read_parquet(PCA_PATH)
    ts_col = "datetime" if "datetime" in df.columns else "date"
    timestamps = pd.to_datetime(df[ts_col])
    E = df.drop(columns=[ts_col]).values.astype(np.float32)
    print(f"  Loaded: {E.shape}", flush=True)
    return E, timestamps


def run_tomato(E):
    n = E.shape[0]
    print(f"\n  ToMATo grid search (logDTM, {E.shape[1]}D, n={n}) ...", flush=True)

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

        results.append({"k": k, "k_auto": k_auto, "gap_size": gap_size,
                         "gap_norm": gap_norm, "tomato": t})
        print(f"    k={k:>3d}  K={k_auto:>3d}  gap_norm={gap_norm:.4f}", flush=True)

    best = max(results, key=lambda r: r["gap_norm"])
    print(f"  -> Best: k={best['k']}, K={best['k_auto']}", flush=True)

    tomato = best["tomato"]
    tomato.n_clusters_ = best["k_auto"]
    labels = np.asarray(tomato.labels_)

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    R{u}: {c} ({c/n*100:.1f}%)", flush=True)

    info = {
        "experiment": EXPERIMENT,
        "method": "grid_search",
        "density_type": "logDTM",
        "knn_k": best["k"],
        "n_clusters_raw": int(best["k_auto"]),
        "gap_norm": round(best["gap_norm"], 4),
        "input_shape": [int(n), int(E.shape[1])],
        "cluster_sizes_raw": {int(u): int(c) for u, c in zip(unique, counts)},
    }
    return labels, info


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
            print(f"    Iter {iteration}: merge {b}->{a} (p={best_p:.4f}) K={k_now}", flush=True)

    unique_final = np.unique(labels_v)
    remap = {old: new for new, old in enumerate(unique_final)}
    labels_out = np.full(len(labels), -1, dtype=int)
    labels_out[valid] = np.array([remap[l] for l in labels_v])

    k_after = len(unique_final)
    print(f"    K: {k_before} -> {k_after}", flush=True)

    unique_m, counts_m = np.unique(labels_out[labels_out >= 0], return_counts=True)
    for u, c in zip(unique_m, counts_m):
        print(f"      R{u}: {c} ({c/valid.sum()*100:.1f}%)", flush=True)

    return labels_out, {"k_before_merge": k_before, "k_after_merge": k_after,
                         "cluster_sizes_merged": {int(u): int(c) for u, c in zip(unique_m, counts_m)}}


def make_plots(E, labels, timestamps):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    valid = labels >= 0
    K = len(np.unique(labels[valid]))
    colors = [CMAP(i % 20) for i in range(K)]

    if E.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        for k in range(K):
            mask = labels == k
            ax.scatter(E[mask, 0], E[mask, 1], s=3, alpha=0.4, color=colors[k], label=f"R{k}")
        ax.legend(markerscale=5, fontsize=7, ncol=2)
        ax.set_title(f"Regimes (K={K})")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "01_cluster_scatter.png", dpi=DPI)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 3))
    dt = timestamps.values
    for k in range(K):
        mask = labels == k
        ax.scatter(dt[mask], np.full(mask.sum(), k), s=2, color=colors[k], alpha=0.6)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"R{k}" for k in range(K)])
    ax.set_title("Timeline"); fig.tight_layout()
    fig.savefig(OUT_DIR / "02_timeline.png", dpi=DPI)
    plt.close(fig)


def main():
    t0 = time.time()
    print("=" * 65)
    print(f"  NEPOOL TDA -- Step 04: ToMATo + Tukey Merge  (exp {EXPERIMENT})")
    print("=" * 65, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    E, timestamps = load_data()
    labels_raw, info = run_tomato(E)
    labels, merge_info = tukey_merge(labels_raw, timestamps.values)
    info.update(merge_info)

    make_plots(E, labels, timestamps)

    pd.DataFrame({"datetime": timestamps.values, "cluster": labels}).to_parquet(
        OUT_DIR / "labels.parquet", index=False)
    with open(OUT_DIR / "tomato_report.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    print(f"\n  Done in {time.time()-t0:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
