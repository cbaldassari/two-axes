"""
ablation_features.py
====================
Feature ablation study for the FE representation.

Tests subsets of the 19 hand-crafted features through the full pipeline
(DiffMaps -> ToMATo -> Tukey HSD merge) and reports eta^2 for each.

Uses existing FE embeddings (results/exp_FE/embeddings.parquet) — no need
to re-run MOMENT or preprocessing.

Output: results/ablation_features.csv + console table
"""
from __future__ import annotations

import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from gudhi.clustering.tomato import Tomato
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
SEED = C.RANDOM_STATE
KNN_GRID = [20, 40, 60, 80, 100, 150]

# Feature groups (indices into the 19 features)
GROUPS = {
    "distributional": list(range(0, 11)),    # mean, std, skew, kurt, min, max, range, median, p5, p95, iqr
    "acf":            list(range(11, 15)),    # acf_1h, acf_6h, acf_24h, acf_168h
    "vol_24h":        [15],                   # vol_24h
    "lmp_raw":        list(range(16, 19)),    # lmp_mean, lmp_p95, lmp_std
}

# Ablation configurations
CONFIGS = [
    ("ALL_19",              list(range(19)),                      "All 19 features (baseline)"),
    ("distributional_11",   GROUPS["distributional"],             "Distributional only (11)"),
    ("acf_4",               GROUPS["acf"],                        "ACF only (4)"),
    ("vol_1",               GROUPS["vol_24h"],                    "vol_24h only (1)"),
    ("lmp_raw_3",           GROUPS["lmp_raw"],                    "LMP raw only (3)"),
    ("no_lmp_raw_16",       list(range(16)),                      "Without LMP raw (16)"),
    ("no_distributional_8", GROUPS["acf"] + GROUPS["vol_24h"] + GROUPS["lmp_raw"],
                                                                  "Without distributional (8)"),
    ("no_acf_15",           GROUPS["distributional"] + GROUPS["vol_24h"] + GROUPS["lmp_raw"],
                                                                  "Without ACF (15)"),
    ("distrib_lmp_14",      GROUPS["distributional"] + GROUPS["lmp_raw"],
                                                                  "Distributional + LMP raw (14)"),
    ("acf_vol_5",           GROUPS["acf"] + GROUPS["vol_24h"],    "ACF + vol_24h (5)"),
    ("acf_lmp_7",           GROUPS["acf"] + GROUPS["vol_24h"] + GROUPS["lmp_raw"],
                                                                  "ACF + vol + LMP raw (7)"),
]


def load_data():
    """Load FE features + LMP for eta^2."""
    fe_path = RESULTS_DIR / "exp_FE" / "embeddings.parquet"
    df = pd.read_parquet(fe_path)
    timestamps = pd.to_datetime(df["datetime"])
    feats = df.drop(columns=["datetime"]).values.astype(np.float64)

    pre = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    lmp_lookup = pre.set_index("datetime")["lmp"]
    lmp = lmp_lookup.reindex(timestamps).values

    return feats, timestamps, lmp


def diffusion_maps(X_raw, n_comp):
    """Compute Diffusion Maps coordinates."""
    X = StandardScaler().fit_transform(X_raw).astype(np.float64)
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
    np.random.seed(SEED)
    v0 = np.random.randn(n)
    eigenvalues, eigenvectors = eigsh(P, k=n_eig + 1, which="LM", v0=v0)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][1:]
    eigenvectors = eigenvectors[:, idx][:, 1:]

    coords = eigenvectors[:, :n_comp] * eigenvalues[:n_comp][np.newaxis, :]
    return coords.astype(np.float32)


def auto_n_comp(X_raw, lmp):
    """Sweep n_comp [2,4,6,8,11], pick best silhouette with K>=3. Fast version."""
    best_sil = -2.0
    best_nc = 2
    candidates = [2, 4, 6, 8, 11]

    # Compute DiffMaps with max components
    max_nc = max(candidates)
    X = StandardScaler().fit_transform(X_raw).astype(np.float64)
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

    n_eig = min(max_nc + 5, n - 2)
    np.random.seed(SEED)
    v0 = np.random.randn(n)
    eigenvalues, eigenvectors = eigsh(P, k=n_eig + 1, which="LM", v0=v0)

    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort][1:]
    eigenvectors = eigenvectors[:, idx_sort][:, 1:]

    knn_tomato = max(10, int(np.sqrt(n)))

    for nc in candidates:
        if nc > eigenvectors.shape[1]:
            continue
        coords = (eigenvectors[:, :nc] * eigenvalues[:nc][np.newaxis, :]).astype(np.float32)

        tomato = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1, k=knn_tomato)
        tomato.fit(coords)

        diagram = np.array(tomato.diagram_) if len(tomato.diagram_) > 0 else np.empty((0, 2))
        if diagram.shape[0] > 0:
            pers = np.sort(diagram[:, 1] - diagram[:, 0])[::-1]
        else:
            pers = np.array([])

        if len(pers) >= 2:
            gaps = pers[:-1] - pers[1:]
            gap_idx = int(np.argmax(gaps))
            k_auto = gap_idx + 2
        else:
            k_auto = max(1, len(pers) + 1)

        k_auto = max(k_auto, 2)
        tomato.n_clusters_ = k_auto
        labels = np.asarray(tomato.labels_)

        n_unique = len(np.unique(labels))
        if n_unique >= 3:
            sil = float(silhouette_score(coords, labels, random_state=SEED))
            if sil > best_sil:
                best_sil = sil
                best_nc = nc

    return best_nc


def run_tomato(coords):
    """Run ToMATo with grid search on k."""
    n = coords.shape[0]
    results = []
    for k in KNN_GRID:
        t = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1, k=k)
        t.fit(coords)
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
    return np.asarray(tomato.labels_), best["k_auto"]


def tukey_merge(labels, lmp, alpha=0.05):
    """Iterative Tukey HSD merge."""
    valid = ~np.isnan(lmp)
    labels_v = labels[valid].copy()
    lmp_v = lmp[valid]

    k_before = len(np.unique(labels_v))

    merged = True
    while merged:
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

    # Remap to 0..K-1
    unique_final = np.unique(labels_v)
    remap = {old: new for new, old in enumerate(unique_final)}
    labels_out = np.full(len(labels), -1, dtype=int)
    labels_out[valid] = np.array([remap[l] for l in labels_v])

    return labels_out, k_before, len(unique_final)


def compute_eta2(labels, lmp):
    """Compute eta-squared (ANOVA) on LMP."""
    valid = (labels >= 0) & (~np.isnan(lmp))
    if valid.sum() == 0:
        return 0.0
    labels_v = labels[valid]
    lmp_v = lmp[valid]
    grand_mean = lmp_v.mean()
    ss_total = ((lmp_v - grand_mean) ** 2).sum()
    ss_between = 0.0
    for u in np.unique(labels_v):
        mask = labels_v == u
        ss_between += mask.sum() * (lmp_v[mask].mean() - grand_mean) ** 2
    return ss_between / max(ss_total, 1e-12)


def compute_transition_rate(labels):
    """Fraction of consecutive windows that change regime."""
    valid = labels >= 0
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return 0.0
    consecutive = np.diff(idx) == 1
    changes = labels[idx[1:]] != labels[idx[:-1]]
    transitions = (consecutive & changes).sum()
    total = consecutive.sum()
    return transitions / max(total, 1) if total > 0 else 0.0


def compute_sojourn(labels):
    """Mean sojourn time in hours (stride=6h)."""
    valid = labels >= 0
    lv = labels[valid]
    if len(lv) == 0:
        return 0.0
    runs = []
    current = lv[0]
    length = 1
    for i in range(1, len(lv)):
        if lv[i] == current:
            length += 1
        else:
            runs.append(length)
            current = lv[i]
            length = 1
    runs.append(length)
    return np.mean(runs) * 6  # convert to hours


def run_ablation_config(name, feat_indices, description, feats_all, timestamps, lmp):
    """Run full pipeline for one feature subset."""
    t0 = time.time()
    X = feats_all[:, feat_indices]
    n_feat = len(feat_indices)

    # Special case: 1 feature -> skip DiffMaps, cluster directly
    if n_feat == 1:
        coords = StandardScaler().fit_transform(X).astype(np.float32)
        # Can't do DiffMaps on 1D, use raw
        n_comp = 1
    else:
        # Auto select n_comp
        n_comp = auto_n_comp(X, lmp)
        coords = diffusion_maps(X, n_comp)

    # ToMATo
    labels_raw, k_raw = run_tomato(coords)

    # Tukey merge
    labels, k_before, k_after = tukey_merge(labels_raw, lmp)

    # Metrics
    eta2 = compute_eta2(labels, lmp)
    trans = compute_transition_rate(labels)
    sojourn = compute_sojourn(labels)

    elapsed = time.time() - t0

    result = {
        "config": name,
        "description": description,
        "n_features": n_feat,
        "n_comp_diffmaps": n_comp,
        "K_raw": k_raw,
        "K_merged": k_after,
        "eta2": round(eta2, 4),
        "transition_rate": round(trans, 4),
        "sojourn_h": round(sojourn, 1),
        "time_s": round(elapsed, 1),
    }
    return result


def main():
    print("\n" + "=" * 75)
    print("  FEATURE ABLATION STUDY")
    print("  Pipeline: StandardScaler -> DiffMaps(auto) -> ToMATo(grid) -> Tukey")
    print("=" * 75)

    feats_all, timestamps, lmp = load_data()
    feat_names = C.FE_FEATURES
    print(f"  Loaded {feats_all.shape[0]} windows x {feats_all.shape[1]} features")
    print(f"  Features: {feat_names}\n")

    results = []
    for i, (name, indices, desc) in enumerate(CONFIGS):
        sel_names = [feat_names[j] for j in indices]
        print(f"  [{i+1}/{len(CONFIGS)}] {name} ({len(indices)}D): {sel_names[:5]}{'...' if len(sel_names)>5 else ''}")
        r = run_ablation_config(name, indices, desc, feats_all, timestamps, lmp)
        results.append(r)
        print(f"         K={r['K_merged']}  eta2={r['eta2']:.4f}  sojourn={r['sojourn_h']:.0f}h  trans={r['transition_rate']:.1%}  ({r['time_s']:.0f}s)\n")

    # Summary table
    df = pd.DataFrame(results)
    print("\n" + "=" * 75)
    print("  ABLATION RESULTS")
    print("=" * 75)
    print(df[["config", "n_features", "K_merged", "eta2", "transition_rate", "sojourn_h"]].to_string(index=False))

    out_path = RESULTS_DIR / "ablation_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
