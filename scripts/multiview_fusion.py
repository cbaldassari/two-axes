"""
multiview_fusion.py
===================
Multi-view integration of FE and MOMENT representations.

Two approaches:
  D: Late Fusion   — concat diffusion coordinates (FE 11D + MOMENT 2D = 13D)
  E: Kernel Fusion  — average Gaussian kernels, then DiffMaps on fused kernel

Both followed by ToMATo + Tukey HSD merge.
"""

import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from gudhi.clustering.tomato import Tomato

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS = Path("results")
SEED = 42


def run_kernel_fusion_diffmaps(X1, X2, n_comp):
    X1 = StandardScaler().fit_transform(X1).astype(np.float64)
    X2 = StandardScaler().fit_transform(X2).astype(np.float64)
    n = X1.shape[0]
    k_nn = max(15, int(np.sqrt(n)))

    def make_kernel(X):
        nn = NearestNeighbors(n_neighbors=k_nn, metric="euclidean", n_jobs=-1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        sigma = float(np.median(distances[:, -1]))
        epsilon = sigma ** 2
        rows = np.repeat(np.arange(n), k_nn)
        cols = indices.ravel()
        vals = np.exp(-(distances ** 2).ravel() / epsilon)
        K = csr_matrix((vals, (rows, cols)), shape=(n, n))
        return (K + K.T) / 2.0

    K1 = make_kernel(X1)
    K2 = make_kernel(X2)
    K_fused = (K1 + K2) / 2.0

    d = np.array(K_fused.sum(axis=1)).flatten()
    D_inv = diags(np.power(d, -1.0))
    K_a = D_inv @ K_fused @ D_inv
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


def run_tomato_tukey(E, lmp):
    KNN_GRID = [20, 40, 60, 80, 100, 150]
    best = None
    for k in KNN_GRID:
        t = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1, k=k)
        t.fit(E)
        diagram = np.array(t.diagram_) if len(t.diagram_) > 0 else np.empty((0, 2))
        if diagram.shape[0] > 0:
            pers = np.sort(diagram[:, 1] - diagram[:, 0])[::-1]
            if len(pers) >= 2:
                gaps = pers[:-1] - pers[1:]
                gap_idx = int(np.argmax(gaps))
                k_auto = gap_idx + 2
                gap_size = float(gaps[gap_idx])
            else:
                k_auto = max(1, len(pers) + 1)
                gap_size = 0.0
        else:
            k_auto = 2
            gap_size = 0.0
        med = float(np.median(np.abs(pers))) if len(pers) > 0 else 1.0
        gap_norm = gap_size / max(med, 1e-12)
        if best is None or gap_norm > best[1]:
            best = (t, gap_norm, k_auto, k)

    tomato, _, k_auto, best_k = best
    tomato.n_clusters_ = k_auto
    labels = np.asarray(tomato.labels_)
    print(f"    ToMATo: k={best_k}, K_raw={k_auto}")

    # Tukey merge
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    valid = ~np.isnan(lmp)
    labels_v = labels[valid].copy()
    lmp_v = lmp[valid]
    k_before = len(np.unique(labels_v))

    merged = True
    while merged:
        unique_labels = np.unique(labels_v)
        if len(unique_labels) <= 2:
            break
        result = pairwise_tukeyhsd(lmp_v, labels_v, alpha=0.05)
        pairs = list(combinations(sorted(unique_labels), 2))
        best_p = -1.0
        best_pair = None
        for i in range(min(len(result.reject), len(pairs))):
            if not result.reject[i] and result.pvalues[i] > best_p:
                best_p = result.pvalues[i]
                best_pair = pairs[i]
        if best_pair is None:
            merged = False
        else:
            a, b = best_pair
            labels_v[labels_v == b] = a

    unique_final = np.unique(labels_v)
    remap = {old: new for new, old in enumerate(unique_final)}
    labels_out = np.full(len(labels), -1, dtype=int)
    labels_out[valid] = np.array([remap[l] for l in labels_v])
    k_after = len(unique_final)

    # Metrics
    valid_mask = labels_out >= 0
    K = len(np.unique(labels_out[valid_mask]))
    lmp_valid = lmp[valid_mask]
    lab_valid = labels_out[valid_mask]
    grand_mean = lmp_valid.mean()
    ss_total = ((lmp_valid - grand_mean) ** 2).sum()
    ss_between = sum(
        (lab_valid == u).sum() * (lmp_valid[lab_valid == u].mean() - grand_mean) ** 2
        for u in np.unique(lab_valid)
    )
    eta2 = ss_between / max(ss_total, 1e-12)
    sil = float(silhouette_score(E[valid_mask], labels_out[valid_mask])) if K >= 2 else -1
    transitions = int(np.sum(labels_out[1:] != labels_out[:-1]))
    trans_rate = transitions / (len(labels_out) - 1)
    sojourns = []
    current_len = 1
    for i in range(1, len(labels_out)):
        if labels_out[i] == labels_out[i - 1]:
            current_len += 1
        else:
            sojourns.append(current_len)
            current_len = 1
    sojourns.append(current_len)
    sojourn_h = np.mean(sojourns) * 6

    # Tukey check
    tukey_result = pairwise_tukeyhsd(lmp_valid, lab_valid, alpha=0.05)
    n_sig = int(np.sum(tukey_result.reject))
    n_tot = len(tukey_result.reject)

    return K, eta2, sil, trans_rate, sojourn_h, k_before, k_after, n_sig, n_tot


def main():
    print("=" * 65)
    print("  Multi-View Fusion: FE + MOMENT")
    print("=" * 65)

    # Load data
    dc_fe = pd.read_parquet(RESULTS / "exp_FE/step03/pca_embeddings.parquet")
    dc_mom = pd.read_parquet(RESULTS / "exp_C/step03/pca_embeddings.parquet")
    ts = pd.to_datetime(dc_fe["datetime"])

    E_fe = dc_fe.drop(columns=["datetime"]).values.astype(np.float32)
    E_mom = dc_mom.drop(columns=["datetime"]).values.astype(np.float32)

    raw_fe = pd.read_parquet(RESULTS / "exp_FE/embeddings.parquet")
    raw_mom = pd.read_parquet(RESULTS / "exp_C/embeddings.parquet")
    ts_col_fe = [c for c in raw_fe.columns if "time" in c.lower() or "date" in c.lower()][0]
    ts_col_mom = [c for c in raw_mom.columns if "time" in c.lower() or "date" in c.lower()][0]
    R_fe = raw_fe.drop(columns=[ts_col_fe]).values.astype(np.float32)
    R_mom_raw = raw_mom.drop(columns=[ts_col_mom]).values.astype(np.float32)

    # PCA-MP on MOMENT: 1024 -> ~53D (avoid 1024D kernel which is too slow)
    from utils.pca_dim_selector import marchenko_pastur_n_components
    from sklearn.decomposition import PCA
    X_m = StandardScaler().fit_transform(R_mom_raw)
    nm, pm = X_m.shape
    pca_full = PCA(random_state=SEED).fit(X_m)
    n_comp_mp, _ = marchenko_pastur_n_components(pca_full.explained_variance_, nm, pm)
    R_mom = PCA(n_components=n_comp_mp, random_state=SEED).fit_transform(X_m).astype(np.float32)
    print(f"  MOMENT PCA-MP: {pm}D -> {n_comp_mp}D (for kernel fusion)")

    pre = pd.read_parquet(RESULTS / "preprocessed.parquet")
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    lmp = pre.set_index("datetime")["lmp"].reindex(ts).values

    n = len(E_fe)
    print(f"  n={n}")
    print(f"  FE DiffCoords: {E_fe.shape[1]}D, MOMENT DiffCoords: {E_mom.shape[1]}D")
    print(f"  FE raw: {R_fe.shape[1]}D, MOMENT PCA: {R_mom.shape[1]}D")

    # ── D: Late Fusion ──
    print()
    print("-" * 65)
    print("  D: LATE FUSION (FE 11D + MOMENT 2D = 13D diffusion coords)")
    print("-" * 65)
    t0 = time.time()
    E_late = np.hstack([
        StandardScaler().fit_transform(E_fe),
        StandardScaler().fit_transform(E_mom),
    ])
    print(f"  Combined shape: {E_late.shape}")
    K, eta2, sil, tr, soj, k_pre, k_post, tsig, ttot = run_tomato_tukey(E_late, lmp)
    print(f"  K: {k_pre} -> {k_post}")
    print(f"  eta2={eta2:.4f}  sil={sil:.4f}  trans={tr:.4f}  sojourn={soj:.0f}h  Tukey={tsig}/{ttot}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── E: Kernel Fusion ──
    print()
    print("-" * 65)
    print("  E: KERNEL FUSION (average kernels -> DiffMaps)")
    print("-" * 65)

    # Sweep n_comp
    print("  Sweeping n_comp...")
    best_nc = 2
    best_sil_sweep = -1
    for nc in range(2, 16):
        t0 = time.time()
        coords = run_kernel_fusion_diffmaps(R_fe, R_mom, nc)
        t_tomato = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1,
                          k=int(np.sqrt(n)))
        t_tomato.fit(coords)
        diag = np.array(t_tomato.diagram_) if len(t_tomato.diagram_) > 0 else np.empty((0, 2))
        if diag.shape[0] > 0:
            pers = np.sort(diag[:, 1] - diag[:, 0])[::-1]
            if len(pers) >= 2:
                gaps = pers[:-1] - pers[1:]
                k_auto = int(np.argmax(gaps)) + 2
            else:
                k_auto = max(1, len(pers) + 1)
        else:
            k_auto = 2
        k_auto = max(k_auto, 2)
        t_tomato.n_clusters_ = k_auto
        lab = np.asarray(t_tomato.labels_)
        if len(np.unique(lab)) >= 3:
            s = float(silhouette_score(coords, lab))
        else:
            s = -1
        elapsed = time.time() - t0
        print(f"    n_comp={nc:>2d}  K={k_auto:>3d}  sil={s:>+.4f}  ({elapsed:.1f}s)")
        if s > best_sil_sweep and len(np.unique(lab)) >= 3:
            best_sil_sweep = s
            best_nc = nc

    print(f"  -> Best n_comp={best_nc} (sil={best_sil_sweep:.4f})")

    t0 = time.time()
    E_kernel = run_kernel_fusion_diffmaps(R_fe, R_mom, best_nc)
    K, eta2, sil, tr, soj, k_pre, k_post, tsig, ttot = run_tomato_tukey(E_kernel, lmp)
    print(f"  K: {k_pre} -> {k_post}")
    print(f"  eta2={eta2:.4f}  sil={sil:.4f}  trans={tr:.4f}  sojourn={soj:.0f}h  Tukey={tsig}/{ttot}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Comparison ──
    print()
    print("=" * 65)
    print("  COMPARISON TABLE")
    print("=" * 65)
    header = f"{'Config':<30s} {'K':>3s} {'eta2':>7s} {'Sil':>7s} {'Trans%':>7s} {'Sojourn':>8s} {'Tukey':>8s}"
    print(header)
    print("-" * 75)

    for name, exp in [("B:FE+ToMATo", "exp_FE"), ("A:MOMENT+ToMATo", "exp_C"),
                       ("C:COMBO+ToMATo (naive)", "exp_COMBO")]:
        qpath = RESULTS / exp / "step05/quality_report.json"
        if qpath.exists():
            with open(qpath) as f:
                q = json.load(f)
            eco = q["economic"]
            geo = q["geometric"]
            Kq = len(eco["lmp_per_regime"])
            print(f"{name:<30s} {Kq:>3d} {eco['eta_squared']:>7.4f} {geo['silhouette_avg']:>7.4f} "
                  f"{eco['transition_rate']:>7.4f} {eco['sojourn_mean_hours']:>7.0f}h "
                  f"{eco['pairwise_significant']}/{eco['pairwise_total']}")

    # Re-run D and E for clean numbers
    E_late2 = np.hstack([StandardScaler().fit_transform(E_fe), StandardScaler().fit_transform(E_mom)])
    K_d, eta2_d, sil_d, tr_d, soj_d, _, _, tsig_d, ttot_d = run_tomato_tukey(E_late2, lmp)
    print(f"{'D:Late Fusion (DC concat)':<30s} {K_d:>3d} {eta2_d:>7.4f} {sil_d:>7.4f} "
          f"{tr_d:>7.4f} {soj_d:>7.0f}h {tsig_d}/{ttot_d}")

    E_kernel2 = run_kernel_fusion_diffmaps(R_fe, R_mom, best_nc)
    K_e, eta2_e, sil_e, tr_e, soj_e, _, _, tsig_e, ttot_e = run_tomato_tukey(E_kernel2, lmp)
    print(f"{'E:Kernel Fusion (avg kernel)':<30s} {K_e:>3d} {eta2_e:>7.4f} {sil_e:>7.4f} "
          f"{tr_e:>7.4f} {soj_e:>7.0f}h {tsig_e}/{ttot_e}")

    print("=" * 65)


if __name__ == "__main__":
    main()
