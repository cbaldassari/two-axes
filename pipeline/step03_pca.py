"""
step03_pca.py
=============
NEPOOL TDA Pipeline -- Step 03: Diffusion Maps

Diffusion Maps (Coifman & Lafon 2006) on the embedding space.
Preserves manifold geometry for ToMATo density-based clustering.

Component selection: automatic via clustering stability criterion.
  Sweep n_comp = [3, 5, 7, 10, 13, 16, 20], for each run ToMATo (logDTM,
  k=sqrt(n), max_gap) and evaluate silhouette. Pick the n_comp that
  maximizes silhouette — the one where clusters are most separable.

Input  : results/exp_{X}/embeddings.parquet
Output : results/exp_{X}/step03/
           pca_embeddings.parquet   (diffusion coordinates)
           pca_report.json

Uso:
  python pipeline/step03_pca.py --exp D
  python pipeline/step03_pca.py --exp D --n-comp 10   # force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C
from utils.pca_dim_selector import marchenko_pastur_n_components

_parser = argparse.ArgumentParser(description="Step 03 -- Diffusion Maps")
_parser.add_argument("--exp", default="D", choices=["A", "B", "C", "D", "E", "FE", "FT", "COMBO", "D2"])
_parser.add_argument("--n-comp", type=int, default=None,
                     help="Force n_components (default: auto via stability)")
_args = _parser.parse_args()
EXPERIMENT  = _args.exp
N_COMP_FORCED = _args.n_comp

RESULTS_DIR     = Path(C.RESULTS_DIR)
EMBEDDINGS_PATH = RESULTS_DIR / f"exp_{EXPERIMENT}" / "embeddings.parquet"
OUT_DIR         = RESULTS_DIR / f"exp_{EXPERIMENT}" / "step03"
SEED            = C.RANDOM_STATE
DPI             = 150


# =====================================================================
#  Load
# =====================================================================

def load_embeddings():
    df = pd.read_parquet(EMBEDDINGS_PATH)
    ts_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
    timestamps = pd.to_datetime(df[ts_col])
    E = df.drop(columns=[ts_col]).values.astype(np.float32)
    return E, timestamps


# =====================================================================
#  Diffusion Maps core
# =====================================================================

def pca_denoise(E: np.ndarray) -> tuple[np.ndarray, dict]:
    """PCA denoising via Marchenko-Pastur before Diffusion Maps."""
    from sklearn.decomposition import PCA
    X = StandardScaler().fit_transform(E)
    n, p = X.shape

    pca_full = PCA(random_state=SEED).fit(X)
    n_comp_mp, lambda_max = marchenko_pastur_n_components(
        pca_full.explained_variance_, n, p
    )
    pca = PCA(n_components=n_comp_mp, random_state=SEED)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA denoise: {p}D -> {n_comp_mp}D ({var:.1%} var, MP lambda_max={lambda_max:.2f})",
          flush=True)
    info = {"pca_denoise_dim": n_comp_mp, "pca_denoise_variance": round(float(var), 4),
            "pca_denoise_lambda_max": round(float(lambda_max), 4)}
    return X_pca, info


def compute_diffusion_maps(E: np.ndarray, n_comp: int,
                           denoise: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute Diffusion Maps coordinates.
    If denoise=True, applies PCA Marchenko-Pastur first.
    Returns (coords, eigenvalues_all, pca_info).
    """
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh

    pca_info = {}
    if denoise and E.shape[1] > 100:
        E, pca_info = pca_denoise(E)

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
    np.random.seed(SEED)
    v0 = np.random.randn(n)
    eigenvalues, eigenvectors = eigsh(P, k=n_eig + 1, which="LM", v0=v0)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][1:]
    eigenvectors = eigenvectors[:, idx][:, 1:]

    coords = eigenvectors[:, :n_comp] * eigenvalues[:n_comp][np.newaxis, :]
    return coords.astype(np.float32), eigenvalues, pca_info


# =====================================================================
#  Automatic n_comp selection via clustering stability
# =====================================================================

def auto_select_n_comp(E: np.ndarray) -> tuple[int, dict]:
    """
    Sweep n_comp candidates. For each, compute diffusion coordinates,
    run ToMATo (logDTM, k=sqrt(n), max_gap), evaluate eta-squared on LMP.
    Pick the n_comp that maximizes eta² (economic separability of regimes).

    Returns (best_n_comp, sweep_results).
    """
    from gudhi.clustering.tomato import Tomato

    candidates = list(range(2, 21))
    n = E.shape[0]
    knn_tomato = max(10, int(np.sqrt(n)))

    print(f"\n  Auto n_comp selection (sweep {candidates}, criterion=eta^2) ...", flush=True)

    # Load LMP for eta² computation
    pre = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    lmp_lookup = pre.set_index("datetime")["lmp"]

    # Load timestamps for matching
    df_emb = pd.read_parquet(EMBEDDINGS_PATH)
    ts_col = [c for c in df_emb.columns if "time" in c.lower() or "date" in c.lower()][0]
    timestamps = pd.to_datetime(df_emb[ts_col])
    lmp = lmp_lookup.reindex(timestamps).values
    valid_lmp = ~np.isnan(lmp)

    # Compute diffusion maps once with max components
    max_comp = max(candidates)
    coords_full, eigenvalues, pca_info = compute_diffusion_maps(E, max_comp)

    sweep = []
    for nc in candidates:
        coords = coords_full[:, :nc]

        # ToMATo
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

        # eta² on LMP
        n_unique = len(np.unique(labels))
        if n_unique >= 2 and valid_lmp.sum() > 0:
            labels_v = labels[valid_lmp]
            lmp_v = lmp[valid_lmp]
            grand_mean = lmp_v.mean()
            ss_total = ((lmp_v - grand_mean) ** 2).sum()
            ss_between = 0.0
            for u in np.unique(labels_v):
                mask = labels_v == u
                ss_between += mask.sum() * (lmp_v[mask].mean() - grand_mean) ** 2
            eta2 = ss_between / max(ss_total, 1e-12)
        else:
            eta2 = 0.0

        # Silhouette (secondary metric)
        if n_unique >= 2:
            sil = float(silhouette_score(coords, labels, random_state=SEED))
        else:
            sil = -1.0

        sweep.append({
            "n_comp": nc,
            "k_auto": k_auto,
            "eta2": round(eta2, 4),
            "silhouette": round(sil, 4),
            "n_unique": n_unique,
        })
        print(f"    n_comp={nc:>2d}  K={k_auto:>3d}  eta2={eta2:.4f}  sil={sil:>+.4f}", flush=True)

    # Pick best: highest silhouette with K >= 3 (geometric criterion, no LMP bias)
    valid_sweeps = [s for s in sweep if s["n_unique"] >= 3 and s["silhouette"] > 0]
    if valid_sweeps:
        best = max(valid_sweeps, key=lambda s: s["silhouette"])
    else:
        best = max(sweep, key=lambda s: s["silhouette"])

    best_nc = best["n_comp"]
    print(f"\n  -> Best: n_comp={best_nc} (sil={best['silhouette']:.4f}, K={best['k_auto']})",
          flush=True)

    return best_nc, {"sweep": sweep, "eigenvalues_top30": eigenvalues[:30].tolist(),
                     **pca_info}


# =====================================================================
#  Plots
# =====================================================================

def make_plots(coords, timestamps, eigenvalues, n_comp, sweep):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 01: Eigenvalues + sweep results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_show = min(30, len(eigenvalues))
    axes[0].bar(range(1, n_show + 1), eigenvalues[:n_show], color="#4a90d9", alpha=0.8)
    axes[0].axvline(n_comp + 0.5, color="green", ls=":", lw=1.5, label=f"n_comp={n_comp}")
    axes[0].set_xlabel("Component"); axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Diffusion eigenvalues"); axes[0].legend()

    ncs = [s["n_comp"] for s in sweep]
    eta2s = [s.get("eta2", s.get("silhouette", 0)) for s in sweep]
    axes[1].plot(ncs, eta2s, "o-", color="#e07b39", lw=2, ms=8)
    axes[1].axvline(n_comp, color="green", ls=":", lw=1.5)
    axes[1].set_xlabel("n_components"); axes[1].set_ylabel("eta^2")
    axes[1].set_title("n_comp selection (max eta^2 on LMP)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_scree.png", dpi=DPI)
    plt.close(fig)

    # 02: 2D scatter
    if coords.shape[1] >= 2:
        dt = pd.to_datetime(timestamps)
        years = dt.dt.year
        fig, ax = plt.subplots(figsize=(10, 8))
        for y in sorted(years.unique()):
            mask = years == y
            ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.4, label=str(y))
        ax.set_xlabel("DC1"); ax.set_ylabel("DC2")
        ax.set_title(f"Diffusion Maps 2D (n_comp={n_comp})")
        ax.legend(markerscale=5)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "03_pca2d.png", dpi=DPI)
        plt.close(fig)

    print(f"  Plots saved -> {OUT_DIR}/", flush=True)


# =====================================================================
#  Main
# =====================================================================

def main():
    t0 = time.time()
    print("=" * 65)
    print(f"  NEPOOL TDA -- Step 03: Diffusion Maps")
    print(f"  Experiment : {EXPERIMENT}")
    print("=" * 65, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Loading embeddings from {EMBEDDINGS_PATH}...", flush=True)
    E, timestamps = load_embeddings()
    print(f"  shape: {E.shape}", flush=True)

    # Select n_comp
    if N_COMP_FORCED:
        n_comp = N_COMP_FORCED
        print(f"\n  n_comp forced: {n_comp}", flush=True)
        coords, eigenvalues, pca_info = compute_diffusion_maps(E, n_comp)
        sweep_info = {"sweep": [], "eigenvalues_top30": eigenvalues[:30].tolist(), **pca_info}
    else:
        n_comp, sweep_info = auto_select_n_comp(E)
        coords, eigenvalues, _ = compute_diffusion_maps(E, n_comp)

    print(f"\n  Diffusion Maps: {E.shape[1]}D -> {n_comp}D", flush=True)

    # Save
    cols = [f"dc_{i}" for i in range(n_comp)]
    df = pd.DataFrame(coords, columns=cols)
    df.insert(0, "datetime", timestamps.values)
    out_path = OUT_DIR / "pca_embeddings.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}", flush=True)

    report = {
        "experiment": EXPERIMENT,
        "method": "diffusion_maps",
        "n_components": n_comp,
        "auto_selection": "stability" if not N_COMP_FORCED else "forced",
        **sweep_info,
    }
    for key in ["eigenvalues_top30"]:
        if key in report:
            report[key] = [round(float(v), 6) for v in report[key]]
    if "sweep" in report:
        for s in report["sweep"]:
            s["silhouette"] = round(s["silhouette"], 4)

    with open(OUT_DIR / "pca_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Plots
    make_plots(coords, timestamps, eigenvalues,
               n_comp, sweep_info.get("sweep", []))

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
