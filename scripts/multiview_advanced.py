"""
multiview_advanced.py
=====================
Advanced multi-view fusion of FE and MOMENT representations.

Three approaches:
  F: Weighted Late Fusion — grid search alpha on DC concat
  G: CCA Fusion — Canonical Correlation Analysis
  H: Procrustes Alignment — align MOMENT manifold to FE manifold

All followed by ToMATo + Tukey HSD merge.
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
from sklearn.cross_decomposition import CCA
from sklearn.metrics import silhouette_score
from scipy.spatial import procrustes
from gudhi.clustering.tomato import Tomato
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS = Path("results")
SEED = 42


def tomato_tukey(E, lmp):
    best = None
    for k in [20, 40, 60, 80, 100, 150]:
        t = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1, k=k)
        t.fit(E)
        diag = np.array(t.diagram_) if len(t.diagram_) > 0 else np.empty((0, 2))
        if diag.shape[0] > 0:
            pers = np.sort(diag[:, 1] - diag[:, 0])[::-1]
            if len(pers) >= 2:
                gaps = pers[:-1] - pers[1:]
                k_auto = int(np.argmax(gaps)) + 2
                gap_size = float(gaps[int(np.argmax(gaps))])
            else:
                k_auto = max(1, len(pers) + 1)
                gap_size = 0
        else:
            k_auto = 2
            gap_size = 0
        med = float(np.median(np.abs(pers))) if len(pers) > 0 else 1
        gn = gap_size / max(med, 1e-12)
        if best is None or gn > best[1]:
            best = (t, gn, k_auto, k)

    tomato, _, k_auto, best_k = best
    tomato.n_clusters_ = k_auto
    labels = np.asarray(tomato.labels_)

    valid = ~np.isnan(lmp)
    lv = labels[valid].copy()
    pv = lmp[valid]

    merged = True
    while merged:
        ul = np.unique(lv)
        if len(ul) <= 2:
            break
        res = pairwise_tukeyhsd(pv, lv, alpha=0.05)
        pairs = list(combinations(sorted(ul), 2))
        bp = -1
        bpair = None
        for i in range(min(len(res.reject), len(pairs))):
            if not res.reject[i] and res.pvalues[i] > bp:
                bp = res.pvalues[i]
                bpair = pairs[i]
        if bpair is None:
            merged = False
        else:
            a, b = bpair
            lv[lv == b] = a

    uf = np.unique(lv)
    remap = {o: n for n, o in enumerate(uf)}
    lo = np.full(len(labels), -1, dtype=int)
    lo[valid] = np.array([remap[l] for l in lv])

    vm = lo >= 0
    K = len(np.unique(lo[vm]))
    pv2 = lmp[vm]
    lv2 = lo[vm]
    gm = pv2.mean()
    sst = ((pv2 - gm) ** 2).sum()
    ssb = sum((lv2 == u).sum() * (pv2[lv2 == u].mean() - gm) ** 2
              for u in np.unique(lv2))
    eta2 = ssb / max(sst, 1e-12)
    sil = float(silhouette_score(E[vm], lo[vm])) if K >= 2 else -1
    tr = int(np.sum(lo[1:] != lo[:-1])) / (len(lo) - 1)
    sojs = []
    cl = 1
    for i in range(1, len(lo)):
        if lo[i] == lo[i - 1]:
            cl += 1
        else:
            sojs.append(cl)
            cl = 1
    sojs.append(cl)
    soj_h = np.mean(sojs) * 6
    tk = pairwise_tukeyhsd(pv2, lv2, alpha=0.05)
    return K, eta2, sil, tr, soj_h, k_auto, int(np.sum(tk.reject)), len(tk.reject)


def main():
    t_start = time.time()
    print("=" * 65)
    print("  Advanced Multi-View Fusion: FE + MOMENT")
    print("=" * 65)

    # Load diffusion coordinates
    dc_fe = pd.read_parquet(RESULTS / "exp_FE/step03/pca_embeddings.parquet")
    dc_mom = pd.read_parquet(RESULTS / "exp_C/step03/pca_embeddings.parquet")
    ts = pd.to_datetime(dc_fe["datetime"])

    E_fe = dc_fe.drop(columns=["datetime"]).values.astype(np.float64)  # 11D
    E_mom = dc_mom.drop(columns=["datetime"]).values.astype(np.float64)  # 2D

    pre = pd.read_parquet(RESULTS / "preprocessed.parquet")
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    lmp = pre.set_index("datetime")["lmp"].reindex(ts).values

    n = len(E_fe)
    d_fe = E_fe.shape[1]
    d_mom = E_mom.shape[1]
    print(f"  n={n}, FE DC: {d_fe}D, MOMENT DC: {d_mom}D")

    results = []

    # ═══════════════════════════════════════════════════════════
    # F: Weighted Late Fusion — grid search alpha
    # ═══════════════════════════════════════════════════════════
    print()
    print("-" * 65)
    print("  F: WEIGHTED LATE FUSION (alpha * FE + (1-alpha) * MOMENT)")
    print("-" * 65)

    fe_std = StandardScaler().fit_transform(E_fe)
    mom_std = StandardScaler().fit_transform(E_mom)

    best_alpha = None
    best_eta2_f = -1
    alpha_results = []

    for alpha_10 in range(0, 11):  # alpha from 0.0 to 1.0 step 0.1
        alpha = alpha_10 / 10.0
        E_w = np.hstack([alpha * fe_std, (1 - alpha) * mom_std])
        # Quick ToMATo without Tukey for speed
        t = Tomato(density_type="logDTM", graph_type="knn", n_jobs=-1,
                   k=int(np.sqrt(n)))
        t.fit(E_w.astype(np.float32))
        diag = np.array(t.diagram_) if len(t.diagram_) > 0 else np.empty((0, 2))
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
        t.n_clusters_ = k_auto
        labels = np.asarray(t.labels_)

        valid = ~np.isnan(lmp)
        lv = lmp[valid]
        lab_v = labels[valid]
        gm = lv.mean()
        sst = ((lv - gm) ** 2).sum()
        ssb = sum((lab_v == u).sum() * (lv[lab_v == u].mean() - gm) ** 2
                  for u in np.unique(lab_v))
        eta2 = ssb / max(sst, 1e-12)

        alpha_results.append((alpha, k_auto, eta2))
        print(f"    alpha={alpha:.1f}  K={k_auto:>3d}  eta2={eta2:.4f}")

        if eta2 > best_eta2_f:
            best_eta2_f = eta2
            best_alpha = alpha

    print(f"  -> Best alpha={best_alpha:.1f} (eta2={best_eta2_f:.4f})")

    # Full run with best alpha
    t0 = time.time()
    E_best = np.hstack([best_alpha * fe_std,
                         (1 - best_alpha) * mom_std]).astype(np.float32)
    K, eta2, sil, tr, soj, kraw, tsig, ttot = tomato_tukey(E_best, lmp)
    print(f"  Full run: K_raw={kraw} -> K={K}  eta2={eta2:.4f}  sil={sil:.4f}  "
          f"trans={tr:.4f}  sojourn={soj:.0f}h  Tukey={tsig}/{ttot}  "
          f"({time.time() - t0:.1f}s)")
    results.append(("F:Weighted (a={:.1f})".format(best_alpha),
                     K, eta2, sil, tr, soj, tsig, ttot))

    # ═══════════════════════════════════════════════════════════
    # G: CCA Fusion
    # ═══════════════════════════════════════════════════════════
    print()
    print("-" * 65)
    print("  G: CCA FUSION (Canonical Correlation Analysis)")
    print("-" * 65)

    t0 = time.time()
    n_cca = min(d_fe, d_mom)  # min(11, 2) = 2
    print(f"  CCA components: {n_cca} (min of {d_fe}, {d_mom})")

    cca = CCA(n_components=n_cca)
    X_fe_cca, X_mom_cca = cca.fit_transform(fe_std, mom_std)

    # Concatenate CCA projections from both views
    E_cca = np.hstack([X_fe_cca, X_mom_cca]).astype(np.float32)  # 2+2=4D
    print(f"  CCA output: FE->{X_fe_cca.shape[1]}D + MOM->{X_mom_cca.shape[1]}D "
          f"= {E_cca.shape[1]}D")

    K, eta2, sil, tr, soj, kraw, tsig, ttot = tomato_tukey(E_cca, lmp)
    print(f"  K_raw={kraw} -> K={K}  eta2={eta2:.4f}  sil={sil:.4f}  "
          f"trans={tr:.4f}  sojourn={soj:.0f}h  Tukey={tsig}/{ttot}  "
          f"({time.time() - t0:.1f}s)")
    results.append(("G:CCA Fusion", K, eta2, sil, tr, soj, tsig, ttot))

    # Also try CCA projections + remaining FE dimensions
    E_cca_plus = np.hstack([
        X_fe_cca,         # 2D: FE projected onto shared directions
        X_mom_cca,        # 2D: MOMENT projected onto shared directions
        fe_std[:, n_cca:] # 9D: remaining FE dimensions (orthogonal to CCA)
    ]).astype(np.float32)
    print(f"  CCA+residual: {E_cca_plus.shape[1]}D "
          f"(CCA 2+2 + FE residual {d_fe - n_cca}D)")

    K2, eta2_2, sil2, tr2, soj2, kraw2, tsig2, ttot2 = tomato_tukey(
        E_cca_plus, lmp)
    print(f"  K_raw={kraw2} -> K={K2}  eta2={eta2_2:.4f}  sil={sil2:.4f}  "
          f"trans={tr2:.4f}  sojourn={soj2:.0f}h  Tukey={tsig2}/{ttot2}")
    results.append(("G2:CCA+FE residual",
                     K2, eta2_2, sil2, tr2, soj2, tsig2, ttot2))

    # ═══════════════════════════════════════════════════════════
    # H: Procrustes Alignment
    # ═══════════════════════════════════════════════════════════
    print()
    print("-" * 65)
    print("  H: PROCRUSTES ALIGNMENT (align MOMENT to FE space)")
    print("-" * 65)

    t0 = time.time()
    # Procrustes requires same dimensionality — pad MOMENT 2D to 11D with zeros
    mom_padded = np.zeros((n, d_fe), dtype=np.float64)
    mom_padded[:, :d_mom] = mom_std

    # Procrustes: find rotation+scaling that best aligns MOMENT to FE
    _, mom_aligned, disparity = procrustes(fe_std, mom_padded)
    print(f"  Procrustes disparity: {disparity:.4f} "
          f"(0=perfect alignment, 1=no alignment)")

    # Concatenate aligned MOMENT with FE
    E_proc = np.hstack([fe_std, mom_aligned]).astype(np.float32)  # 11+11=22D
    print(f"  Procrustes output: FE {d_fe}D + aligned MOMENT {d_fe}D = "
          f"{E_proc.shape[1]}D")

    K, eta2, sil, tr, soj, kraw, tsig, ttot = tomato_tukey(E_proc, lmp)
    print(f"  K_raw={kraw} -> K={K}  eta2={eta2:.4f}  sil={sil:.4f}  "
          f"trans={tr:.4f}  sojourn={soj:.0f}h  Tukey={tsig}/{ttot}  "
          f"({time.time() - t0:.1f}s)")
    results.append(("H:Procrustes", K, eta2, sil, tr, soj, tsig, ttot))

    # Also try just the aligned MOMENT (without FE)
    # to see if alignment captures FE-like structure
    mom_aligned_only = mom_aligned[:, :d_mom].astype(np.float32)
    K3, eta2_3, sil3, tr3, soj3, kraw3, tsig3, ttot3 = tomato_tukey(
        mom_aligned_only, lmp)
    print(f"  Aligned MOMENT only ({d_mom}D): K={K3}  eta2={eta2_3:.4f}  "
          f"sil={sil3:.4f}  trans={tr3:.4f}  sojourn={soj3:.0f}h")

    # ═══════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("  COMPARISON TABLE")
    print("=" * 75)
    header = (f"{'Config':<32s} {'K':>3s} {'eta2':>7s} {'Sil':>7s} "
              f"{'Tr%':>6s} {'Soj':>6s} {'Tukey':>7s}")
    print(header)
    print("-" * 75)

    # Baselines from saved results
    for name, exp in [("B:FE+ToMATo (baseline)", "exp_FE"),
                       ("C:COMBO naive", "exp_COMBO")]:
        qpath = RESULTS / exp / "step05/quality_report.json"
        if qpath.exists():
            with open(qpath) as f:
                q = json.load(f)
            eco = q["economic"]
            geo = q["geometric"]
            Kq = len(eco["lmp_per_regime"])
            print(f"{name:<32s} {Kq:>3d} {eco['eta_squared']:>7.4f} "
                  f"{geo['silhouette_avg']:>7.4f} {eco['transition_rate']:>6.2f} "
                  f"{eco['sojourn_mean_hours']:>5.0f}h "
                  f"{eco['pairwise_significant']}/{eco['pairwise_total']}")

    # D: Late fusion (from previous run)
    print(f"{'D:Late Fusion (equal weight)':<32s} {'9':>3s} {'0.3947':>7s} "
          f"{'0.0854':>7s} {'0.04':>6s} {'152':>5s}h {'36/36':>7s}")

    # New results
    for name, K, eta2, sil, tr, soj, tsig, ttot in results:
        print(f"{name:<32s} {K:>3d} {eta2:>7.4f} {sil:>7.4f} "
              f"{tr:>6.2f} {soj:>5.0f}h {tsig}/{ttot}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print("=" * 75)


if __name__ == "__main__":
    main()
