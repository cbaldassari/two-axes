"""
Sweep over MOMENT axis hyperparameters: ACF enrichment, DiffMaps d, ToMATo k_nn.
Uses cached MOMENT embeddings + FE features. Reports ΔBIC for each config.
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import kurtosis, kstest, studentized_range, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

W, S, SEED = C.DARCSINH["W"], C.DARCSINH["S"], C.RANDOM_STATE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = Path("results_darcsinh/split_W512_S6")


def _acf(x, lag):
    n = len(x); m = x.mean(); v = ((x-m)**2).sum()
    if v < 1e-15 or lag >= n: return 0.0
    return float(((x[:n-lag]-m)*(x[lag:]-m)).sum() / v)


def diffusion_maps_fixed(X, d):
    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > 50:
        Xs = PCA(n_components=50, random_state=SEED).fit_transform(Xs)
    Xt = torch.tensor(Xs, dtype=torch.float64, device=DEVICE)
    dists = torch.cdist(Xt, Xt)
    eps = float(torch.median(dists[dists > 0]).item()) ** 2
    K = torch.exp(-dists ** 2 / eps)
    P = torch.diag(1.0 / K.sum(dim=1)) @ K
    evals, evecs = torch.linalg.eigh(P)
    evals, evecs = evals.flip(0), evecs.flip(1)
    coords = (evecs[:, 1:d+1] * evals[1:d+1]).cpu().numpy()
    del Xt, dists, K, P, evals, evecs; torch.cuda.empty_cache()
    return coords


def tomato_fixed(X, k_nn):
    from gudhi.clustering.tomato import Tomato
    tmt = Tomato(density_type="KDE", graph_type="knn", n_neighbors=k_nn)
    tmt.fit(X)
    if hasattr(tmt, "diagram_") and len(tmt.diagram_) > 1:
        deaths = np.sort([d for _, d in tmt.diagram_ if d < np.inf])
        if len(deaths) > 1:
            n = len(deaths) - np.argmax(np.diff(deaths))
            tmt.n_clusters_ = n
        else: n = 1
    else: n = 1
    return tmt.labels_.copy(), n


def tukey_merge(labels, target, alpha=0.05, min_size=20):
    m = labels.copy()
    def _relabel(m):
        for j, v in enumerate(np.unique(m[m >= 0])): m[m == v] = j
        return m
    u, counts = np.unique(m[m >= 0], return_counts=True)
    for k, c in sorted(zip(u, counts), key=lambda x: x[1]):
        if c < min_size and len(np.unique(m[m >= 0])) > 1:
            mk = target[m == k].mean()
            others = [kk for kk in np.unique(m[m >= 0]) if kk != k]
            nearest = min(others, key=lambda kk: abs(target[m == kk].mean() - mk))
            m[m == k] = nearest
    m = _relabel(m)
    while True:
        mask = m >= 0; x, z = target[mask], m[mask]; u = np.unique(z); K = len(u)
        if K <= 1: break
        N_total = len(x)
        g_means = np.array([x[z==k].mean() for k in u])
        g_ns = np.array([np.sum(z==k) for k in u])
        ss_within = sum(((x[z==k]-x[z==k].mean())**2).sum() for k in u)
        df_within = N_total - K
        if df_within <= 0: break
        mse = ss_within / df_within
        order = np.argsort(g_means)
        g_means_s, g_ns_s, u_s = g_means[order], g_ns[order], u[order]
        q_crit = studentized_range.ppf(1-alpha, K, df_within)
        best_pair, best_diff = None, np.inf
        for i in range(K):
            for j in range(i+1, K):
                diff = abs(g_means_s[i]-g_means_s[j])
                se = np.sqrt(mse*0.5*(1.0/g_ns_s[i]+1.0/g_ns_s[j]))
                q_stat = diff/se if se > 1e-15 else np.inf
                if q_stat < q_crit and diff < best_diff:
                    best_diff = diff; best_pair = (u_s[i], u_s[j])
        if best_pair is None: break
        m[m == best_pair[1]] = best_pair[0]; m = _relabel(m)
    return m


def compute_bic(lmp, lab_E, lab_D, N_win):
    n = len(lmp)
    h_E = np.full(n, -1, dtype=int); h_D = np.full(n, -1, dtype=int)
    for i in range(N_win):
        s0, s1 = i*S, min((i+1)*S, n)
        h_E[s0:s1], h_D[s0:s1] = lab_E[i], lab_D[i]
    tail = min(N_win*S, n)
    if tail < n: h_E[tail:], h_D[tail:] = lab_E[-1], lab_D[-1]

    y, x = lmp[1:], lmp[:-1]
    hE, hD = h_E[1:], h_D[1:]
    valid = (hE >= 0) & (hD >= 0)
    y_v, x_v, hE_v, hD_v = y[valid], x[valid], hE[valid], hD[valid]

    def _loglik_model(key_fn, keys):
        eps_all = []
        n_params = 0
        for key in keys:
            mask = np.array([key_fn(i) == key for i in range(len(y_v))])
            if mask.sum() < 30: continue
            yc, xc = y_v[mask], x_v[mask]
            var_x = xc.var(ddof=1)
            if var_x < 1e-15: continue
            phi = np.cov(yc, xc, ddof=1)[0,1] / var_x
            c = yc.mean() - phi * xc.mean()
            eps_all.append(yc - c - phi * xc)
            n_params += 2
        eps = np.concatenate(eps_all)
        n_obs = len(eps)
        sigma2 = eps.var()
        ll = -0.5 * n_obs * (np.log(2*np.pi*sigma2) + 1)
        return ll, n_params, n_obs, float(kurtosis(eps, fisher=False))

    # Model E
    keys_E = sorted(np.unique(hE_v))
    ll_E, k_E, n_E, kurt_E = _loglik_model(lambda i: hE_v[i], keys_E)
    bic_E = -2*ll_E + k_E*np.log(n_E)

    # Model ED
    keys_ED = sorted(set(zip(hE_v, hD_v)))
    ll_ED, k_ED, n_ED, kurt_ED = _loglik_model(lambda i: (hE_v[i], hD_v[i]), keys_ED)
    bic_ED = -2*ll_ED + k_ED*np.log(n_ED)

    # D -> phi
    phi_vals, d_labels = [], []
    for key in keys_ED:
        e, d = key
        mask = (hE_v == e) & (hD_v == d)
        if mask.sum() < 30: continue
        yc, xc = y_v[mask], x_v[mask]
        var_x = xc.var(ddof=1)
        if var_x < 1e-15: continue
        phi_vals.append(np.cov(yc, xc, ddof=1)[0,1] / var_x)
        d_labels.append(d)
    rho, rho_p = spearmanr(d_labels, phi_vals) if len(phi_vals) > 3 else (0, 1)

    K_D = len(np.unique(lab_D[lab_D >= 0]))
    n_cells = k_ED // 2

    return {
        "K_D": K_D, "cells": n_cells, "k_E": k_E, "k_ED": k_ED,
        "bic_E": round(bic_E), "bic_ED": round(bic_ED), "delta_bic": round(bic_ED - bic_E),
        "kurt_E": round(kurt_E, 1), "kurt_ED": round(kurt_ED, 1),
        "rho": round(rho, 3), "rho_p": round(rho_p, 4)
    }


def main():
    # Load cached data
    mom = pd.read_parquet(BASE / "moment_embeddings.parquet").drop(columns=["datetime"]).values
    labels_fe = pd.read_parquet(BASE / "labels.parquet")
    lab_fe_m = labels_fe["regime_E"].values
    pre = pd.read_parquet(BASE / "preprocessed.parquet")
    lmp = pre["lmp"].values
    N = len(labels_fe)

    # Compute per-window ACFs on r_t
    r = pre["r"].values
    starts = list(range(0, len(r) - W + 1, S))
    wr_mom = np.array([r[s:s+W] for s in starts], dtype=np.float32)
    acf_features = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        w = wr_mom[i].astype(np.float64)
        acf_features[i, 0] = _acf(w, 1)
        acf_features[i, 1] = _acf(w, 6)
        acf_features[i, 2] = _acf(w, 24)
        acf_features[i, 3] = _acf(w, 168)

    acf6 = acf_features[:, 1]

    # Sweep
    results = []
    configs = []
    for enrich in [False, True]:
        for d in [3, 5, 7, 10]:
            for k_nn in [10, 15, 20]:
                configs.append((enrich, d, k_nn))

    print(f"Testing {len(configs)} configurations...\n")
    print(f"{'enrich':>7s} {'d':>3s} {'k_nn':>5s} | {'K_D':>4s} {'cells':>6s} {'ΔBIC':>7s} {'kurt_E':>7s} {'kurt_ED':>8s} {'ρ(D,φ)':>7s} {'p':>7s}")
    print("-" * 80)

    for enrich, d, k_nn in configs:
        t0 = time.time()

        # Build representation
        if enrich:
            X = np.hstack([mom, acf_features * 10])  # scale ACF to match MOMENT magnitude
        else:
            X = mom

        # DiffMaps
        dm = diffusion_maps_fixed(X, d)

        # ToMATo
        lab, modes = tomato_fixed(dm, k_nn)

        # Tukey merge
        lab_m = tukey_merge(lab, acf6)
        K_D = len(np.unique(lab_m[lab_m >= 0]))

        if K_D < 2:
            print(f"{'Y' if enrich else 'N':>7s} {d:>3d} {k_nn:>5d} | {K_D:>4d}   SKIP (K_D < 2)")
            continue

        # BIC
        r = compute_bic(lmp, lab_fe_m, lab_m, N)
        r["enrich"] = enrich
        r["d"] = d
        r["k_nn"] = k_nn
        r["modes"] = modes
        results.append(r)

        dt = time.time() - t0
        print(f"{'Y' if enrich else 'N':>7s} {d:>3d} {k_nn:>5d} | {r['K_D']:>4d} {r['cells']:>6d} {r['delta_bic']:>+7d} "
              f"{r['kurt_E']:>7.1f} {r['kurt_ED']:>8.1f} {r['rho']:>7.3f} {r['rho_p']:>7.4f}  [{dt:.1f}s]")

    # Summary
    df = pd.DataFrame(results).sort_values("delta_bic")
    print(f"\n{'='*60}")
    print(f"TOP 5 configurations by ΔBIC:")
    print(df.head().to_string(index=False))
    df.to_csv("results_darcsinh/sweep_results.csv", index=False)
    print(f"\nSaved to results_darcsinh/sweep_results.csv")


if __name__ == "__main__":
    main()
