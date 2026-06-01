"""
Dual-axis regime detection.

  arcsinh(p_t) -> MSTL(24,168,8760) -> r_t
  FE receives dr_t (stationary innovations) — order-invariant stats need stationarity
  MOMENT receives r_t (persistent process) — sequence model needs temporal structure

  --mode split (default): FE on dr_t, MOMENT on r_t
  --mode r:  both on r_t (exploratory baseline)
  --mode dr: both on dr_t (validation — confirms MOMENT needs temporal structure)

Usage:
  python pipeline_darcsinh/run_pipeline.py
  python pipeline_darcsinh/run_pipeline.py --mode r
  python pipeline_darcsinh/run_pipeline.py --skip-moment
"""
from __future__ import annotations
import argparse, gc, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew, kurtosis, kstest, studentized_range
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from statsmodels.tsa.seasonal import MSTL

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

W, S, SEED = C.DARCSINH["W"], C.DARCSINH["S"], C.RANDOM_STATE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _t(t0, label):
    print(f"    [{time.time()-t0:6.1f}s] {label}", flush=True)


# ---------- preprocessing ----------

def preprocess(df):
    dt = df["datetime"].values
    lmp = df["lmp"].values
    arcsinh = df["arcsinh_lmp"].values

    # 1. MSTL: remove deterministic components (seasonality, trend)
    t0 = time.time()
    s = pd.Series(arcsinh, index=pd.DatetimeIndex(dt))
    resid = MSTL(s, periods=C.DARCSINH["mstl_periods"]).fit().resid.values
    acf1r = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    _t(t0, f"MSTL residual: n={len(resid)}, ACF1={acf1r:.3f}")

    # 2. Dual branch: r_t for MOMENT, delta_r_t for FE
    dr = np.diff(resid)
    r = resid[1:]       # align with dr (both length n-1)
    dt, lmp = dt[1:], lmp[1:]
    acf1dr = np.corrcoef(dr[:-1], dr[1:])[0, 1]
    _t(t0, f"Delta r: n={len(dr)}, ACF1={acf1dr:.3f}")

    return pd.DataFrame({"datetime": dt, "lmp": lmp, "r": r, "dr": dr})


# ---------- sliding windows ----------

def make_windows(vals, lmp, dt):
    starts = list(range(0, len(vals) - W + 1, S))
    wv = np.array([vals[s:s+W] for s in starts], dtype=np.float32)
    wl = np.array([lmp[s:s+W] for s in starts], dtype=np.float32)
    ts = np.array([dt[s+W-1] for s in starts])
    return wv, wl, ts


# ---------- FE (15 features, no ACF) ----------

def compute_fe(wr, wl):
    N = len(wr)
    fe = np.empty((N, 15), dtype=np.float32)
    for i in range(N):
        r, l = wr[i].astype(np.float64), wl[i].astype(np.float64)
        fe[i, 0] = r.mean()
        fe[i, 1] = r.std()
        fe[i, 2] = float(skew(r))
        fe[i, 3] = float(kurtosis(r, fisher=False))
        fe[i, 4] = r.min()
        fe[i, 5] = r.max()
        fe[i, 6] = r.max() - r.min()
        fe[i, 7] = float(np.median(r))
        fe[i, 8] = float(np.percentile(r, 5))
        fe[i, 9] = float(np.percentile(r, 95))
        fe[i, 10] = float(np.percentile(r, 75) - np.percentile(r, 25))
        nf = (len(r) // 24) * 24
        fe[i, 11] = float(np.abs(np.diff(r[:nf].reshape(-1,24), axis=1)).mean()) if nf >= 24 else float(np.abs(np.diff(r)).mean())
        fe[i, 12] = l.mean()
        fe[i, 13] = float(np.percentile(l, 95))
        fe[i, 14] = l.std()
    return fe


# ---------- MOMENT ----------

def compute_moment(wr):
    from momentfm import MOMENTPipeline
    # Ensure deterministic behavior
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"    Loading MOMENT on {DEVICE} ...", flush=True)
    model = MOMENTPipeline.from_pretrained(
        C.EMBEDDING["model"], model_kwargs={"task_name": "embedding"})
    model.init(); model = model.to(DEVICE)
    N, bs = len(wr), C.EMBEDDING["batch_size"]
    with torch.no_grad():
        d_model = model(x_enc=torch.zeros(1,1,W, device=DEVICE)).embeddings.shape[-1]
    emb = np.empty((N, d_model), dtype=np.float32)
    for s in range(0, N, bs):
        e = min(s + bs, N)
        x = torch.tensor(wr[s:e], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            emb[s:e] = model(x_enc=x).embeddings.float().cpu().numpy()
        if (s // bs) % 20 == 0:
            print(f"      {e}/{N}", flush=True)
    del model, x
    gc.collect(); torch.cuda.empty_cache()
    return emb


# ---------- Diffusion Maps (GPU) ----------

def diffusion_maps(X, label="", fixed_d=None):
    t0 = time.time()
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
    all_coords = (evecs[:, 1:21] * evals[1:21]).cpu().numpy()
    del Xt, dists, K, P, evals, evecs; torch.cuda.empty_cache()
    _t(t0, f"DiffMaps {label} kernel+eigh done")

    if fixed_d is not None:
        best_d, best_s = fixed_d, 0.0
        cd = all_coords[:, :fixed_d]
        nc = max(2, min(10, len(cd) // 50))
        km = KMeans(n_clusters=nc, n_init=5, random_state=SEED).fit(cd)
        best_s = silhouette_score(cd, km.labels_)
        _t(t0, f"DiffMaps {label} fixed d={fixed_d} (sil={best_s:.3f})")
    else:
        t0 = time.time()
        best_d, best_s = 2, -1
        for d in range(2, 21):
            cd = all_coords[:, :d]
            nc = max(2, min(10, len(cd) // 50))
            km = KMeans(n_clusters=nc, n_init=5, random_state=SEED).fit(cd)
            s = silhouette_score(cd, km.labels_)
            if s > best_s: best_d, best_s = d, s
        _t(t0, f"DiffMaps {label} silhouette sweep -> d={best_d} (sil={best_s:.3f})")
    return all_coords[:, :best_d], best_d, best_s


# ---------- ToMATo ----------

def tomato_cluster(X):
    from gudhi.clustering.tomato import Tomato
    best_lab, best_k, best_n = None, None, 0
    for k in [20, 40, 60, 80, 100, 150]:
        tmt = Tomato(density_type="KDE", graph_type="knn", n_neighbors=k)
        tmt.fit(X)
        if hasattr(tmt, "diagram_") and len(tmt.diagram_) > 1:
            deaths = np.sort([d for _, d in tmt.diagram_ if d < np.inf])
            if len(deaths) > 1:
                n = len(deaths) - np.argmax(np.diff(deaths))
                tmt.n_clusters_ = n
            else: n = 1
        else: n = 1
        if n > best_n: best_n, best_lab, best_k = n, tmt.labels_.copy(), k
    return best_lab, best_k, best_n


# ---------- Tukey merge ----------

def tukey_merge(labels, target, alpha=0.05, min_size=20):
    """Tukey HSD merge -- pure numpy, no statsmodels."""
    m = labels.copy()

    def _relabel(m):
        for j, v in enumerate(np.unique(m[m >= 0])): m[m == v] = j
        return m

    # Phase 1: absorb tiny modes into nearest neighbor
    u, counts = np.unique(m[m >= 0], return_counts=True)
    for k, c in sorted(zip(u, counts), key=lambda x: x[1]):
        if c < min_size and len(np.unique(m[m >= 0])) > 1:
            mk = target[m == k].mean()
            others = [kk for kk in np.unique(m[m >= 0]) if kk != k]
            nearest = min(others, key=lambda kk: abs(target[m == kk].mean() - mk))
            m[m == nearest] = np.where(m[m == nearest] == nearest, nearest, m[m == nearest])
            m[m == k] = nearest
    m = _relabel(m)
    print(f"      After absorb: {len(np.unique(m[m>=0]))} modes", flush=True)

    # Phase 2: vectorized Tukey HSD merge
    while True:
        mask = m >= 0
        x, z = target[mask], m[mask]
        u = np.unique(z)
        K = len(u)
        if K <= 1: break
        N_total = len(x)

        g_means = np.array([x[z == k].mean() for k in u])
        g_ns = np.array([np.sum(z == k) for k in u])

        ss_within = sum(((x[z == k] - x[z == k].mean()) ** 2).sum() for k in u)
        df_within = N_total - K
        if df_within <= 0: break
        mse = ss_within / df_within

        order = np.argsort(g_means)
        g_means_s = g_means[order]
        g_ns_s = g_ns[order]
        u_s = u[order]

        q_crit = studentized_range.ppf(1 - alpha, K, df_within)

        best_pair, best_diff = None, np.inf
        for i in range(K):
            for j in range(i + 1, K):
                diff = abs(g_means_s[i] - g_means_s[j])
                se = np.sqrt(mse * 0.5 * (1.0/g_ns_s[i] + 1.0/g_ns_s[j]))
                q_stat = diff / se if se > 1e-15 else np.inf
                if q_stat < q_crit and diff < best_diff:
                    best_diff = diff
                    best_pair = (u_s[i], u_s[j])

        if best_pair is None: break
        m[m == best_pair[1]] = best_pair[0]
        m = _relabel(m)

    return m


# ---------- helpers ----------

def _acf(x, lag):
    n = len(x); m = x.mean(); v = ((x-m)**2).sum()
    if v < 1e-15 or lag >= n: return 0.0
    return float(((x[:n-lag]-m)*(x[lag:]-m)).sum() / v)

def eta2(vals, labels):
    mask = labels >= 0; x, z = vals[mask], labels[mask]
    gm = x.mean(); ss_t = ((x-gm)**2).sum()
    if ss_t < 1e-15: return 0.0
    ss_b = sum(len(x[z==k]) * (x[z==k].mean()-gm)**2 for k in np.unique(z))
    return float(ss_b / ss_t)


# ---------- AR(1) validation ----------

def ar1_validation(lmp, lab_E, lab_D, N_win, out):
    """Global AR(1) on hourly LMP (levels); phi, sigma(Dp), kurtosis by E and (E,D)."""
    from scipy.stats import spearmanr
    n = len(lmp)

    # Map window labels to hourly (stride-block assignment)
    h_E = np.full(n, -1, dtype=int)
    h_D = np.full(n, -1, dtype=int)
    for i in range(N_win):
        s0, s1 = i * S, min((i + 1) * S, n)
        h_E[s0:s1], h_D[s0:s1] = lab_E[i], lab_D[i]
    tail = min(N_win * S, n)
    if tail < n:
        h_E[tail:], h_D[tail:] = lab_E[-1], lab_D[-1]

    # Global AR(1): p_t = c + phi * p_{t-1} + eps_t
    y, x = lmp[1:], lmp[:-1]
    dp = y - x  # price changes
    cov_m = np.cov(y, x, ddof=1)
    phi_g = cov_m[0, 1] / cov_m[1, 1]
    c_g = y.mean() - phi_g * x.mean()
    eps = y - c_g - phi_g * x

    hE, hD = h_E[1:], h_D[1:]  # align with residuals
    valid = (hE >= 0) & (hD >= 0)

    kurt_g = float(kurtosis(eps, fisher=False))
    sig_dp_g = float(dp.std())
    hl_g = -np.log(2) / np.log(abs(phi_g)) if abs(phi_g) < 1 else np.inf
    print(f"\n  AR(1) global: c={c_g:.4f}, phi={phi_g:.6f}, sigma_eps={eps.std():.2f}")
    print(f"  sigma(Dp)={sig_dp_g:.2f}, half-life={hl_g:.1f}h, kurtosis={kurt_g:.1f}", flush=True)

    # Helper: compute per-cell stats
    y_v, x_v, dp_v, eps_v = y[valid], x[valid], dp[valid], eps[valid]
    hE_v, hD_v = hE[valid], hD[valid]

    def _cell(mask):
        yc, xc, dc, ec = y_v[mask], x_v[mask], dp_v[mask], eps_v[mask]
        nc = len(yc)
        if nc < 30: return None
        var_x = xc.var(ddof=1)
        if var_x < 1e-15: return None
        phi_c = np.cov(yc, xc, ddof=1)[0, 1] / var_x
        sig_dp = float(dc.std())
        sig_eps = float(ec.std())
        k = float(kurtosis(ec, fisher=False))
        hl = -np.log(2) / np.log(abs(phi_c)) if 0 < abs(phi_c) < 1 else 9999.0
        ks_s, ks_p = kstest(ec / sig_eps, 'norm') if sig_eps > 1e-15 else (1.0, 0.0)
        return {"n": nc, "phi": round(phi_c, 4), "half_life": round(hl, 1),
                "sig_dp": round(sig_dp, 2), "sig_eps": round(sig_eps, 2),
                "kurt": round(k, 1), "KS": round(ks_s, 3), "KS_p": round(ks_p, 4)}

    # --- By E ---
    rows_E = []
    for e in np.sort(np.unique(hE_v)):
        row = _cell(hE_v == e)
        if row: row["E"] = int(e); rows_E.append(row)
    df_E = pd.DataFrame(rows_E)[["E","n","phi","half_life","sig_dp","sig_eps","kurt","KS","KS_p"]]

    # --- By (E, D) ---
    rows_ED = []
    for e in np.sort(np.unique(hE_v)):
        for dd in np.sort(np.unique(hD_v)):
            mask = (hE_v == e) & (hD_v == dd)
            if mask.sum() < 30: continue
            row = _cell(mask)
            if row: row["E"] = int(e); row["D"] = int(dd); rows_ED.append(row)
    df_ED = pd.DataFrame(rows_ED)[["E","D","n","phi","half_life","sig_dp","sig_eps","kurt","KS","KS_p"]]

    # Save
    df_E.to_csv(out / "ar1_by_E.csv", index=False)
    df_ED.to_csv(out / "ar1_by_ED.csv", index=False)

    # Print
    print(f"\n  AR(1) by E ({len(df_E)} groups):", flush=True)
    print("  " + df_E.to_string(index=False).replace("\n", "\n  "), flush=True)
    print(f"\n  AR(1) by (E,D) ({len(df_ED)} cells):", flush=True)
    print("  " + df_ED.to_string(index=False).replace("\n", "\n  "), flush=True)

    # Summary: kurtosis cascade
    kurt_E = df_E["kurt"].mean() if len(df_E) > 0 else kurt_g
    kurt_ED = df_ED["kurt"].mean() if len(df_ED) > 0 else kurt_g
    pass_E = (df_E["KS_p"] > 0.05).mean() * 100 if len(df_E) > 0 else 0
    pass_ED = (df_ED["KS_p"] > 0.05).mean() * 100 if len(df_ED) > 0 else 0
    print(f"\n  Kurtosis: global={kurt_g:.1f} -> by E={kurt_E:.1f} -> by (E,D)={kurt_ED:.1f}")
    print(f"  KS normality pass: by E={pass_E:.0f}% -> by (E,D)={pass_ED:.0f}%", flush=True)

    # Summary: phi and sig_dp range within each E
    if len(df_ED) > 1:
        print(f"\n  phi / sig_dp range within E:", flush=True)
        for e in df_E["E"].values:
            sub = df_ED[df_ED["E"] == e]
            if len(sub) > 1:
                print(f"    E{e}: phi=[{sub['phi'].min():.4f}, {sub['phi'].max():.4f}] "
                      f"range={sub['phi'].max()-sub['phi'].min():.4f}  "
                      f"sig_dp=[{sub['sig_dp'].min():.2f}, {sub['sig_dp'].max():.2f}]", flush=True)

        # eta-squared of D on phi (unweighted, each cell = 1 obs)
        phi_vals = df_ED["phi"].values
        d_labels = df_ED["D"].values
        gm = phi_vals.mean(); ss_t = ((phi_vals - gm)**2).sum()
        eta2_D = 0.0
        if ss_t > 1e-15:
            ss_b = sum(len(phi_vals[d_labels==d])*(phi_vals[d_labels==d].mean()-gm)**2
                       for d in np.unique(d_labels))
            eta2_D = ss_b / ss_t
        rho, rho_p = spearmanr(d_labels, phi_vals)
        print(f"\n  D -> phi: eta2={eta2_D:.3f}, Spearman rho={rho:.3f} (p={rho_p:.4f})", flush=True)

    # ── Model comparison: phi(E) vs phi(E,D) ──
    # Build per-E and per-(E,D) parameter lookup
    phi_E_map, c_E_map = {}, {}
    for _, row in df_E.iterrows():
        e = int(row["E"])
        mask = hE_v == e
        yc, xc = y_v[mask], x_v[mask]
        var_x = xc.var(ddof=1)
        phi_E_map[e] = np.cov(yc, xc, ddof=1)[0, 1] / var_x if var_x > 1e-15 else 0
        c_E_map[e] = yc.mean() - phi_E_map[e] * xc.mean()

    phi_ED_map, c_ED_map = {}, {}
    for _, row in df_ED.iterrows():
        e, d = int(row["E"]), int(row["D"])
        mask = (hE_v == e) & (hD_v == d)
        yc, xc = y_v[mask], x_v[mask]
        var_x = xc.var(ddof=1)
        phi_ED_map[(e, d)] = np.cov(yc, xc, ddof=1)[0, 1] / var_x if var_x > 1e-15 else 0
        c_ED_map[(e, d)] = yc.mean() - phi_ED_map[(e, d)] * xc.mean()

    # Compute residuals for both models
    eps_E = np.full(len(y_v), np.nan)
    eps_ED = np.full(len(y_v), np.nan)
    for i in range(len(y_v)):
        e, d = int(hE_v[i]), int(hD_v[i])
        if e in phi_E_map:
            eps_E[i] = y_v[i] - c_E_map[e] - phi_E_map[e] * x_v[i]
        if (e, d) in phi_ED_map:
            eps_ED[i] = y_v[i] - c_ED_map[(e, d)] - phi_ED_map[(e, d)] * x_v[i]

    valid_E = ~np.isnan(eps_E)
    valid_ED = ~np.isnan(eps_ED)

    # Log-likelihood (Gaussian)
    def _loglik(residuals):
        n = len(residuals)
        sigma2 = residuals.var()
        return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

    n_E, n_ED = valid_E.sum(), valid_ED.sum()
    ll_E = _loglik(eps_E[valid_E])
    ll_ED = _loglik(eps_ED[valid_ED])
    k_E = 2 * len(phi_E_map)      # c + phi per E
    k_ED = 2 * len(phi_ED_map)    # c + phi per (E,D)
    aic_E = -2 * ll_E + 2 * k_E
    aic_ED = -2 * ll_ED + 2 * k_ED
    bic_E = -2 * ll_E + k_E * np.log(n_E)
    bic_ED = -2 * ll_ED + k_ED * np.log(n_ED)

    kurt_resid_E = float(kurtosis(eps_E[valid_E], fisher=False))
    kurt_resid_ED = float(kurtosis(eps_ED[valid_ED], fisher=False))
    rmse_E = float(np.sqrt((eps_E[valid_E]**2).mean()))
    rmse_ED = float(np.sqrt((eps_ED[valid_ED]**2).mean()))

    print(f"\n  {'='*50}")
    print(f"  MODEL COMPARISON: phi(E) vs phi(E,D)")
    print(f"  {'='*50}")
    print(f"  {'':>20s}  {'phi(E)':>12s}  {'phi(E,D)':>12s}")
    print(f"  {'params':>20s}  {k_E:>12d}  {k_ED:>12d}")
    print(f"  {'RMSE':>20s}  {rmse_E:>12.2f}  {rmse_ED:>12.2f}")
    print(f"  {'kurtosis':>20s}  {kurt_resid_E:>12.1f}  {kurt_resid_ED:>12.1f}")
    print(f"  {'log-likelihood':>20s}  {ll_E:>12.0f}  {ll_ED:>12.0f}")
    print(f"  {'AIC':>20s}  {aic_E:>12.0f}  {aic_ED:>12.0f}")
    print(f"  {'BIC':>20s}  {bic_E:>12.0f}  {bic_ED:>12.0f}")
    delta_aic = aic_ED - aic_E
    delta_bic = bic_ED - bic_E
    print(f"  {'delta AIC (ED-E)':>20s}  {delta_aic:>+12.0f}  {'<-- ED wins' if delta_aic < 0 else '<-- E wins'}")
    print(f"  {'delta BIC (ED-E)':>20s}  {delta_bic:>+12.0f}  {'<-- ED wins' if delta_bic < 0 else '<-- E wins'}")
    print(flush=True)

    # Save comparison
    comp = {"model": ["phi(E)", "phi(E,D)"],
            "params": [k_E, k_ED], "RMSE": [round(rmse_E,2), round(rmse_ED,2)],
            "kurtosis": [round(kurt_resid_E,1), round(kurt_resid_ED,1)],
            "loglik": [round(ll_E,0), round(ll_ED,0)],
            "AIC": [round(aic_E,0), round(aic_ED,0)],
            "BIC": [round(bic_E,0), round(bic_ED,0)]}
    pd.DataFrame(comp).to_csv(out / "model_comparison.csv", index=False)

    return {"phi_global": round(phi_g, 6), "c": round(c_g, 4), "sig_dp_global": round(sig_dp_g, 2),
            "kurt_global": round(kurt_g, 1), "kurt_E": round(kurt_E, 1), "kurt_ED": round(kurt_ED, 1),
            "pass_E": round(pass_E, 1), "pass_ED": round(pass_ED, 1),
            "rmse_E": round(rmse_E, 2), "rmse_ED": round(rmse_ED, 2),
            "aic_E": round(aic_E, 0), "aic_ED": round(aic_ED, 0),
            "bic_E": round(bic_E, 0), "bic_ED": round(bic_ED, 0)}


# ---------- Monte Carlo moment comparison ----------

def simulate_moments(lmp, hE, hD, df_E, df_ED, out, n_sim=1000, seed=42):
    """Simulate AR(1) on LMP from phi(E) and phi(E,D), compare Dp moments to empirical."""
    rng = np.random.default_rng(seed)
    n = len(lmp)

    valid = (hE >= 0) & (hD >= 0)

    # Build parameter maps from LMP
    y, x = lmp[1:], lmp[:-1]
    hE_y, hD_y = hE[1:], hD[1:]
    valid_y = valid[1:] & valid[:-1]

    phi_E, c_E, sig_E = {}, {}, {}
    for _, row in df_E.iterrows():
        e = int(row["E"])
        mask = (hE_y == e) & valid_y
        if mask.sum() < 30: continue
        yc, xc = y[mask], x[mask]
        var_x = xc.var(ddof=1)
        if var_x < 1e-15: continue
        phi_E[e] = np.cov(yc, xc, ddof=1)[0, 1] / var_x
        c_E[e] = yc.mean() - phi_E[e] * xc.mean()
        sig_E[e] = float((yc - c_E[e] - phi_E[e] * xc).std())

    phi_ED, c_ED, sig_ED = {}, {}, {}
    for _, row in df_ED.iterrows():
        e, d = int(row["E"]), int(row["D"])
        mask = (hE_y == e) & (hD_y == d) & valid_y
        if mask.sum() < 30: continue
        yc, xc = y[mask], x[mask]
        var_x = xc.var(ddof=1)
        if var_x < 1e-15: continue
        phi_ED[(e, d)] = np.cov(yc, xc, ddof=1)[0, 1] / var_x
        c_ED[(e, d)] = yc.mean() - phi_ED[(e, d)] * xc.mean()
        sig_ED[(e, d)] = float((yc - c_ED[(e, d)] - phi_ED[(e, d)] * xc).std())

    # Simulate AR(1) on LMP
    def _simulate(phi_map, c_map, sig_map, key_fn):
        p_sim = np.zeros((n_sim, n))
        p_sim[:, 0] = lmp[0]
        eps = rng.standard_normal((n_sim, n))
        phi_gb = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
        c_gb = y.mean() - phi_gb * x.mean()
        sig_gb = float((y - c_gb - phi_gb * x).std())
        for t in range(1, n):
            key = key_fn(t)
            if key in phi_map:
                p_sim[:, t] = c_map[key] + phi_map[key] * p_sim[:, t-1] + sig_map[key] * eps[:, t]
            else:
                p_sim[:, t] = c_gb + phi_gb * p_sim[:, t-1] + sig_gb * eps[:, t]
        return p_sim

    print(f"\n  Simulating {n_sim} trajectories...", flush=True)
    t0 = time.time()
    p_E = _simulate(phi_E, c_E, sig_E, lambda t: int(hE[t]))
    p_ED = _simulate(phi_ED, c_ED, sig_ED, lambda t: (int(hE[t]), int(hD[t])))
    _t(t0, "Monte Carlo done")

    # Compute Dp
    dp_emp = np.diff(lmp)
    dp_E = np.diff(p_E, axis=1)
    dp_ED = np.diff(p_ED, axis=1)
    hE_dp = hE[1:]

    # --- Aggregate moments ---
    agg_emp = {"mean": dp_emp.mean(), "std": dp_emp.std(),
               "skew": float(skew(dp_emp)), "kurt": float(kurtosis(dp_emp, fisher=False))}
    agg_E = {"mean": float(dp_E.mean()), "std": float(dp_E.std()),
             "skew": float(skew(dp_E.ravel())), "kurt": float(kurtosis(dp_E.ravel(), fisher=False))}
    agg_ED = {"mean": float(dp_ED.mean()), "std": float(dp_ED.std()),
              "skew": float(skew(dp_ED.ravel())), "kurt": float(kurtosis(dp_ED.ravel(), fisher=False))}

    print(f"\n  {'='*50}")
    print(f"  AGGREGATE Dp MOMENTS")
    print(f"  {'='*50}")
    print(f"  {'':>12s}  {'Empirical':>10s}  {'phi(E)':>10s}  {'phi(E,D)':>10s}")
    for m in ["mean", "std", "skew", "kurt"]:
        print(f"  {m:>12s}  {agg_emp[m]:>10.2f}  {agg_E[m]:>10.2f}  {agg_ED[m]:>10.2f}")

    # --- Per-E kurtosis ---
    print(f"\n  {'='*50}")
    print(f"  WITHIN-E KURTOSIS of Dp (empirical vs simulated)")
    print(f"  {'='*50}")
    print(f"  {'E':>4s}  {'n':>6s}  {'Empirical':>10s}  {'phi(E)':>10s}  {'phi(E,D)':>10s}  {'Winner':>8s}")

    rows_per_E = []
    for e in np.sort(np.unique(hE_dp[hE_dp >= 0])):
        mask = hE_dp == e
        if mask.sum() < 50: continue
        k_emp = float(kurtosis(dp_emp[mask], fisher=False))
        k_sim_E = float(np.mean([kurtosis(dp_E[s, mask], fisher=False) for s in range(n_sim)]))
        k_sim_ED = float(np.mean([kurtosis(dp_ED[s, mask], fisher=False) for s in range(n_sim)]))
        err_E = abs(k_emp - k_sim_E)
        err_ED = abs(k_emp - k_sim_ED)
        winner = "ED" if err_ED < err_E else "E" if err_E < err_ED else "tie"
        rows_per_E.append({"E": int(e), "n": int(mask.sum()), "kurt_emp": round(k_emp, 1),
                           "kurt_phi_E": round(k_sim_E, 1), "kurt_phi_ED": round(k_sim_ED, 1),
                           "winner": winner})
        print(f"  {e:>4d}  {mask.sum():>6d}  {k_emp:>10.1f}  {k_sim_E:>10.1f}  {k_sim_ED:>10.1f}  {winner:>8s}")

    df_perE = pd.DataFrame(rows_per_E)
    n_ED_wins = (df_perE["winner"] == "ED").sum()
    n_E_wins = (df_perE["winner"] == "E").sum()
    print(f"\n  Score: phi(E,D) wins {n_ED_wins}/{len(df_perE)}, phi(E) wins {n_E_wins}/{len(df_perE)}")
    print(flush=True)

    df_perE.to_csv(out / "moment_comparison_per_E.csv", index=False)
    pd.DataFrame([{"moment": m, "empirical": round(agg_emp[m], 4),
                    "phi_E": round(agg_E[m], 4), "phi_ED": round(agg_ED[m], 4)}
                   for m in ["mean", "std", "skew", "kurt"]]).to_csv(out / "moment_comparison.csv", index=False)
    return df_perE


# ---------- run ----------

def run_one(df_raw, mode="split", skip_moment=False):
    out = Path("results_darcsinh") / f"{mode}_W{W}_S{S}"
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()

    print(f"\n{'='*60}\n  Dual-axis pipeline — mode={mode}\n{'='*60}", flush=True)

    # 1. Preprocess
    df = preprocess(df_raw)
    df.to_parquet(out / "preprocessed.parquet", index=False)

    # 2. Windows
    t0 = time.time()
    if mode == "split":
        wr_fe, wl, ts = make_windows(df["dr"].values, df["lmp"].values, df["datetime"].values)
        wr_mom, _, _ = make_windows(df["r"].values, df["lmp"].values, df["datetime"].values)
        _label = "FE on dr_t, MOMENT on r_t"
    else:
        signal = df[mode].values
        wr_fe, wl, ts = make_windows(signal, df["lmp"].values, df["datetime"].values)
        wr_mom = wr_fe
        _label = f"both on {mode}_t"
    N = len(wr_fe)
    _t(t0, f"Windows: N={N} ({_label})")

    # 3. FE
    t0 = time.time()
    fe = compute_fe(wr_fe, wl)
    pd.DataFrame(fe, columns=C.FE_FEATURES_15).assign(datetime=ts).to_parquet(
        out / "fe_features.parquet", index=False)
    _t(t0, f"FE: {fe.shape}")

    # 4. MOMENT
    mp = out / "moment_embeddings.parquet"
    if skip_moment and mp.exists():
        t0 = time.time()
        mom = pd.read_parquet(mp).drop(columns=["datetime"]).values
        _t(t0, f"MOMENT cached: {mom.shape}")
    else:
        t0 = time.time()
        mom = compute_moment(wr_mom)
        pd.DataFrame(mom, columns=[f"mom_{i}" for i in range(mom.shape[1])]).assign(
            datetime=ts).to_parquet(mp, index=False)
        _t(t0, f"MOMENT: {mom.shape}")

    # 5. Diffusion Maps (FE: silhouette sweep; MOM: fixed d=5)
    dm_fe, d_fe, s_fe = diffusion_maps(StandardScaler().fit_transform(fe), "FE")
    dm_mom, d_mom, s_mom = diffusion_maps(mom, "MOM", fixed_d=5)

    # 6. ToMATo
    t0 = time.time()
    lab_fe, knn_fe, modes_fe = tomato_cluster(dm_fe)
    lab_mom, knn_mom, modes_mom = tomato_cluster(dm_mom)
    _t(t0, f"ToMATo: FE={modes_fe} modes (k={knn_fe}), MOM={modes_mom} modes (k={knn_mom})")

    # 7. Merge targets
    t0 = time.time()
    lmp_mean = np.array([wl[i].mean() for i in range(N)])
    acf6 = np.array([_acf(wr_mom[i].astype(np.float64), 6) for i in range(N)])
    _t(t0, "Merge targets computed")

    # 8. Tukey merge
    t0 = time.time()
    lab_fe_m = tukey_merge(lab_fe, lmp_mean)
    lab_mom_m = tukey_merge(lab_mom, acf6)
    K_fe = len(np.unique(lab_fe_m[lab_fe_m >= 0]))
    K_mom = len(np.unique(lab_mom_m[lab_mom_m >= 0]))
    _t(t0, f"Tukey: FE {modes_fe}->{K_fe}, MOM {modes_mom}->{K_mom}")

    # 9. eta-squared
    print(f"\n  eta-squared:", flush=True)
    rows = []
    for j, name in enumerate(C.FE_FEATURES_15):
        e_fe, e_mom = eta2(fe[:, j], lab_fe_m), eta2(fe[:, j], lab_mom_m)
        rows.append({"feature": name, "eta2_FE": round(e_fe, 3), "eta2_MOM": round(e_mom, 3)})
        print(f"    {name:>12s}  FE={e_fe:.3f}  MOM={e_mom:.3f}", flush=True)
    for lag, name in [(1, "acf_1h"), (6, "acf_6h"), (24, "acf_24h"), (168, "acf_168h")]:
        vals = np.array([_acf(wr_mom[i].astype(np.float64), lag) for i in range(N)])
        e_fe, e_mom = eta2(vals, lab_fe_m), eta2(vals, lab_mom_m)
        rows.append({"feature": name, "eta2_FE": round(e_fe, 3), "eta2_MOM": round(e_mom, 3)})
        print(f"    {name:>12s}  FE={e_fe:.3f}  MOM={e_mom:.3f}  (diag)", flush=True)
    pd.DataFrame(rows).to_csv(out / "eta_squared.csv", index=False)

    # 10. ARI
    ari = adjusted_rand_score(lab_fe_m, lab_mom_m)
    print(f"\n  ARI = {ari:.3f}", flush=True)

    # 11. AR(1) validation + model comparison
    t0 = time.time()
    ar1 = ar1_validation(df["lmp"].values, lab_fe_m, lab_mom_m, N, out)
    _t(t0, "AR(1) validation")

    # 12. Monte Carlo moment comparison
    t0 = time.time()
    n_h = len(df["lmp"].values)
    h_E = np.full(n_h, -1, dtype=int)
    h_D = np.full(n_h, -1, dtype=int)
    for i in range(N):
        s0, s1 = i * S, min((i + 1) * S, n_h)
        h_E[s0:s1], h_D[s0:s1] = lab_fe_m[i], lab_mom_m[i]
    tail = min(N * S, n_h)
    if tail < n_h:
        h_E[tail:], h_D[tail:] = lab_fe_m[-1], lab_mom_m[-1]
    df_E_csv = pd.read_csv(out / "ar1_by_E.csv")
    df_ED_csv = pd.read_csv(out / "ar1_by_ED.csv")
    simulate_moments(df["lmp"].values, h_E, h_D, df_E_csv, df_ED_csv, out)
    _t(t0, "Monte Carlo moments")

    # 13. Save
    pd.DataFrame({"datetime": ts, "regime_E": lab_fe_m, "regime_D": lab_mom_m,
                   "lmp_mean": lmp_mean, "acf_6h": acf6}).to_parquet(
        out / "labels.parquet", index=False)

    summary = {"N": N, "d_FE": d_fe, "d_MOM": d_mom,
               "sil_FE": round(s_fe, 3), "sil_MOM": round(s_mom, 3),
               "knn_FE": knn_fe, "knn_MOM": knn_mom,
               "modes_FE": modes_fe, "modes_MOM": modes_mom,
               "K_FE": K_fe, "K_MOM": K_mom, "ARI": round(ari, 3),
               **ar1}
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)

    _t(T0, "TOTAL")
    return summary


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["split", "r", "dr"], default="split",
                   help="split (default): FE on dr, MOM on r. r/dr: both on same signal.")
    p.add_argument("--skip-moment", action="store_true")
    a = p.parse_args()

    df = pd.read_parquet(Path(C.RESULTS_DIR) / "preprocessed.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])

    run_one(df, mode=a.mode, skip_moment=a.skip_moment)

if __name__ == "__main__":
    main()
