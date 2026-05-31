"""
ACF baseline vs MOMENT comparison.
Answers: does MOMENT add value beyond simple ACF-based clustering?
"""
import numpy as np, pandas as pd, torch, warnings, sys
from scipy.stats import studentized_range, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
W, S = 512, 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── Load data ──
lab = pd.read_parquet('results_darcsinh/split_W512_S6/labels.parquet')
pre = pd.read_parquet('results_darcsinh/split_W512_S6/preprocessed.parquet')

r = pre['r'].values
lmp = pre['lmp'].values
N = len(lab)
starts = list(range(0, len(r) - W + 1, S))

# ── Helper functions ──
def acf_lag(x, lag):
    n = len(x); m = x.mean(); v = ((x-m)**2).sum()
    if v < 1e-15 or lag >= n: return 0.0
    return float(((x[:n-lag]-m)*(x[lag:]-m)).sum() / v)

def eta2(vals, labels):
    mask = labels >= 0; x, z = vals[mask], labels[mask]
    gm = x.mean(); ss_t = ((x-gm)**2).sum()
    if ss_t < 1e-15: return 0.0
    ss_b = sum(len(x[z==k])*(x[z==k].mean()-gm)**2 for k in np.unique(z))
    return float(ss_b/ss_t)

def tomato_cluster(X):
    from gudhi.clustering.tomato import Tomato
    best_lab, best_k, best_n = None, None, 0
    for k in [20, 40, 60, 80, 100, 150]:
        tmt = Tomato(density_type='KDE', graph_type='knn', n_neighbors=k)
        tmt.fit(X)
        if hasattr(tmt, 'diagram_') and len(tmt.diagram_) > 1:
            deaths = np.sort([d for _, d in tmt.diagram_ if d < np.inf])
            if len(deaths) > 1:
                n = len(deaths) - np.argmax(np.diff(deaths))
                tmt.n_clusters_ = n
            else: n = 1
        else: n = 1
        if n > best_n: best_n, best_lab, best_k = n, tmt.labels_.copy(), k
    return best_lab, best_k, best_n

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
        mask = m >= 0; x, z = target[mask], m[mask]
        u = np.unique(z); K = len(u)
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
        m[m == best_pair[1]] = best_pair[0]
        m = _relabel(m)
    return m

def diffusion_maps_gpu(X, fixed_d=None):
    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > 50:
        Xs = PCA(n_components=50, random_state=SEED).fit_transform(Xs)
    Xt = torch.tensor(Xs, dtype=torch.float64, device=DEVICE)
    dists = torch.cdist(Xt, Xt)
    eps = float(torch.median(dists[dists > 0]).item()) ** 2
    K = torch.exp(-dists**2 / eps)
    P = torch.diag(1.0/K.sum(dim=1)) @ K
    evals, evecs = torch.linalg.eigh(P)
    evals, evecs = evals.flip(0), evecs.flip(1)
    all_coords = (evecs[:, 1:21] * evals[1:21]).cpu().numpy()
    del Xt, dists, K, P; torch.cuda.empty_cache()
    if fixed_d is not None:
        return all_coords[:, :fixed_d], fixed_d
    best_d, best_s = 2, -1
    for d in range(2, 21):
        cd = all_coords[:, :d]
        nc = max(2, min(10, len(cd)//50))
        km = KMeans(n_clusters=nc, n_init=5, random_state=SEED).fit(cd)
        s = silhouette_score(cd, km.labels_)
        if s > best_s: best_d, best_s = d, s
    return all_coords[:, :best_d], best_d

# ══════════════════════════════════════════════════════════════
# STEP 1: Compute ACF feature vector for each window
# ══════════════════════════════════════════════════════════════
print('\nStep 1: Computing ACF features per window...')
acf_features = np.empty((N, 5), dtype=np.float64)
for i in range(N):
    s = starts[i]
    win = r[s:s+W].astype(np.float64)
    acf_features[i, 0] = acf_lag(win, 1)
    acf_features[i, 1] = acf_lag(win, 6)
    acf_features[i, 2] = acf_lag(win, 24)
    acf_features[i, 3] = acf_lag(win, 168)
    alpha_i = 1.0 - acf_features[i, 0]
    acf_features[i, 4] = -np.log(2)/np.log(abs(1-alpha_i)) if 0 < abs(1-alpha_i) < 1 else 9999.0

acf6 = acf_features[:, 1]
alpha_all = 1.0 - acf_features[:, 0]
lmp_mean = np.array([lmp[starts[i]:starts[i]+W].mean() for i in range(N)])
print(f'  ACF features shape: {acf_features.shape}')

# ══════════════════════════════════════════════════════════════
# STEP 2: Run same pipeline on ACF features
# ══════════════════════════════════════════════════════════════
print('\nStep 2: Diffusion Maps on ACF features...')
dm_acf, d_acf = diffusion_maps_gpu(StandardScaler().fit_transform(acf_features))
print(f'  DiffMaps ACF: d={d_acf}')

print('Step 2b: ToMATo on ACF diffusion coords...')
lab_acf_raw, knn_acf, modes_acf = tomato_cluster(dm_acf)
print(f'  ToMATo ACF: {modes_acf} modes (k={knn_acf})')

print('Step 2c: Tukey merge on ACF 6h...')
lab_acf_m = tukey_merge(lab_acf_raw, acf6)
K_acf = len(np.unique(lab_acf_m[lab_acf_m >= 0]))
print(f'  Tukey ACF: {modes_acf} -> {K_acf} regimes')

# Load MOMENT partition
lab_mom_m = lab['regime_D'].values
lab_fe_m = lab['regime_E'].values
K_mom = len(np.unique(lab_mom_m[lab_mom_m >= 0]))

# ══════════════════════════════════════════════════════════════
# STEP 3: Compare D_ACF with D_MOMENT
# ══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('STEP 3: COMPARISON D_ACF vs D_MOMENT')
print('='*60)

ari_acf_mom = adjusted_rand_score(lab_acf_m, lab_mom_m)
print(f'  ARI(D_ACF, D_MOMENT) = {ari_acf_mom:.3f}')
print(f'  K_ACF = {K_acf}, K_MOMENT = {K_mom}')

eta2_acf_alpha = eta2(alpha_all, lab_acf_m)
eta2_mom_alpha = eta2(alpha_all, lab_mom_m)
eta2_fe_alpha = eta2(alpha_all, lab_fe_m)
print(f'  eta2(D_ACF) on alpha  = {eta2_acf_alpha:.3f}')
print(f'  eta2(D_MOM) on alpha  = {eta2_mom_alpha:.3f}')
print(f'  eta2(E)     on alpha  = {eta2_fe_alpha:.3f}')

eta2_acf_acf6 = eta2(acf6, lab_acf_m)
eta2_mom_acf6 = eta2(acf6, lab_mom_m)
print(f'  eta2(D_ACF) on ACF6h  = {eta2_acf_acf6:.3f}')
print(f'  eta2(D_MOM) on ACF6h  = {eta2_mom_acf6:.3f}')

eta2_acf_lmp = eta2(lmp_mean, lab_acf_m)
eta2_mom_lmp = eta2(lmp_mean, lab_mom_m)
print(f'  eta2(D_ACF) on LMP    = {eta2_acf_lmp:.3f}')
print(f'  eta2(D_MOM) on LMP    = {eta2_mom_lmp:.3f}')

# BIC comparison
def compute_bic(alpha_vals, lab_E, lab_D):
    valid = (lab_E >= 0) & (lab_D >= 0)
    a, eL, dL = alpha_vals[valid], lab_E[valid], lab_D[valid]
    n_params = 0; ll = 0.0; n_total = len(a)
    for e in np.unique(eL):
        for d in np.unique(dL):
            mask = (eL == e) & (dL == d)
            if mask.sum() < 5: continue
            ac = a[mask]; mu = ac.mean(); sig = ac.std()
            if sig < 1e-15: sig = 1e-15
            ll += -0.5 * len(ac) * (np.log(2*np.pi*sig**2) + 1)
            n_params += 2
    bic = -2*ll + n_params * np.log(n_total)
    return bic, n_params, ll

bic_e_only, k_e, _ = compute_bic(alpha_all, lab_fe_m, np.zeros_like(lab_fe_m))
bic_e_dacf, k_ea, _ = compute_bic(alpha_all, lab_fe_m, lab_acf_m)
bic_e_dmom, k_em, _ = compute_bic(alpha_all, lab_fe_m, lab_mom_m)

print(f'\n  BIC model E only:     {bic_e_only:12.0f}  (k={k_e})')
print(f'  BIC model (E, D_ACF): {bic_e_dacf:12.0f}  (k={k_ea})')
print(f'  BIC model (E, D_MOM): {bic_e_dmom:12.0f}  (k={k_em})')
print(f'  DBIC (E,D_ACF) - E:   {bic_e_dacf - bic_e_only:+.0f}')
print(f'  DBIC (E,D_MOM) - E:   {bic_e_dmom - bic_e_only:+.0f}')
print(f'  DBIC (E,D_ACF) vs (E,D_MOM): {bic_e_dacf - bic_e_dmom:+.0f}')

rho_acf, p_acf = spearmanr(lab_acf_m, alpha_all)
rho_mom, p_mom = spearmanr(lab_mom_m, alpha_all)
print(f'\n  Spearman D_ACF -> alpha: rho={rho_acf:.3f} (p={p_acf:.1e})')
print(f'  Spearman D_MOM -> alpha: rho={rho_mom:.3f} (p={p_mom:.1e})')

# ══════════════════════════════════════════════════════════════
# STEP 4: Does MOMENT capture structure BEYOND ACF?
# ══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('STEP 4: DOES MOMENT CAPTURE STRUCTURE BEYOND ACF?')
print('='*60)

# Partial: regress alpha on D_ACF dummies, take residual
alpha_resid = alpha_all.copy()
for k in np.unique(lab_acf_m[lab_acf_m >= 0]):
    mask = lab_acf_m == k
    alpha_resid[mask] -= alpha_all[mask].mean()

eta2_mom_alpha_resid = eta2(alpha_resid, lab_mom_m)
eta2_acf_alpha_resid = eta2(alpha_resid, lab_acf_m)
print(f'  eta2(D_ACF) on alpha residual (sanity ~0): {eta2_acf_alpha_resid:.4f}')
print(f'  eta2(D_MOM) on alpha residual (after ACF): {eta2_mom_alpha_resid:.4f}')
print(f'  -> MOMENT explains {eta2_mom_alpha_resid*100:.1f}% additional alpha variance')

ari_acf_e = adjusted_rand_score(lab_acf_m, lab_fe_m)
ari_mom_e = adjusted_rand_score(lab_mom_m, lab_fe_m)
print(f'\n  ARI(D_ACF, E) = {ari_acf_e:.3f}  (orthogonality with price)')
print(f'  ARI(D_MOM, E) = {ari_mom_e:.3f}')

def eta2_joint(alpha_vals, lab_E, lab_D):
    valid = (lab_E >= 0) & (lab_D >= 0)
    a, eL, dL = alpha_vals[valid], lab_E[valid], lab_D[valid]
    gm = a.mean(); ss_t = ((a-gm)**2).sum()
    if ss_t < 1e-15: return 0.0
    ss_b = 0
    for e in np.unique(eL):
        for d in np.unique(dL):
            mask = (eL==e) & (dL==d)
            if mask.sum() < 1: continue
            ss_b += mask.sum() * (a[mask].mean() - gm)**2
    return ss_b / ss_t

eta2_joint_acf = eta2_joint(alpha_all, lab_fe_m, lab_acf_m)
eta2_joint_mom = eta2_joint(alpha_all, lab_fe_m, lab_mom_m)
print(f'\n  eta2(E, D_ACF) on alpha = {eta2_joint_acf:.3f}')
print(f'  eta2(E, D_MOM) on alpha = {eta2_joint_mom:.3f}')
print(f'  Gain D_ACF over E alone: +{(eta2_joint_acf - eta2_fe_alpha)*100:.1f}pp')
print(f'  Gain D_MOM over E alone: +{(eta2_joint_mom - eta2_fe_alpha)*100:.1f}pp')

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
if ari_acf_mom > 0.5:
    print(f'  D_ACF and D_MOMENT are SIMILAR (ARI={ari_acf_mom:.3f})')
else:
    print(f'  D_ACF and D_MOMENT are DIFFERENT (ARI={ari_acf_mom:.3f})')
print(f'  Both capture persistence (eta2 ACF6h: ACF={eta2_acf_acf6:.3f}, MOM={eta2_mom_acf6:.3f})')
print(f'  Both orthogonal to price (ARI with E: ACF={ari_acf_e:.3f}, MOM={ari_mom_e:.3f})')
print(f'  MOMENT extra alpha variance: {eta2_mom_alpha_resid*100:.1f}%')
print(f'  BIC advantage MOMENT over ACF: {bic_e_dacf - bic_e_dmom:+.0f}')
