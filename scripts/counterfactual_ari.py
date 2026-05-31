"""
Counterfactual ARI experiments: is the orthogonality an artifact of input choice?
1. FE + ACF features -> does ARI with MOMENT rise?
2. MOMENT on non-deseasonalized levels -> does ARI with FE rise?
3. ACF-only clustering -> ARI with both FE and MOMENT

Output: results_darcsinh/split_W512_S6/counterfactual_report.txt
"""
import numpy as np, pandas as pd, torch, warnings, sys, gc
from pathlib import Path
from scipy.stats import studentized_range
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from statsmodels.tsa.seasonal import MSTL
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
W, S = 512, 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

OUT = Path('results_darcsinh/split_W512_S6')

# ── Load data ──
lab = pd.read_parquet(OUT / 'labels.parquet')
pre = pd.read_parquet(OUT / 'preprocessed.parquet')
fe_orig = pd.read_parquet(OUT / 'fe_features.parquet').drop(columns=['datetime']).values

r = pre['r'].values
lmp = pre['lmp'].values
N = len(lab)
starts = list(range(0, len(r) - W + 1, S))

E_orig = lab['regime_E'].values
D_orig = lab['regime_D'].values
lmp_mean = lab['lmp_mean'].values
acf6_orig = lab['acf_6h'].values

# ── Helpers ──
def acf_lag(x, lag):
    n = len(x); m = x.mean(); v = ((x-m)**2).sum()
    if v < 1e-15 or lag >= n: return 0.0
    return float(((x[:n-lag]-m)*(x[lag:]-m)).sum() / v)

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

def run_pipeline(features, merge_target, label):
    print(f'\n  [{label}] DiffMaps ({features.shape[1]}D)...')
    dm, d = diffusion_maps_gpu(features)
    print(f'  [{label}] ToMATo (d={d})...')
    lab_raw, knn, modes = tomato_cluster(dm)
    print(f'  [{label}] Tukey merge ({modes} modes)...')
    lab_m = tukey_merge(lab_raw, merge_target)
    K = len(np.unique(lab_m[lab_m >= 0]))
    print(f'  [{label}] Result: {modes} -> {K} regimes')
    return lab_m, K

# Compute ACF features for all windows
print('Computing ACF features...')
acf_features = np.empty((N, 4), dtype=np.float64)
for i in range(N):
    s = starts[i]
    win = r[s:s+W].astype(np.float64)
    acf_features[i, 0] = acf_lag(win, 1)
    acf_features[i, 1] = acf_lag(win, 6)
    acf_features[i, 2] = acf_lag(win, 24)
    acf_features[i, 3] = acf_lag(win, 168)

report = []
report.append('='*70)
report.append('COUNTERFACTUAL ARI EXPERIMENTS')
report.append('='*70)

# ══════════════════════════════════════════════════════
# EXP 1: FE + ACF features
# ══════════════════════════════════════════════════════
print('\n--- EXP 1: FE + 4 ACF features ---')
fe_plus_acf = np.hstack([fe_orig, acf_features])  # 15 + 4 = 19
fe_plus_acf_std = StandardScaler().fit_transform(fe_plus_acf)

lab_fe_acf, K_fe_acf = run_pipeline(fe_plus_acf_std, lmp_mean, 'FE+ACF')

ari_feacf_dmom = adjusted_rand_score(lab_fe_acf, D_orig)
ari_feacf_eorig = adjusted_rand_score(lab_fe_acf, E_orig)

report.append('\n--- EXP 1: FE enriched with 4 ACF features (19D) ---')
report.append(f'  K = {K_fe_acf}')
report.append(f'  ARI(FE+ACF, D_MOMENT) = {ari_feacf_dmom:.3f}  (baseline FE vs D: {adjusted_rand_score(E_orig, D_orig):.3f})')
report.append(f'  ARI(FE+ACF, E_orig)   = {ari_feacf_eorig:.3f}')
report.append(f'  -> Adding ACF to FE {"raises" if ari_feacf_dmom > 0.05 else "does not raise"} concordance with MOMENT')

# ══════════════════════════════════════════════════════
# EXP 2: MOMENT on raw arcsinh (not deseasonalized)
# ══════════════════════════════════════════════════════
print('\n--- EXP 2: MOMENT on arcsinh(LMP) without MSTL ---')
# Load raw dataset and compute arcsinh
df_raw = pd.read_parquet('isone_dataset.parquet')
arcsinh_raw = np.arcsinh(df_raw['lmp'].values)

# Make windows on raw arcsinh (no MSTL)
starts_raw = list(range(0, len(arcsinh_raw) - W + 1, S))
N_raw = min(len(starts_raw), N)  # should be similar

# Compute MOMENT embeddings on raw arcsinh windows
from momentfm import MOMENTPipeline
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-large', model_kwargs={'task_name': 'embedding'})
model.init(); model = model.to(DEVICE)
BS = 64

with torch.no_grad():
    d_model = model(x_enc=torch.zeros(1,1,W, device=DEVICE)).embeddings.shape[-1]

mom_raw = np.empty((N_raw, d_model), dtype=np.float32)
wr_raw = np.array([arcsinh_raw[starts_raw[i]:starts_raw[i]+W] for i in range(N_raw)], dtype=np.float32)

for s in range(0, N_raw, BS):
    e = min(s + BS, N_raw)
    x = torch.tensor(wr_raw[s:e], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    with torch.no_grad():
        mom_raw[s:e] = model(x_enc=x).embeddings.float().cpu().numpy()
    if (s // BS) % 20 == 0:
        print(f'    {e}/{N_raw}')

del model; gc.collect(); torch.cuda.empty_cache()

# Use first N windows to match
mom_raw = mom_raw[:N]

# Compute ACF6h on raw arcsinh for merge target
acf6_raw = np.array([acf_lag(arcsinh_raw[starts_raw[i]:starts_raw[i]+W].astype(np.float64), 6) for i in range(N)])

lab_mom_raw, K_mom_raw = run_pipeline(mom_raw, acf6_raw, 'MOM-raw')

ari_momraw_eorig = adjusted_rand_score(lab_mom_raw, E_orig)
ari_momraw_dorig = adjusted_rand_score(lab_mom_raw, D_orig)

report.append('\n--- EXP 2: MOMENT on arcsinh(LMP) without MSTL ---')
report.append(f'  K = {K_mom_raw}')
report.append(f'  ARI(MOM-raw, E_orig)   = {ari_momraw_eorig:.3f}  (does it capture price?)')
report.append(f'  ARI(MOM-raw, D_orig)   = {ari_momraw_dorig:.3f}  (does it still capture persistence?)')
report.append(f'  -> Without MSTL, MOMENT {"starts capturing price" if ari_momraw_eorig > 0.05 else "still does not capture price"}')

# ══════════════════════════════════════════════════════
# EXP 3: ACF-only clustering
# ══════════════════════════════════════════════════════
print('\n--- EXP 3: ACF-only clustering ---')
acf_std = StandardScaler().fit_transform(acf_features)
lab_acf_only, K_acf = run_pipeline(acf_std, acf_features[:, 1], 'ACF-only')  # merge on ACF 6h

ari_acf_eorig = adjusted_rand_score(lab_acf_only, E_orig)
ari_acf_dorig = adjusted_rand_score(lab_acf_only, D_orig)

report.append('\n--- EXP 3: ACF-only clustering (4 ACF features) ---')
report.append(f'  K = {K_acf}')
report.append(f'  ARI(ACF-only, E_orig)  = {ari_acf_eorig:.3f}')
report.append(f'  ARI(ACF-only, D_orig)  = {ari_acf_dorig:.3f}')

# ══════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════
report.append('\n' + '='*70)
report.append('SUMMARY: ARI TABLE')
report.append('='*70)
report.append(f'  {"Partition":<25s} {"vs E_orig":>10s} {"vs D_orig":>10s} {"K":>4s}')
report.append(f'  {"-"*25} {"-"*10} {"-"*10} {"-"*4}')
report.append(f'  {"E_orig (baseline)":<25s} {"---":>10s} {adjusted_rand_score(E_orig, D_orig):>10.3f} {9:>4d}')
report.append(f'  {"D_orig (baseline)":<25s} {adjusted_rand_score(D_orig, E_orig):>10.3f} {"---":>10s} {9:>4d}')
report.append(f'  {"FE + 4 ACF":<25s} {ari_feacf_eorig:>10.3f} {ari_feacf_dmom:>10.3f} {K_fe_acf:>4d}')
report.append(f'  {"MOMENT on raw arcsinh":<25s} {ari_momraw_eorig:>10.3f} {ari_momraw_dorig:>10.3f} {K_mom_raw:>4d}')
report.append(f'  {"ACF-only":<25s} {ari_acf_eorig:>10.3f} {ari_acf_dorig:>10.3f} {K_acf:>4d}')
report.append('')
report.append('  Interpretation:')
report.append('  - If FE+ACF vs D_orig >> 0.012: adding ACF breaks orthogonality')
report.append('  - If MOM-raw vs E_orig >> 0.012: removing MSTL breaks orthogonality')
report.append('  - If both stay ~0: orthogonality is robust to input variations')

text = '\n'.join(report)
print('\n' + text)
with open(OUT / 'counterfactual_report.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f'\nSaved to {OUT / "counterfactual_report.txt"}')
