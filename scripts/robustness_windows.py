"""
Robustness check: do results hold with non-overlapping windows?
Compares N=7217 (S=6) vs ~85 independent windows (S=512) vs N_eff corrected.
Output: results_darcsinh/split_W512_S6/robustness_report.txt
"""
import numpy as np, pandas as pd, sys
from pathlib import Path
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')

OUT = Path('results_darcsinh/split_W512_S6')

lab = pd.read_parquet(OUT / 'labels.parquet')
pre = pd.read_parquet(OUT / 'preprocessed.parquet')

r = pre['r'].values
lmp = pre['lmp'].values
W, S = 512, 6
N = len(lab)
starts_S6 = list(range(0, len(r) - W + 1, S))

E = lab['regime_E'].values
D = lab['regime_D'].values
lmp_mean = lab['lmp_mean'].values
acf6 = lab['acf_6h'].values

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

def eta2_joint(vals, lE, lD):
    valid = (lE >= 0) & (lD >= 0)
    a, eL, dL = vals[valid], lE[valid], lD[valid]
    gm = a.mean(); ss_t = ((a-gm)**2).sum()
    if ss_t < 1e-15: return 0.0
    ss_b = 0
    for e in np.unique(eL):
        for d in np.unique(dL):
            mask = (eL==e)&(dL==d)
            if mask.sum() < 1: continue
            ss_b += mask.sum()*(a[mask].mean()-gm)**2
    return ss_b/ss_t

def compute_bic(alpha_vals, lab_E, lab_D, n_eff=None):
    valid = (lab_E >= 0) & (lab_D >= 0)
    a, eL, dL = alpha_vals[valid], lab_E[valid], lab_D[valid]
    n_params = 0; ll = 0.0; n_total = len(a)
    if n_eff is None: n_eff = n_total
    for e in np.unique(eL):
        for d in np.unique(dL):
            mask = (eL == e) & (dL == d)
            if mask.sum() < 5: continue
            ac = a[mask]; sig = ac.std()
            if sig < 1e-15: sig = 1e-15
            ll += -0.5 * len(ac) * (np.log(2*np.pi*sig**2) + 1)
            n_params += 2
    bic = -2*ll + n_params * np.log(n_eff)
    return bic, n_params

# Compute alpha for all windows
alpha_all = np.array([1.0 - acf_lag(r[starts_S6[i]:starts_S6[i]+W].astype(np.float64), 1)
                      for i in range(N)])

report = []
report.append('='*70)
report.append('ROBUSTNESS CHECK: OVERLAPPING vs INDEPENDENT WINDOWS')
report.append('='*70)

# ══════════════════════════════════════════════════════
# SCENARIO 1: All N=7217 windows (baseline, as in paper)
# ══════════════════════════════════════════════════════
report.append('\n--- SCENARIO 1: All windows (N=7217, S=6) ---')
e2_E = eta2(alpha_all, E)
e2_D = eta2(alpha_all, D)
e2_ED = eta2_joint(alpha_all, E, D)
bic_E, k_E = compute_bic(alpha_all, E, np.zeros_like(E))
bic_ED, k_ED = compute_bic(alpha_all, E, D)
dbic = bic_ED - bic_E
report.append(f'  N = {N}')
report.append(f'  eta2(E) = {e2_E:.3f}, eta2(D) = {e2_D:.3f}, eta2(E,D) = {e2_ED:.3f}')
report.append(f'  BIC(E) = {bic_E:.0f} (k={k_E}), BIC(E,D) = {bic_ED:.0f} (k={k_ED})')
report.append(f'  DBIC = {dbic:.0f}')

# ══════════════════════════════════════════════════════
# SCENARIO 2: Non-overlapping windows (S=W=512)
# ══════════════════════════════════════════════════════
report.append('\n--- SCENARIO 2: Non-overlapping windows (S=512) ---')
# Take every (W/S)th window = every 85th-ish
step = W // S  # 512/6 ~ 85
idx_indep = list(range(0, N, step))
N_indep = len(idx_indep)

E_ind = E[idx_indep]
D_ind = D[idx_indep]
alpha_ind = alpha_all[idx_indep]

e2_E_ind = eta2(alpha_ind, E_ind)
e2_D_ind = eta2(alpha_ind, D_ind)
e2_ED_ind = eta2_joint(alpha_ind, E_ind, D_ind)

# BIC on independent subsample
bic_E_ind, k_E_ind = compute_bic(alpha_ind, E_ind, np.zeros_like(E_ind))
bic_ED_ind, k_ED_ind = compute_bic(alpha_ind, E_ind, D_ind)
dbic_ind = bic_ED_ind - bic_E_ind

report.append(f'  N_indep = {N_indep} (step={step})')
report.append(f'  eta2(E) = {e2_E_ind:.3f}, eta2(D) = {e2_D_ind:.3f}, eta2(E,D) = {e2_ED_ind:.3f}')
report.append(f'  BIC(E) = {bic_E_ind:.0f} (k={k_E_ind}), BIC(E,D) = {bic_ED_ind:.0f} (k={k_ED_ind})')
report.append(f'  DBIC = {dbic_ind:.0f}')

# ══════════════════════════════════════════════════════
# SCENARIO 3: All windows, BIC with N_eff
# ══════════════════════════════════════════════════════
report.append('\n--- SCENARIO 3: All windows, BIC corrected with N_eff ---')

# Estimate N_eff from autocorrelation of alpha series
# Using Bartlett formula: N_eff = N / (1 + 2 * sum_k rho(k))
rho_sum = 0
max_lag = min(500, N//2)
for lag in range(1, max_lag+1):
    rho_k = np.corrcoef(alpha_all[:-lag], alpha_all[lag:])[0,1]
    if abs(rho_k) < 2/np.sqrt(N):  # truncate at significance
        break
    rho_sum += rho_k

n_eff = N / (1 + 2 * rho_sum)
report.append(f'  Autocorrelation of alpha: first {lag} lags significant')
report.append(f'  Sum of autocorrelations: {rho_sum:.1f}')
report.append(f'  N_eff (Bartlett) = {n_eff:.0f} (vs N={N})')
report.append(f'  Inflation factor: {N/n_eff:.1f}x')

bic_E_eff, _ = compute_bic(alpha_all, E, np.zeros_like(E), n_eff=n_eff)
bic_ED_eff, _ = compute_bic(alpha_all, E, D, n_eff=n_eff)
dbic_eff = bic_ED_eff - bic_E_eff

report.append(f'  BIC_eff(E) = {bic_E_eff:.0f}, BIC_eff(E,D) = {bic_ED_eff:.0f}')
report.append(f'  DBIC_eff = {dbic_eff:.0f}')

# ══════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════
report.append('\n' + '='*70)
report.append('SUMMARY TABLE')
report.append('='*70)
report.append(f'  {"Scenario":<30s} {"N":>6s} {"eta2(E)":>8s} {"eta2(D)":>8s} {"eta2(ED)":>9s} {"DBIC":>8s} {"Conclusion":>12s}')
report.append(f'  {"-"*30} {"-"*6} {"-"*8} {"-"*8} {"-"*9} {"-"*8} {"-"*12}')
report.append(f'  {"All windows (paper)":<30s} {N:>6d} {e2_E:>8.3f} {e2_D:>8.3f} {e2_ED:>9.3f} {dbic:>8.0f} {"2-axis wins":>12s}')
report.append(f'  {"Independent (S=W)":<30s} {N_indep:>6d} {e2_E_ind:>8.3f} {e2_D_ind:>8.3f} {e2_ED_ind:>9.3f} {dbic_ind:>8.0f} {"2-axis wins" if dbic_ind < -10 else "uncertain":>12s}')
report.append(f'  {"All, BIC with N_eff":<30s} {int(n_eff):>6d} {e2_E:>8.3f} {e2_D:>8.3f} {e2_ED:>9.3f} {dbic_eff:>8.0f} {"2-axis wins" if dbic_eff < -10 else "uncertain":>12s}')
report.append('')
report.append(f'  Note: eta2 does not depend on N (it is a ratio), only DBIC changes.')
report.append(f'  "2-axis wins" = DBIC < -10 (Kass & Raftery strong evidence threshold)')

text = '\n'.join(report)
print(text)
with open(OUT / 'robustness_report.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f'\nSaved to {OUT / "robustness_report.txt"}')
