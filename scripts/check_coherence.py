"""
Mechanical coherence check: verify all numbers in the paper text
match pipeline output and internal consistency.
Output: results_darcsinh/split_W512_S6/coherence_report.txt
"""
import numpy as np, pandas as pd, sys, re
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
sys.stdout.reconfigure(encoding='utf-8')

OUT = Path('results_darcsinh/split_W512_S6')
PAPER = Path('paper/paper_v10_it.tex')

lab = pd.read_parquet(OUT / 'labels.parquet')
pre = pd.read_parquet(OUT / 'preprocessed.parquet')
eta_csv = pd.read_csv(OUT / 'eta_squared.csv')

r = pre['r'].values
lmp = pre['lmp'].values
W, S = 512, 6
N = len(lab)
starts = list(range(0, len(r) - W + 1, S))

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

alpha_all = np.array([1.0 - acf_lag(r[starts[i]:starts[i]+W].astype(np.float64), 1) for i in range(N)])

report = []
def check(name, expected, actual, tol=0.01):
    ok = abs(expected - actual) < tol if isinstance(expected, float) else expected == actual
    status = 'OK' if ok else 'MISMATCH'
    report.append(f'  [{status}] {name}: paper={expected}, data={actual}')
    return ok

report.append('='*70)
report.append('COHERENCE CHECK REPORT')
report.append('='*70)

# --- N windows ---
report.append('\n--- Window count ---')
check('N windows', 7217, N)

# --- Table 6: Economic regimes ---
report.append('\n--- Table 6: Economic regimes (n sums to N?) ---')
n_sum_E = sum((E==e).sum() for e in range(9))
check('Sum n(E)', 7217, n_sum_E)

e_order = [8,3,5,0,7,6,4,2,1]
e_lmp_expected = [29.7, 41.2, 49.1, 55.1, 63.9, 84.2, 102.7, 120.0, 151.1]
e_n_expected = [839, 2534, 477, 555, 1643, 815, 136, 153, 65]
for e, lmp_exp, n_exp in zip(e_order, e_lmp_expected, e_n_expected):
    mask = E == e
    check(f'E{e} n', n_exp, int(mask.sum()))
    check(f'E{e} LMP mean', lmp_exp, round(lmp_mean[mask].mean(), 1))

# --- Table 7: Dynamic regimes ---
report.append('\n--- Table 7: Dynamic regimes (n sums to N?) ---')
n_sum_D = sum((D==d).sum() for d in range(9))
check('Sum n(D)', 7217, n_sum_D)

d_order = [7,6,4,5,2,0,1,3,8]
d_acf_expected = [0.380, 0.464, 0.512, 0.607, 0.682, 0.751, 0.796, 0.871, 0.936]
d_n_expected = [381, 587, 119, 250, 2400, 2785, 154, 477, 64]
for d, acf_exp, n_exp in zip(d_order, d_acf_expected, d_n_expected):
    mask = D == d
    check(f'D{d} n', n_exp, int(mask.sum()))
    check(f'D{d} ACF6h mean', acf_exp, round(acf6[mask].mean(), 3))

# --- ARI ---
report.append('\n--- ARI ---')
ari = adjusted_rand_score(E, D)
check('ARI', 0.012, round(ari, 3))

# --- Populated cells ---
report.append('\n--- Grid E x D ---')
cells = 0
for e in range(9):
    for d in range(9):
        if ((E==e)&(D==d)).sum() > 0:
            cells += 1
check('Populated cells', 70, cells)

# --- eta2 from Table 5 ---
report.append('\n--- Table 5: eta2 ---')
eta2_fe_lmp = eta2(lmp_mean, E)
eta2_mom_acf = eta2(acf6, D)
check('eta2_FE(lmp_mean)', 0.495, round(eta2_fe_lmp, 3))
check('eta2_MOM(acf_6h)', 0.420, round(eta2_mom_acf, 3))
# Cross
eta2_fe_acf = eta2(acf6, E)
eta2_mom_lmp = eta2(lmp_mean, D)
check('eta2_FE(acf_6h)', 0.181, round(eta2_fe_acf, 3))
check('eta2_MOM(lmp_mean)', 0.085, round(eta2_mom_lmp, 3))

# --- eta2 on alpha (Section 4.4) ---
report.append('\n--- Section 4.4: eta2 on alpha ---')
eta2_E_alpha = eta2(alpha_all, E)
eta2_D_alpha = eta2(alpha_all, D)
# Joint
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

eta2_ED_alpha = eta2_joint(alpha_all, E, D)
check('eta2(E) on alpha', 0.161, round(eta2_E_alpha, 3))
check('eta2(D) on alpha', 0.376, round(eta2_D_alpha, 3))
check('eta2(E,D) on alpha', 0.466, round(eta2_ED_alpha, 3))

# --- Alpha range in E3 ---
report.append('\n--- Alpha range in E3 (n>=20 cells) ---')
e3_mask = E == 3
alpha_e3 = alpha_all[e3_mask]
check('alpha(E3) mean', 0.078, round(alpha_e3.mean(), 3))

alphas_e3_cells = []
for d in range(9):
    mask = (E==3) & (D==d)
    if mask.sum() >= 20:
        alphas_e3_cells.append(alpha_all[mask].mean())
if alphas_e3_cells:
    check('alpha E3 min (n>=20)', 0.031, round(min(alphas_e3_cells), 3))
    check('alpha E3 max (n>=20)', 0.159, round(max(alphas_e3_cells), 3))

# --- Transition rate ---
report.append('\n--- Transition rate ---')
trans = (E[1:] != E[:-1]).mean()
check('Transition rate', 0.102, round(trans, 3))
dwell = S / trans
check('Dwell time (h)', 59, round(dwell))

# --- DBIC ---
report.append('\n--- DBIC (from model_comparison.csv if available) ---')
mc_path = OUT / 'model_comparison.csv'
if mc_path.exists():
    mc = pd.read_csv(mc_path)
    bic_vals = mc.set_index('model')['BIC']
    if 'phi(E)' in bic_vals.index and 'phi(E,D)' in bic_vals.index:
        dbic = bic_vals['phi(E,D)'] - bic_vals['phi(E)']
        check('DBIC alpha', -3875, round(dbic))

# --- K values ---
report.append('\n--- K values ---')
check('K_E', 9, len(np.unique(E[E>=0])))
check('K_D', 9, len(np.unique(D[D>=0])))

# --- Modes before merge ---
report.append('\n--- Initial modes (from summary.csv) ---')
sm_path = OUT / 'summary.csv'
if sm_path.exists():
    sm = pd.read_csv(sm_path)
    if 'modes_FE' in sm.columns:
        check('FE initial modes', 48, int(sm['modes_FE'].iloc[0]))
    if 'modes_MOM' in sm.columns:
        check('MOM initial modes', 47, int(sm['modes_MOM'].iloc[0]))

# --- Print and save ---
report.append('\n' + '='*70)
n_ok = sum(1 for r in report if '[OK]' in r)
n_mismatch = sum(1 for r in report if '[MISMATCH]' in r)
report.append(f'TOTAL: {n_ok} OK, {n_mismatch} MISMATCH')
report.append('='*70)

text = '\n'.join(report)
print(text)
with open(OUT / 'coherence_report.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f'\nSaved to {OUT / "coherence_report.txt"}')
