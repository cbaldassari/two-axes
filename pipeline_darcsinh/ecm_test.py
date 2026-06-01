"""ECM test: Delta r_t = c - phi * r_{t-1} + eps_t"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kurtosis, kstest, spearmanr
from numpy.linalg import lstsq
from statsmodels.tsa.seasonal import MSTL
import config as C

S = 6
BASE = Path("results_darcsinh/split_W512_S6")

# Load labels
labels = pd.read_parquet(BASE / "labels.parquet")
N_win = len(labels)

# Recompute full MSTL residual
df_raw = pd.read_parquet(Path(C.RESULTS_DIR) / "preprocessed.parquet")
arcsinh = df_raw["arcsinh_lmp"].values
dt = pd.to_datetime(df_raw["datetime"].values)
s = pd.Series(arcsinh, index=pd.DatetimeIndex(dt))
resid = MSTL(s, periods=C.DARCSINH["mstl_periods"]).fit().resid.values
print(f"MSTL residual: n={len(resid)}")

# ECM variables
dr = np.diff(resid)    # Delta r_t, length n-1
r_lag = resid[:-1]     # r_{t-1}, length n-1
n_dr = len(dr)

# Map window labels to hourly on resid indices
h_E = np.full(len(resid), -1, dtype=int)
h_D = np.full(len(resid), -1, dtype=int)
for i in range(N_win):
    # preprocessed df starts at index 1 of resid (dropped first for alignment with dr)
    s0 = i * S + 1
    s1 = min(s0 + S, len(resid))
    h_E[s0:s1] = labels["regime_E"].iloc[i]
    h_D[s0:s1] = labels["regime_D"].iloc[i]
tail = min(N_win * S + 1, len(resid))
if tail < len(resid):
    h_E[tail:] = labels["regime_E"].iloc[-1]
    h_D[tail:] = labels["regime_D"].iloc[-1]

# Align labels with ECM (use label of r_{t-1})
hE = h_E[:-1]
hD = h_D[:-1]
valid = (hE >= 0) & (hD >= 0)
y = dr[valid]
x = r_lag[valid]
hE_v = hE[valid]
hD_v = hD[valid]

print(f"ECM data: n={len(y)} valid hours")

# Global ECM
X_g = np.column_stack([np.ones(len(x)), x])
beta_g, _, _, _ = lstsq(X_g, y, rcond=None)
c_g, neg_phi_g = beta_g
phi_g = -neg_phi_g
eps_g = y - X_g @ beta_g
hl_g = -np.log(2) / np.log(1 - phi_g) if 0 < phi_g < 1 else 9999

print(f"\nECM global: phi={phi_g:.6f}, c={c_g:.6f}, half-life={hl_g:.1f}h")
print(f"  sigma_eps={eps_g.std():.4f}, kurtosis={kurtosis(eps_g, fisher=False):.1f}")


def cell_ecm(mask):
    yc, xc = y[mask], x[mask]
    nc = len(yc)
    if nc < 30:
        return None
    Xc = np.column_stack([np.ones(nc), xc])
    beta, _, _, _ = lstsq(Xc, yc, rcond=None)
    c_cell, neg_phi = beta
    phi = -neg_phi
    ec = yc - Xc @ beta
    sig = float(ec.std())
    k = float(kurtosis(ec, fisher=False))
    hl = -np.log(2) / np.log(1 - phi) if 0 < phi < 1 else 9999.0
    ks_s, ks_p = kstest(ec / sig, "norm") if sig > 1e-15 else (1.0, 0.0)
    return {"n": nc, "phi": round(phi, 6), "hl": round(hl, 1),
            "sig": round(sig, 4), "kurt": round(k, 1), "KS_p": round(ks_p, 4)}


# By E
print(f"\nBy E:")
print(f"{'E':>3s} {'n':>6s} {'phi':>10s} {'hl':>7s} {'sig':>8s} {'kurt':>6s}")
rows_E = []
for e in sorted(np.unique(hE_v)):
    rc = cell_ecm(hE_v == e)
    if rc:
        rc["E"] = e
        rows_E.append(rc)
        print(f"{e:>3d} {rc['n']:>6d} {rc['phi']:>10.6f} {rc['hl']:>7.1f} {rc['sig']:>8.4f} {rc['kurt']:>6.1f}")

# By (E,D)
rows_ED = []
for e in sorted(np.unique(hE_v)):
    for d in sorted(np.unique(hD_v)):
        mask = (hE_v == e) & (hD_v == d)
        if mask.sum() < 30:
            continue
        rc = cell_ecm(mask)
        if rc:
            rc["E"] = e
            rc["D"] = d
            rows_ED.append(rc)

df_E = pd.DataFrame(rows_E)
df_ED = pd.DataFrame(rows_ED)

# phi range within E
print(f"\nphi range within E:")
for e in df_E["E"].values:
    sub = df_ED[df_ED["E"] == e]
    if len(sub) > 1:
        print(f"  E{e}: phi=[{sub['phi'].min():.6f}, {sub['phi'].max():.6f}] "
              f"range={sub['phi'].max()-sub['phi'].min():.6f} "
              f"hl=[{sub['hl'].min():.0f}h, {sub['hl'].max():.0f}h]")

# D -> phi
rho, rho_p = spearmanr(df_ED["D"].values, df_ED["phi"].values)
phi_vals = df_ED["phi"].values
d_labels = df_ED["D"].values
gm = phi_vals.mean()
ss_t = ((phi_vals - gm) ** 2).sum()
ss_b = sum(len(phi_vals[d_labels == d]) * (phi_vals[d_labels == d].mean() - gm) ** 2
           for d in np.unique(d_labels))
eta2_D = ss_b / ss_t if ss_t > 1e-15 else 0
print(f"\nD -> phi: eta2={eta2_D:.3f}, Spearman rho={rho:.3f} (p={rho_p:.6f})")

# Kurtosis cascade
kurt_E = df_E["kurt"].mean()
kurt_ED = df_ED["kurt"].mean()
pass_E = (df_E["KS_p"] > 0.05).mean() * 100
pass_ED = (df_ED["KS_p"] > 0.05).mean() * 100
print(f"\nKurtosis: global={kurtosis(eps_g, fisher=False):.1f} -> by E={kurt_E:.1f} -> by (E,D)={kurt_ED:.1f}")
print(f"KS pass: by E={pass_E:.0f}% -> by (E,D)={pass_ED:.0f}%")

# BIC
def loglik(res):
    n = len(res)
    s2 = res.var()
    return -0.5 * n * (np.log(2 * np.pi * s2) + 1)

eps_E_all = []
for _, row in df_E.iterrows():
    e = int(row["E"])
    mask = hE_v == e
    Xc = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta, _, _, _ = lstsq(Xc, y[mask], rcond=None)
    eps_E_all.append(y[mask] - Xc @ beta)

eps_ED_all = []
for _, row in df_ED.iterrows():
    e, d = int(row["E"]), int(row["D"])
    mask = (hE_v == e) & (hD_v == d)
    Xc = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta, _, _, _ = lstsq(Xc, y[mask], rcond=None)
    eps_ED_all.append(y[mask] - Xc @ beta)

eps_E_cat = np.concatenate(eps_E_all)
ll_E = loglik(eps_E_cat)
k_E = 2 * len(df_E)

eps_ED_cat = np.concatenate(eps_ED_all)
ll_ED = loglik(eps_ED_cat)
k_ED = 2 * len(df_ED)

n_obs = len(eps_E_cat)
bic_E = -2 * ll_E + k_E * np.log(n_obs)
bic_ED = -2 * ll_ED + k_ED * np.log(n_obs)
aic_E = -2 * ll_E + 2 * k_E
aic_ED = -2 * ll_ED + 2 * k_ED

print(f"\n{'='*50}")
print(f"MODEL COMPARISON (ECM on r_t)")
print(f"{'='*50}")
print(f"  phi(E):   params={k_E}, AIC={aic_E:.0f}, BIC={bic_E:.0f}")
print(f"  phi(E,D): params={k_ED}, AIC={aic_ED:.0f}, BIC={bic_ED:.0f}")
print(f"  dAIC={aic_ED - aic_E:+.0f}, dBIC={bic_ED - bic_E:+.0f}")
winner = "ED wins" if bic_ED < bic_E else "E wins"
print(f"  {winner} on BIC")
print(f"\nCells: {len(df_E)} E, {len(df_ED)} (E,D)")

# Save
df_E.to_csv(BASE / "ecm_by_E.csv", index=False)
df_ED.to_csv(BASE / "ecm_by_ED.csv", index=False)
