"""
regime_switching_joint.py
=========================
Regime-switching model on the full 9x8 = 72 joint (E,D) state space.

1. Estimates joint 72x72 transition matrix from training data (2021-2024)
2. Estimates per-cell AR(1)+jump parameters
3. Simulates 1000 paths on 2025 test set
4. Compares joint model vs Kronecker (independent) factorization
5. Saves all results to results/dual_axis/
"""
from __future__ import annotations
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
OUT_DIR = RESULTS_DIR / "dual_axis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2024-12-31"
N_SIM = 1000
SEED = 42
n_E, n_D = 9, 8
n_states = n_E * n_D


def load_and_prepare():
    fe = pd.read_parquet(RESULTS_DIR / "exp_FE" / "step04" / "labels.parquet")
    mom = pd.read_parquet(RESULTS_DIR / "exp_C" / "step04" / "labels.parquet")
    pre = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    lmp_lookup = pre.set_index("datetime")["lmp"]
    fe_feats = pd.read_parquet(RESULTS_DIR / "exp_FE" / "embeddings.parquet")

    df = pd.DataFrame({
        "datetime": pd.to_datetime(fe["datetime"]),
        "E": fe["cluster"].values,
        "D": mom["cluster"].values,
    })
    df["lmp"] = lmp_lookup.reindex(df["datetime"]).values
    df["acf_6h"] = fe_feats["acf_6h"].values

    # Order E by mean LMP, D by mean ACF_6h
    e_means = df.groupby("E")["lmp"].mean().sort_values()
    e_map = {old: new for new, old in enumerate(e_means.index)}
    df["E"] = df["E"].map(e_map)
    d_means = df.groupby("D")["acf_6h"].mean().sort_values()
    d_map = {old: new for new, old in enumerate(d_means.index)}
    df["D"] = df["D"].map(d_map)
    df["state"] = df["E"] * n_D + df["D"]
    return df


def estimate_transition(df, col="state", n=n_states):
    T_counts = np.zeros((n, n))
    vals = df[col].values
    idx = df.index.values
    for i in range(len(idx) - 1):
        if idx[i + 1] - idx[i] == 1:
            T_counts[vals[i], vals[i + 1]] += 1
    T = np.zeros_like(T_counts)
    for i in range(n):
        s = T_counts[i].sum()
        if s > 0:
            T[i] = T_counts[i] / s
        else:
            T[i, i] = 1.0
    return T, T_counts


def estimate_cell_params(train):
    cell_mu, cell_sigma, cell_phi = {}, {}, {}
    cell_lambda, cell_mu_jump = {}, {}

    for e in range(n_E):
        for d in range(n_D):
            mask = (train["E"] == e) & (train["D"] == d)
            sub = train[mask]
            n = len(sub)
            if n >= 5:
                lmp = sub["lmp"].values
                mu = float(np.nanmean(lmp))
                sigma = float(np.nanstd(lmp))
                idx_s = sub.index.values
                consec = np.where(np.diff(idx_s) == 1)[0]
                if len(consec) > 5 and sigma > 0:
                    x_t = lmp[consec] - mu
                    x_t1 = lmp[consec + 1] - mu
                    denom = (x_t ** 2).sum()
                    phi = float((x_t * x_t1).sum() / max(denom, 1e-12))
                    phi = np.clip(phi, -0.99, 0.999)
                    resid = x_t1 - phi * x_t
                    threshold = 2.0 * sigma
                    jumps = np.abs(resid) > threshold
                    lam = float(jumps.mean())
                    mu_j = float(np.abs(resid[jumps]).mean()) if jumps.sum() > 0 else 0.0
                else:
                    phi, lam, mu_j = 0.5, 0.0, 0.0
            else:
                e_mask = train["E"] == e
                lmp_e = train[e_mask]["lmp"].values
                mu = float(np.nanmean(lmp_e)) if len(lmp_e) > 0 else 50.0
                sigma = float(np.nanstd(lmp_e)) if len(lmp_e) > 0 else 20.0
                phi, lam, mu_j = 0.5, 0.0, 0.0

            s = e * n_D + d
            cell_mu[s] = mu
            cell_sigma[s] = max(sigma, 1.0)
            cell_phi[s] = phi
            cell_lambda[s] = lam
            cell_mu_jump[s] = mu_j

    return cell_mu, cell_sigma, cell_phi, cell_lambda, cell_mu_jump


def simulate(T, cell_mu, cell_sigma, cell_phi, cell_lambda, cell_mu_jump,
             s0, p0, n_test, n_sim, seed):
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_sim, n_test))
    state_paths = np.zeros((n_sim, n_test), dtype=int)

    for path in range(n_sim):
        s_prev = s0
        p_prev = p0
        for t in range(n_test):
            probs = T[s_prev]
            if probs.sum() > 0:
                s_curr = rng.choice(n_states, p=probs)
            else:
                s_curr = s_prev
            mu = cell_mu[s_curr]
            sigma = cell_sigma[s_curr]
            phi = cell_phi[s_curr]
            lam = cell_lambda[s_curr]
            mu_j = cell_mu_jump[s_curr]

            eps = rng.standard_normal()
            p_new = mu + phi * (p_prev - mu) + sigma * eps
            if lam > 0 and rng.random() < lam:
                p_new += rng.exponential(max(mu_j, 1.0)) * rng.choice([-1, 1])
            p_new = max(p_new, 0.0)

            paths[path, t] = p_new
            state_paths[path, t] = s_curr
            s_prev = s_curr
            p_prev = p_new

    return paths, state_paths


def metrics(test_lmp, sim_paths, rng):
    p5 = np.percentile(sim_paths, 5, axis=0)
    p95 = np.percentile(sim_paths, 95, axis=0)
    coverage = ((test_lmp >= p5) & (test_lmp <= p95)).mean()

    sim_flat = sim_paths.flatten()
    sample = rng.choice(sim_flat, size=len(test_lmp), replace=False)
    ks_stat, ks_pval = sp_stats.ks_2samp(test_lmp, sample)
    wass = sp_stats.wasserstein_distance(test_lmp, sample)

    return {
        "coverage_90": round(coverage, 4),
        "wasserstein": round(wass, 2),
        "ks_stat": round(ks_stat, 4),
        "ks_pval": ks_pval,
        "sim_mean": round(sim_paths.mean(), 1),
        "sim_std": round(sim_paths.std(), 1),
        "actual_mean": round(test_lmp.mean(), 1),
        "actual_std": round(test_lmp.std(), 1),
    }, p5, p95


def make_figure(test_lmp, sim_joint, sim_kron, p5_j, p95_j, p5_k, p95_k):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Price distributions
    ax = axes[0, 0]
    ax.hist(test_lmp, bins=80, density=True, alpha=0.7, color="black", label="Actual 2025")
    ax.hist(sim_joint.flatten(), bins=80, density=True, alpha=0.4, color="steelblue", label="Joint model")
    ax.set_xlabel("LMP ($/MWh)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Price distributions")
    ax.legend()
    ax.set_xlim(0, 300)

    # (b) Joint vs Kron coverage bands
    ax = axes[0, 1]
    x = np.arange(len(test_lmp))
    ax.fill_between(x, p5_j, p95_j, alpha=0.3, color="steelblue", label="Joint 90% band")
    ax.fill_between(x, p5_k, p95_k, alpha=0.2, color="orange", label="Kron 90% band")
    ax.plot(x, test_lmp, "k-", lw=0.5, alpha=0.8, label="Actual")
    ax.set_xlabel("Window index (2025)")
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title("(b) 90% prediction bands")
    ax.legend(fontsize=8)

    # (c) QQ plot
    ax = axes[1, 0]
    sample_j = np.random.default_rng(42).choice(sim_joint.flatten(), size=len(test_lmp), replace=False)
    qq_actual = np.sort(test_lmp)
    qq_sim = np.sort(sample_j)
    ax.scatter(qq_actual, qq_sim, s=2, alpha=0.5)
    lim = max(qq_actual.max(), qq_sim.max())
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("Actual quantiles")
    ax.set_ylabel("Simulated quantiles")
    ax.set_title("(c) QQ plot (joint model)")

    # (d) ACF comparison
    ax = axes[1, 1]
    from statsmodels.tsa.stattools import acf
    max_lag = 50
    acf_actual = acf(test_lmp, nlags=max_lag, fft=True)
    acf_sims = []
    for i in range(min(200, sim_joint.shape[0])):
        acf_sims.append(acf(sim_joint[i], nlags=max_lag, fft=True))
    acf_sims = np.array(acf_sims)
    acf_mean = acf_sims.mean(axis=0)
    acf_lo = np.percentile(acf_sims, 5, axis=0)
    acf_hi = np.percentile(acf_sims, 95, axis=0)
    lags = np.arange(max_lag + 1)
    ax.fill_between(lags, acf_lo, acf_hi, alpha=0.3, color="steelblue", label="Sim 90% band")
    ax.plot(lags, acf_mean, "b-", label="Sim mean")
    ax.plot(lags, acf_actual, "k-", lw=2, label="Actual")
    ax.set_xlabel("Lag (windows x 6h)")
    ax.set_ylabel("ACF")
    ax.set_title("(d) Autocorrelation")
    ax.legend(fontsize=8)

    plt.suptitle("Regime-Switching Model: Out-of-Sample Validation (2025)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_joint_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_joint_validation.png")


def main():
    print("Loading data...")
    df = load_and_prepare()

    train = df[df["datetime"] <= TRAIN_END].copy()
    test = df[df["datetime"] > TRAIN_END].copy()
    print(f"Train: {len(train)} windows, Test: {len(test)} windows")

    # Estimate joint transition matrix
    print("Estimating joint 72x72 transition matrix...")
    T_joint, T_counts = estimate_transition(train)

    # Estimate marginal matrices for Kronecker comparison
    T_E = np.zeros((n_E, n_E))
    T_D = np.zeros((n_D, n_D))
    E_vals = train["E"].values
    D_vals = train["D"].values
    idx_arr = train.index.values
    for i in range(len(idx_arr) - 1):
        if idx_arr[i + 1] - idx_arr[i] == 1:
            T_E[E_vals[i], E_vals[i + 1]] += 1
            T_D[D_vals[i], D_vals[i + 1]] += 1
    for i in range(n_E):
        s = T_E[i].sum()
        if s > 0: T_E[i] /= s
    for i in range(n_D):
        s = T_D[i].sum()
        if s > 0: T_D[i] /= s
    T_kron = np.kron(T_E, T_D)

    # Independence test
    diff = T_joint - T_kron
    frob = np.linalg.norm(diff, "fro")
    frob_rel = frob / np.linalg.norm(T_joint, "fro")
    print(f"T_joint vs kron(T_E,T_D): Frobenius diff = {frob:.4f} (relative {frob_rel*100:.1f}%)")

    # Estimate per-cell parameters
    print("Estimating per-cell AR(1)+jump parameters...")
    cell_mu, cell_sigma, cell_phi, cell_lambda, cell_mu_jump = estimate_cell_params(train)

    # Initial conditions
    e0 = int(train["E"].iloc[-1])
    d0 = int(train["D"].iloc[-1])
    s0 = e0 * n_D + d0
    p0 = float(train["lmp"].iloc[-1])
    test_lmp = test["lmp"].values

    # Simulate joint model
    print(f"Simulating {N_SIM} paths (joint model)...")
    sim_joint, _ = simulate(T_joint, cell_mu, cell_sigma, cell_phi,
                            cell_lambda, cell_mu_jump, s0, p0, len(test), N_SIM, SEED)

    # Simulate Kronecker model
    print(f"Simulating {N_SIM} paths (Kronecker model)...")
    sim_kron, _ = simulate(T_kron, cell_mu, cell_sigma, cell_phi,
                           cell_lambda, cell_mu_jump, s0, p0, len(test), N_SIM, SEED + 1)

    # Metrics
    rng = np.random.default_rng(SEED)
    m_joint, p5_j, p95_j = metrics(test_lmp, sim_joint, rng)
    m_kron, p5_k, p95_k = metrics(test_lmp, sim_kron, rng)

    print(f"\n{'='*60}")
    print(f"OUT-OF-SAMPLE VALIDATION (2025)")
    print(f"{'='*60}")
    print(f"Actual:  mean={m_joint['actual_mean']}, std={m_joint['actual_std']}")
    print(f"")
    print(f"Joint model (72x72 T):")
    print(f"  sim mean={m_joint['sim_mean']}, std={m_joint['sim_std']}")
    print(f"  90% coverage: {m_joint['coverage_90']*100:.1f}%")
    print(f"  Wasserstein:  {m_joint['wasserstein']} $/MWh")
    print(f"  KS stat:      {m_joint['ks_stat']}, p={m_joint['ks_pval']:.2e}")
    print(f"")
    print(f"Kronecker model (T_E x T_D):")
    print(f"  sim mean={m_kron['sim_mean']}, std={m_kron['sim_std']}")
    print(f"  90% coverage: {m_kron['coverage_90']*100:.1f}%")
    print(f"  Wasserstein:  {m_kron['wasserstein']} $/MWh")
    print(f"  KS stat:      {m_kron['ks_stat']}, p={m_kron['ks_pval']:.2e}")

    # Save comparison
    comp = pd.DataFrame({
        "model": ["joint_72x72", "kron_marginal"],
        "coverage_90": [m_joint["coverage_90"], m_kron["coverage_90"]],
        "wasserstein": [m_joint["wasserstein"], m_kron["wasserstein"]],
        "sim_mean": [m_joint["sim_mean"], m_kron["sim_mean"]],
        "sim_std": [m_joint["sim_std"], m_kron["sim_std"]],
        "actual_mean": [m_joint["actual_mean"], m_kron["actual_mean"]],
        "actual_std": [m_joint["actual_std"], m_kron["actual_std"]],
    })
    comp.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\nSaved model_comparison.csv")

    # Save joint transition matrix
    labels = [f"E{e}D{d}" for e in range(n_E) for d in range(n_D)]
    pd.DataFrame(T_joint, index=labels, columns=labels).to_csv(OUT_DIR / "transition_joint.csv")
    print("Saved transition_joint.csv")

    # Make figure
    print("Generating validation figure...")
    make_figure(test_lmp, sim_joint, sim_kron, p5_j, p95_j, p5_k, p95_k)

    print("\nDone.")


if __name__ == "__main__":
    main()
