"""
fig_montecarlo_paths.py
=======================
Visualize Monte Carlo simulated paths vs actual 2025 prices
for each model.
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
OUT_DIR = RESULTS_DIR / "dual_axis"
TRAIN_END = "2024-12-31"
N_SIM = 1000
SEED = 42
n_E, n_D = 9, 8
N_SHOW = 50  # paths to draw


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

    e_means = df.groupby("E")["lmp"].mean().sort_values()
    e_map = {old: new for new, old in enumerate(e_means.index)}
    df["E"] = df["E"].map(e_map)
    d_means = df.groupby("D")["acf_6h"].mean().sort_values()
    d_map = {old: new for new, old in enumerate(d_means.index)}
    df["D"] = df["D"].map(d_map)

    p33 = df.loc[df["datetime"] <= TRAIN_END, "lmp"].quantile(0.33)
    p67 = df.loc[df["datetime"] <= TRAIN_END, "lmp"].quantile(0.67)
    df["C3"] = np.where(df["lmp"] <= p33, 0, np.where(df["lmp"] <= p67, 1, 2))
    return df


def estimate_transition_matrix(labels, n_states):
    T = np.zeros((n_states, n_states))
    for i in range(len(labels) - 1):
        T[labels[i], labels[i + 1]] += 1
    for i in range(n_states):
        s = T[i].sum()
        if s > 0:
            T[i] /= s
        else:
            T[i, i] = 1.0
    return T


def estimate_ar1j(train, col, n_states):
    mu, phi, sigma, lam, mu_j = {}, {}, {}, {}, {}
    for s in range(n_states):
        sub = train[train[col] == s]
        lmp = sub["lmp"].values
        if len(lmp) >= 5:
            m = float(np.nanmean(lmp))
            sig = float(np.nanstd(lmp))
            idx_s = sub.index.values
            consec = np.where(np.diff(idx_s) == 1)[0]
            if len(consec) > 5 and sig > 0:
                x_t = lmp[consec] - m
                x_t1 = lmp[consec + 1] - m
                p = float(np.clip((x_t * x_t1).sum() / max((x_t**2).sum(), 1e-12), -0.99, 0.999))
                resid = x_t1 - p * x_t
                jumps = np.abs(resid) > 2.0 * sig
                la = float(jumps.mean())
                mj = float(np.abs(resid[jumps]).mean()) if jumps.sum() > 0 else 0.0
            else:
                p, la, mj = 0.5, 0.0, 0.0
        else:
            m = float(train["lmp"].mean())
            sig = float(train["lmp"].std())
            p, la, mj = 0.5, 0.0, 0.0
        mu[s] = m
        phi[s] = p
        sigma[s] = max(sig, 1.0)
        lam[s] = la
        mu_j[s] = mj
    return mu, phi, sigma, lam, mu_j


def simulate(T, mu, phi, sigma, lam, mu_j, s0, p0, n_test, n_sim, seed):
    rng = np.random.default_rng(seed)
    n_st = T.shape[0]
    paths = np.zeros((n_sim, n_test))
    for path in range(n_sim):
        s_prev, p_prev = s0, p0
        for t in range(n_test):
            s_curr = rng.choice(n_st, p=T[s_prev])
            m = mu[s_curr]
            p_new = m + phi[s_curr] * (p_prev - m) + sigma[s_curr] * rng.standard_normal()
            if lam[s_curr] > 0 and rng.random() < lam[s_curr]:
                p_new += rng.exponential(max(mu_j[s_curr], 1.0)) * rng.choice([-1, 1])
            paths[path, t] = max(p_new, 0.0)
            s_prev, p_prev = s_curr, p_new
    return paths


def simulate_factored(T_joint, mu_E, sigma_E, lam_E, muj_E, phi_D,
                      s0, p0, n_test, n_sim, seed):
    rng = np.random.default_rng(seed)
    n_st = T_joint.shape[0]
    paths = np.zeros((n_sim, n_test))
    for path in range(n_sim):
        s_prev, p_prev = s0, p0
        for t in range(n_test):
            s_curr = rng.choice(n_st, p=T_joint[s_prev])
            e_curr = s_curr // n_D
            d_curr = s_curr % n_D
            m = mu_E[e_curr]
            p_new = m + phi_D[d_curr] * (p_prev - m) + sigma_E[e_curr] * rng.standard_normal()
            if lam_E[e_curr] > 0 and rng.random() < lam_E[e_curr]:
                p_new += rng.exponential(max(muj_E[e_curr], 1.0)) * rng.choice([-1, 1])
            paths[path, t] = max(p_new, 0.0)
            s_prev, p_prev = s_curr, p_new
    return paths


def main():
    df = load_and_prepare()
    train = df[df["datetime"] <= TRAIN_END].copy().reset_index(drop=True)
    test = df[df["datetime"] > TRAIN_END].copy().reset_index(drop=True)
    test_lmp = test["lmp"].values
    test_dt = pd.to_datetime(test["datetime"])
    p0 = float(train["lmp"].iloc[-1])
    n_test = len(test)

    # Build all models
    models = {}

    # 1. AR(1)+J
    train["_s0"] = 0
    mu1, phi1, sig1, lam1, muj1 = estimate_ar1j(train, "_s0", 1)
    models["AR(1)+J\nno regimes"] = simulate(np.array([[1.0]]), mu1, phi1, sig1, lam1, muj1,
                                              0, p0, n_test, N_SIM, SEED)

    # 2. 3-state classical
    T_c3 = estimate_transition_matrix(train["C3"].values, 3)
    mu3, phi3, sig3, lam3, muj3 = estimate_ar1j(train, "C3", 3)
    models["3-state\nclassical"] = simulate(T_c3, mu3, phi3, sig3, lam3, muj3,
                                            int(train["C3"].iloc[-1]), p0, n_test, N_SIM, SEED)

    # 3. Economic-only (9)
    T_e = estimate_transition_matrix(train["E"].values, n_E)
    mu_e, phi_e, sig_e, lam_e, muj_e = estimate_ar1j(train, "E", n_E)
    models["Economic\nonly (9)"] = simulate(T_e, mu_e, phi_e, sig_e, lam_e, muj_e,
                                            int(train["E"].iloc[-1]), p0, n_test, N_SIM, SEED)

    # 4. Dual-axis joint (72)
    train["state"] = train["E"] * n_D + train["D"]
    T_joint = estimate_transition_matrix(train["state"].values, n_E * n_D)
    mu_j, phi_j, sig_j, lam_j, muj_j = estimate_ar1j(train, "state", n_E * n_D)
    s0_j = int(train["state"].iloc[-1])
    models["Dual-axis\njoint (72)"] = simulate(T_joint, mu_j, phi_j, sig_j, lam_j, muj_j,
                                                s0_j, p0, n_test, N_SIM, SEED)

    # 5. Factored
    phi_D = {}
    for d in range(n_D):
        sub = train[train["D"] == d]
        lmp = sub["lmp"].values
        if len(lmp) >= 10:
            m_all = float(np.nanmean(lmp))
            idx_s = sub.index.values
            consec = np.where(np.diff(idx_s) == 1)[0]
            if len(consec) > 5:
                x_t = lmp[consec] - m_all
                x_t1 = lmp[consec + 1] - m_all
                p = float(np.clip((x_t * x_t1).sum() / max((x_t**2).sum(), 1e-12), -0.99, 0.999))
            else:
                p = 0.5
        else:
            p = 0.5
        phi_D[d] = p
    models["Dual-axis\nfactored"] = simulate_factored(T_joint, mu_e, sig_e, lam_e, muj_e,
                                                       phi_D, s0_j, p0, n_test, N_SIM, SEED)

    # --- Figure ---
    model_names = list(models.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 3.5 * n_models), sharex=True, sharey=True)

    x = np.arange(n_test)
    rng = np.random.default_rng(SEED)

    for ax, name in zip(axes, model_names):
        sim = models[name]
        # Draw N_SHOW random paths
        idx_show = rng.choice(N_SIM, N_SHOW, replace=False)
        for i in idx_show:
            ax.plot(x, sim[i], color="steelblue", alpha=0.08, lw=0.5)

        # 90% band
        p5 = np.percentile(sim, 5, axis=0)
        p95 = np.percentile(sim, 95, axis=0)
        ax.fill_between(x, p5, p95, alpha=0.15, color="steelblue")

        # Median
        med = np.median(sim, axis=0)
        ax.plot(x, med, color="steelblue", lw=1.5, label="Median sim")

        # Actual
        ax.plot(x, test_lmp, color="black", lw=1.2, label="Actual 2025")

        # Coverage
        cov = ((test_lmp >= p5) & (test_lmp <= p95)).mean()
        ax.set_ylabel("$/MWh", fontsize=10)
        ax.set_title(f"{name.replace(chr(10), ' ')}  —  coverage={cov*100:.1f}%",
                     fontsize=12, fontweight="bold", loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 350)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Window index (2025, 6h steps)", fontsize=11)
    plt.suptitle("Monte Carlo paths (50 of 1000) vs actual 2025 prices",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_montecarlo_paths.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_montecarlo_paths.png")


if __name__ == "__main__":
    main()
