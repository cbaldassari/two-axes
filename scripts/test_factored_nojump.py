"""
test_factored_nojump.py
=======================
Compare models by matching the 4 moments (mean, std, skewness, kurtosis)
of simulated vs actual 2025 prices, plus Wasserstein distance and KS test.

Models: AR(1), 3-state classical, Economic-only (9), Factored (E+D).
All without jump. Validation overall and conditional on D group.
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
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


def load():
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

    train_mask = df["datetime"] <= TRAIN_END
    p33 = df.loc[train_mask, "lmp"].quantile(0.33)
    p67 = df.loc[train_mask, "lmp"].quantile(0.67)
    df["C3"] = np.where(df["lmp"] <= p33, 0, np.where(df["lmp"] <= p67, 1, 2))
    return df


def estimate_T(labels, n):
    T = np.zeros((n, n))
    for i in range(len(labels) - 1):
        T[labels[i], labels[i + 1]] += 1
    for i in range(n):
        s = T[i].sum()
        if s > 0: T[i] /= s
        else: T[i, i] = 1.0
    return T


def estimate_ar1(train, col, n):
    mu, phi, sigma = {}, {}, {}
    for s in range(n):
        sub = train[train[col] == s]
        lmp = sub["lmp"].values
        if len(lmp) >= 5:
            m = float(np.nanmean(lmp))
            idx_s = sub.index.values
            consec = np.where(np.diff(idx_s) == 1)[0]
            if len(consec) > 5:
                x_t = lmp[consec] - m
                x_t1 = lmp[consec + 1] - m
                p = float(np.clip((x_t * x_t1).sum() / max((x_t**2).sum(), 1e-12), -0.99, 0.999))
                sig = float(np.std(x_t1 - p * x_t))
            else:
                p, sig = 0.5, float(np.nanstd(lmp))
        else:
            m = float(train["lmp"].mean())
            p, sig = 0.5, float(train["lmp"].std())
        mu[s] = m
        phi[s] = p
        sigma[s] = max(sig, 1.0)
    return mu, phi, sigma


def simulate(T, mu, phi, sigma, s0, p0, n_test, n_sim, seed):
    rng = np.random.default_rng(seed)
    n_st = T.shape[0]
    paths = np.zeros((n_sim, n_test))
    for path in range(n_sim):
        s_prev, p_prev = s0, p0
        for t in range(n_test):
            s_curr = rng.choice(n_st, p=T[s_prev])
            m = mu[s_curr]
            paths[path, t] = max(m + phi[s_curr] * (p_prev - m) + sigma[s_curr] * rng.standard_normal(), 0.0)
            s_prev, p_prev = s_curr, paths[path, t]
    return paths


def simulate_factored(T_joint, mu_E, sigma_E, phi_D, s0, p0, n_test, n_sim, seed):
    rng = np.random.default_rng(seed)
    n_st = T_joint.shape[0]
    paths = np.zeros((n_sim, n_test))
    for path in range(n_sim):
        s_prev, p_prev = s0, p0
        for t in range(n_test):
            s_curr = rng.choice(n_st, p=T_joint[s_prev])
            e = s_curr // n_D
            d = s_curr % n_D
            m = mu_E[e]
            paths[path, t] = max(m + phi_D[d] * (p_prev - m) + sigma_E[e] * rng.standard_normal(), 0.0)
            s_prev, p_prev = s_curr, paths[path, t]
    return paths


def moments(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "skew": float(sp_stats.skew(x)),
        "kurt": float(sp_stats.kurtosis(x)),
    }


def compare(name, sim, actual, rng):
    """Compare simulated pool vs actual: moments + distributional tests."""
    # Pool all simulated values and sample same size as actual
    pool = sim.flatten()
    sample = rng.choice(pool, size=len(actual), replace=False)

    m_act = moments(actual)
    m_sim = moments(sample)
    wass = sp_stats.wasserstein_distance(actual, sample)
    ks_stat, ks_p = sp_stats.ks_2samp(actual, sample)

    return {
        "model": name,
        "act_mean": round(m_act["mean"], 1),
        "sim_mean": round(m_sim["mean"], 1),
        "act_std": round(m_act["std"], 1),
        "sim_std": round(m_sim["std"], 1),
        "act_skew": round(m_act["skew"], 2),
        "sim_skew": round(m_sim["skew"], 2),
        "act_kurt": round(m_act["kurt"], 2),
        "sim_kurt": round(m_sim["kurt"], 2),
        "wasserstein": round(wass, 2),
        "ks_stat": round(ks_stat, 4),
        "ks_p": ks_p,
    }


def main():
    df = load().reset_index(drop=True)
    train = df[df["datetime"] <= TRAIN_END].copy().reset_index(drop=True)
    test = df[df["datetime"] > TRAIN_END].copy().reset_index(drop=True)
    test_lmp = test["lmp"].values
    test_D = test["D"].values
    p0 = float(train["lmp"].iloc[-1])
    n_test = len(test)

    rng = np.random.default_rng(SEED)

    # Build models
    sims = {}

    # 1. AR(1) no regimes
    train["_s0"] = 0
    mu1, phi1, sig1 = estimate_ar1(train, "_s0", 1)
    sims["AR(1)"] = simulate(np.array([[1.0]]), mu1, phi1, sig1, 0, p0, n_test, N_SIM, SEED)

    # 2. 3-state classical
    T_c3 = estimate_T(train["C3"].values, 3)
    mu3, phi3, sig3 = estimate_ar1(train, "C3", 3)
    sims["3-state"] = simulate(T_c3, mu3, phi3, sig3, int(train["C3"].iloc[-1]), p0, n_test, N_SIM, SEED)

    # 3. Economic-only (9)
    T_e = estimate_T(train["E"].values, n_E)
    mu_e, phi_e, sig_e = estimate_ar1(train, "E", n_E)
    sims["Econ-9"] = simulate(T_e, mu_e, phi_e, sig_e, int(train["E"].iloc[-1]), p0, n_test, N_SIM, SEED)

    # 4. Factored (E→level, D→phi)
    train["state"] = train["E"] * n_D + train["D"]
    T_joint = estimate_T(train["state"].values, n_E * n_D)
    s0_j = int(train["state"].iloc[-1])
    phi_D = {}
    for d in range(n_D):
        sub = train[train["D"] == d]
        lmp = sub["lmp"].values
        m_all = float(np.nanmean(lmp))
        idx_s = sub.index.values
        consec = np.where(np.diff(idx_s) == 1)[0]
        if len(consec) > 5:
            x_t = lmp[consec] - m_all
            x_t1 = lmp[consec + 1] - m_all
            p = float(np.clip((x_t * x_t1).sum() / max((x_t**2).sum(), 1e-12), -0.99, 0.999))
        else:
            p = 0.5
        phi_D[d] = p
    sims["Factored"] = simulate_factored(T_joint, mu_e, sig_e, phi_D, s0_j, p0, n_test, N_SIM, SEED)

    # ── Overall comparison ──
    print("=" * 90)
    print("OVERALL: simulated vs actual 2025 (4 moments + distributional tests)")
    print("=" * 90)

    rows = []
    for name, sim in sims.items():
        r = compare(name, sim, test_lmp, rng)
        rows.append(r)

    print(f"\n{'':>12} {'mean':>12} {'std':>12} {'skew':>12} {'kurt':>12} {'Wass':>8} {'KS':>8}")
    print(f"{'Actual':>12} {rows[0]['act_mean']:>12.1f} {rows[0]['act_std']:>12.1f} "
          f"{rows[0]['act_skew']:>12.2f} {rows[0]['act_kurt']:>12.2f}")
    print("-" * 90)
    for r in rows:
        print(f"{r['model']:>12} {r['sim_mean']:>12.1f} {r['sim_std']:>12.1f} "
              f"{r['sim_skew']:>12.2f} {r['sim_kurt']:>12.2f} "
              f"{r['wasserstein']:>8.2f} {r['ks_stat']:>8.4f}")

    # ── Conditional on D group ──
    d_groups = [("Fast (D0-D1)", [0, 1]), ("Moderate (D2-D4)", [2, 3, 4]), ("Persistent (D5-D7)", [5, 6, 7])]

    for dg_name, dvals in d_groups:
        mask = np.isin(test_D, dvals)
        actual_sub = test_lmp[mask]
        print(f"\n{'=' * 90}")
        print(f"CONDITIONAL: {dg_name} (n={mask.sum()})")
        print(f"{'=' * 90}")

        rows_c = []
        for name, sim in sims.items():
            sim_sub = sim[:, mask]
            r = compare(name, sim_sub, actual_sub, rng)
            rows_c.append(r)

        print(f"\n{'':>12} {'mean':>12} {'std':>12} {'skew':>12} {'kurt':>12} {'Wass':>8}")
        print(f"{'Actual':>12} {rows_c[0]['act_mean']:>12.1f} {rows_c[0]['act_std']:>12.1f} "
              f"{rows_c[0]['act_skew']:>12.2f} {rows_c[0]['act_kurt']:>12.2f}")
        print("-" * 80)
        for r in rows_c:
            print(f"{r['model']:>12} {r['sim_mean']:>12.1f} {r['sim_std']:>12.1f} "
                  f"{r['sim_skew']:>12.2f} {r['sim_kurt']:>12.2f} "
                  f"{r['wasserstein']:>8.2f}")

    # ── Figure: QQ plots ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    model_names = list(sims.keys())
    colors = ["#aaaaaa", "#aaaaaa", "#6baed6", "#e67e22"]

    for ax, name, color in zip(axes.flat, model_names, colors):
        sim = sims[name]
        sample = rng.choice(sim.flatten(), size=len(test_lmp), replace=False)
        qq_act = np.sort(test_lmp)
        qq_sim = np.sort(sample)
        ax.scatter(qq_act, qq_sim, s=3, alpha=0.5, color=color)
        lim = max(qq_act.max(), qq_sim.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("Actual quantiles ($/MWh)")
        ax.set_ylabel("Simulated quantiles ($/MWh)")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)

        # Add moments text
        m_s = moments(sample)
        m_a = moments(test_lmp)
        ax.text(0.05, 0.95,
                f"mean: {m_a['mean']:.0f} vs {m_s['mean']:.0f}\n"
                f"std:  {m_a['std']:.0f} vs {m_s['std']:.0f}\n"
                f"skew: {m_a['skew']:.1f} vs {m_s['skew']:.1f}\n"
                f"kurt: {m_a['kurt']:.1f} vs {m_s['kurt']:.1f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle("QQ plots: simulated vs actual 2025 prices (no jump)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_qq_moments.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved fig_qq_moments.png")

    # Save CSV
    pd.DataFrame(rows).to_csv(OUT_DIR / "moments_comparison.csv", index=False)
    print("Saved moments_comparison.csv")


if __name__ == "__main__":
    main()
