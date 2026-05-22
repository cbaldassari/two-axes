"""
report_model_comparison.py
==========================
Report completo: modello a 1 asse vs modello a 2 assi.
Genera un PDF con:
  1. Momenti di Dp empirici vs simulati
  2. Coverage 90%
  3. Distribuzione empirica vs simulata (istogrammi)
  4. Traiettorie simulate vs prezzo reale
  5. QQ plot
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
n_E, n_D = 9, 8
N_SIM = 500
N_SHOW = 5
SEED = 42


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
    return df


def estimate_T(labels, ns):
    T = np.zeros((ns, ns))
    for i in range(len(labels) - 1):
        T[labels[i], labels[i + 1]] += 1
    for i in range(ns):
        s = T[i].sum()
        if s > 0: T[i] /= s
        else: T[i, i] = 1.0
    return T


def est_phi(sub, m):
    lmp = sub["lmp"].values
    idx_s = sub.index.values
    consec = np.where(np.diff(idx_s) == 1)[0]
    if len(consec) > 5:
        x_t = lmp[consec] - m
        x_t1 = lmp[consec + 1] - m
        return float(np.clip((x_t * x_t1).sum() / max((x_t ** 2).sum(), 1e-12), -0.99, 0.999))
    return 0.5


def main():
    df = load().reset_index(drop=True)
    n_steps = len(df)
    actual = df["lmp"].values
    p0 = float(actual[0])

    # Parameters
    mu_E = {e: float(df[df["E"] == e]["lmp"].mean()) for e in range(n_E)}
    sigma_E = {e: float(df[df["E"] == e]["lmp"].std()) for e in range(n_E)}
    phi_E = {e: est_phi(df[df["E"] == e], mu_E[e]) for e in range(n_E)}
    phi_ED = {}
    for e in range(n_E):
        for d in range(n_D):
            sub = df[(df["E"] == e) & (df["D"] == d)]
            if len(sub) >= 10:
                phi_ED[(e, d)] = est_phi(sub, mu_E[e])
            else:
                phi_ED[(e, d)] = phi_E[e]

    T_E = estimate_T(df["E"].values, n_E)
    df["state"] = df["E"] * n_D + df["D"]
    T_joint = estimate_T(df["state"].values, n_E * n_D)

    rng = np.random.default_rng(SEED)

    # Simulate Model 1: phi(E)
    print("Simulating Model 1 (1 axis)...")
    sim_1 = np.zeros((N_SIM, n_steps))
    for path in range(N_SIM):
        s = int(df["E"].iloc[0]); p = p0
        for t in range(n_steps):
            s = rng.choice(n_E, p=T_E[s])
            m = mu_E[s]
            p = max(m + phi_E[s] * (p - m) + sigma_E[s] * rng.standard_normal(), 0.0)
            sim_1[path, t] = p

    # Simulate Model 2: phi(E,D)
    print("Simulating Model 2 (2 axes)...")
    sim_2 = np.zeros((N_SIM, n_steps))
    for path in range(N_SIM):
        s = int(df["state"].iloc[0]); p = p0
        for t in range(n_steps):
            s = rng.choice(n_E * n_D, p=T_joint[s])
            e = s // n_D; d = s % n_D; m = mu_E[e]
            p = max(m + phi_ED[(e, d)] * (p - m) + sigma_E[e] * rng.standard_normal(), 0.0)
            sim_2[path, t] = p

    # Delta p
    dp_emp = np.diff(actual)
    dp_sim1 = np.diff(sim_1, axis=1).flatten()
    dp_sim2 = np.diff(sim_2, axis=1).flatten()
    s1 = rng.choice(dp_sim1, len(dp_emp), replace=False)
    s2 = rng.choice(dp_sim2, len(dp_emp), replace=False)

    # ═══════════════════════════════════════════════
    # PRINT REPORT
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("REPORT: MODELLO 1 ASSE vs 2 ASSI")
    print("=" * 70)

    # Moments of Dp
    print("\n--- Momenti di Dp ---")
    print(f"{'':>15} {'Media':>8} {'Std':>8} {'Skew':>8} {'Kurt(P)':>8} {'Wass':>8}")
    print("-" * 55)
    print(f"{'Empirico':>15} {np.mean(dp_emp):>8.2f} {np.std(dp_emp):>8.1f} "
          f"{sp_stats.skew(dp_emp):>8.2f} {sp_stats.kurtosis(dp_emp, fisher=False):>8.2f}")
    for name, s in [("phi(E)", s1), ("phi(E,D)", s2)]:
        w = sp_stats.wasserstein_distance(dp_emp, s)
        print(f"{name:>15} {np.mean(s):>8.2f} {np.std(s):>8.1f} "
              f"{sp_stats.skew(s):>8.2f} {sp_stats.kurtosis(s, fisher=False):>8.2f} {w:>8.2f}")

    # Moments of price levels
    print("\n--- Momenti del livello di prezzo ---")
    print(f"{'':>15} {'Media':>8} {'Std':>8} {'Skew':>8} {'Kurt(P)':>8}")
    print("-" * 45)
    print(f"{'Empirico':>15} {np.mean(actual):>8.1f} {np.std(actual):>8.1f} "
          f"{sp_stats.skew(actual):>8.2f} {sp_stats.kurtosis(actual, fisher=False):>8.2f}")
    for name, sim in [("phi(E)", sim_1), ("phi(E,D)", sim_2)]:
        pool = rng.choice(sim.flatten(), len(actual), replace=False)
        print(f"{name:>15} {np.mean(pool):>8.1f} {np.std(pool):>8.1f} "
              f"{sp_stats.skew(pool):>8.2f} {sp_stats.kurtosis(pool, fisher=False):>8.2f}")

    # Coverage
    print("\n--- Coverage 90% ---")
    for name, sim in [("phi(E)", sim_1), ("phi(E,D)", sim_2)]:
        p5 = np.percentile(sim, 5, axis=0)
        p95 = np.percentile(sim, 95, axis=0)
        cov = ((actual >= p5) & (actual <= p95)).mean()
        print(f"  {name}: {cov*100:.1f}%")

    # Coverage by D group
    D_vals = df["D"].values
    print("\n--- Coverage 90% per gruppo D ---")
    print(f"{'':>15} {'Fast':>8} {'Moderate':>10} {'Persistent':>12}")
    for name, sim in [("phi(E)", sim_1), ("phi(E,D)", sim_2)]:
        p5 = np.percentile(sim, 5, axis=0)
        p95 = np.percentile(sim, 95, axis=0)
        covs = []
        for dvals in [[0, 1], [2, 3, 4], [5, 6, 7]]:
            mask = np.isin(D_vals, dvals)
            c = ((actual[mask] >= p5[mask]) & (actual[mask] <= p95[mask])).mean()
            covs.append(f"{c*100:.1f}%")
        print(f"  {name:>13} {covs[0]:>8} {covs[1]:>10} {covs[2]:>12}")

    # ═══════════════════════════════════════════════
    # FIGURE (6 panels)
    # ═══════════════════════════════════════════════
    print("\nGenerating figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(n_steps)
    path_colors = plt.cm.Set2(np.linspace(0, 1, N_SHOW))

    # Row 1: Trajectories
    for col, (sim, title, band_color) in enumerate([
        (sim_1, r"Modello 1 asse: $\varphi(E)$", "#e74c3c"),
        (sim_2, r"Modello 2 assi: $\varphi(E,D)$", "#2ecc71"),
    ]):
        ax = axes[0, col]
        # 90% band
        p5 = np.percentile(sim, 5, axis=0)
        p95 = np.percentile(sim, 95, axis=0)
        ax.fill_between(x, p5, p95, alpha=0.12, color=band_color, label="Banda 90%")
        # Sample paths
        idx_show = rng.choice(N_SIM, N_SHOW, replace=False)
        for j, i in enumerate(idx_show):
            ax.plot(x, sim[i], color=path_colors[j], alpha=0.7, lw=0.5,
                    label=f"Sim {j+1}" if j < 3 else None)
        # Actual
        ax.plot(x, actual, "k-", lw=0.8, alpha=0.9, label="Reale")
        # Median
        ax.plot(x, np.median(sim, axis=0), color=band_color, lw=1.2, ls="--",
                label="Mediana sim")
        ax.set_ylabel("$/MWh")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.set_ylim(0, 350)
        ax.grid(alpha=0.2)

    # Row 2: Distributions of Dp
    bins_dp = np.linspace(-150, 150, 80)
    for col, (s, title, color) in enumerate([
        (s1, r"$\Delta p$ simulato: $\varphi(E)$", "#e74c3c"),
        (s2, r"$\Delta p$ simulato: $\varphi(E,D)$", "#2ecc71"),
    ]):
        ax = axes[1, col]
        ax.hist(dp_emp, bins=bins_dp, density=True, alpha=0.5, color="black", label="Empirico")
        ax.hist(s, bins=bins_dp, density=True, alpha=0.5, color=color, label="Simulato")
        ax.set_xlabel("$\Delta p$ ($/MWh)")
        ax.set_ylabel("Densita")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(-150, 150)
        ax.grid(alpha=0.2)

    plt.suptitle("Confronto modelli: 1 asse vs 2 assi\n"
                 "Traiettorie e distribuzioni",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = OUT_DIR / "report_model_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
