"""
fig_sim_comparison.py
=====================
Simulate price paths from both models (1-axis vs 2-axis)
and compare against actual prices.
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
n_E, n_D = 9, 8
N_SIM = 200
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

    # Parameters
    mu_E = {e: float(df[df["E"] == e]["lmp"].mean()) for e in range(n_E)}
    sigma_E = {e: float(df[df["E"] == e]["lmp"].std()) for e in range(n_E)}

    # phi(E)
    phi_E = {e: est_phi(df[df["E"] == e], mu_E[e]) for e in range(n_E)}

    # phi(E,D)
    phi_ED = {}
    for e in range(n_E):
        for d in range(n_D):
            sub = df[(df["E"] == e) & (df["D"] == d)]
            if len(sub) >= 10:
                phi_ED[(e, d)] = est_phi(sub, mu_E[e])
            else:
                phi_ED[(e, d)] = phi_E[e]

    # Transition matrices
    T_E = estimate_T(df["E"].values, n_E)
    df["state"] = df["E"] * n_D + df["D"]
    T_joint = estimate_T(df["state"].values, n_E * n_D)

    # Simulate
    rng = np.random.default_rng(SEED)
    n_steps = len(df)
    p0 = float(df["lmp"].iloc[0])

    # Model 1: phi(E), T_E 9x9
    sim_1ax = np.zeros((N_SIM, n_steps))
    for path in range(N_SIM):
        s = int(df["E"].iloc[0])
        p = p0
        for t in range(n_steps):
            s = rng.choice(n_E, p=T_E[s])
            m = mu_E[s]
            p = max(m + phi_E[s] * (p - m) + sigma_E[s] * rng.standard_normal(), 0.0)
            sim_1ax[path, t] = p

    # Model 2: phi(E,D), T_joint 72x72
    sim_2ax = np.zeros((N_SIM, n_steps))
    for path in range(N_SIM):
        s = int(df["state"].iloc[0])
        p = p0
        for t in range(n_steps):
            s = rng.choice(n_E * n_D, p=T_joint[s])
            e = s // n_D
            d = s % n_D
            m = mu_E[e]
            p = max(m + phi_ED[(e, d)] * (p - m) + sigma_E[e] * rng.standard_normal(), 0.0)
            sim_2ax[path, t] = p

    actual = df["lmp"].values
    x = np.arange(n_steps)

    # Figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    for ax, sim, title, color in [
        (axes[0], sim_1ax, r"Modello a un asse: $\varphi(E)$", "#e74c3c"),
        (axes[1], sim_2ax, r"Modello a due assi: $\varphi(E,D)$", "#2ecc71"),
    ]:
        # Sample paths — few, visible
        idx_show = rng.choice(N_SIM, N_SHOW, replace=False)
        path_colors = plt.cm.tab10(np.linspace(0, 1, N_SHOW))
        for j, i in enumerate(idx_show):
            ax.plot(x, sim[i], color=path_colors[j], alpha=0.5, lw=0.6)

        # 90% band
        p5 = np.percentile(sim, 5, axis=0)
        p95 = np.percentile(sim, 95, axis=0)
        ax.fill_between(x, p5, p95, alpha=0.15, color=color)

        # Median
        med = np.median(sim, axis=0)
        ax.plot(x, med, color=color, lw=1.2, label="Mediana simulata")

        # Actual
        ax.plot(x, actual, "k-", lw=0.7, alpha=0.8, label="Prezzo reale")

        ax.set_ylabel("$/MWh")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(0, 350)
        ax.grid(alpha=0.2)

    axes[1].set_xlabel("Finestra (passo 6h)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_sim_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_sim_comparison.png")


if __name__ == "__main__":
    main()
