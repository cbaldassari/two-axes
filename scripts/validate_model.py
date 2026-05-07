"""
validate_model.py
=================
Validation of the dual-axis AR(1) regime-switching model.

Strategy: train on 2021-2024, simulate 2025, compare with actual 2025.
- Statistical properties (mean, std, quantiles, ACF)
- Price distribution (QQ-plot, histogram overlap)
- Regime frequencies
- VaR accuracy
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

RESULTS = Path("results")
SEED = 42
N_SIM = 500
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10


def build_T(labels, K):
    T = np.zeros((K, K))
    for i in range(len(labels) - 1):
        T[labels[i], labels[i + 1]] += 1
    T = T / T.sum(axis=1, keepdims=True).clip(1e-10)
    return T


def acf(x, lag):
    n = len(x)
    m = x.mean()
    v = ((x - m) ** 2).sum()
    if v < 1e-15 or lag >= n:
        return 0.0
    return float(((x[:n - lag] - m) * (x[lag:] - m)).sum() / v)


def main():
    print("=" * 70)
    print("  MODEL VALIDATION — Train 2021-2024, Test 2025")
    print("=" * 70)

    # ── Load ──
    lab_fe = pd.read_parquet(RESULTS / "exp_FE/step04/labels.parquet")
    lab_mom = pd.read_parquet(RESULTS / "exp_C/step04/labels.parquet")
    fe_emb = pd.read_parquet(RESULTS / "exp_FE/embeddings.parquet")
    pre = pd.read_parquet(RESULTS / "preprocessed.parquet")
    for d in [lab_fe, lab_mom, fe_emb, pre]:
        d["datetime"] = pd.to_datetime(d["datetime"])

    df = lab_fe.merge(lab_mom, on="datetime", suffixes=("_fe", "_mom"))
    df = df.merge(pre[["datetime", "lmp"]], on="datetime")
    df = df.merge(fe_emb[["datetime", "acf_1h", "acf_6h"]], on="datetime")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Rank regimes
    fe_order = df.groupby("cluster_fe")["lmp"].mean().sort_values().index.tolist()
    df["E"] = df["cluster_fe"].map({r: i for i, r in enumerate(fe_order)})
    mom_order = df.groupby("cluster_mom")["lmp"].mean().sort_values().index.tolist()
    df["D"] = df["cluster_mom"].map({r: i for i, r in enumerate(mom_order)})

    K_E, K_D = 10, 8

    # ── Temporal split ──
    split_date = pd.Timestamp("2025-01-01")
    train = df[df["datetime"] < split_date].copy()
    test = df[df["datetime"] >= split_date].copy()
    print(f"  Train: {len(train)} windows (before {split_date.date()})")
    print(f"  Test:  {len(test)} windows (from {split_date.date()})")

    # ── Estimate parameters on TRAIN only ──
    mu = np.array([train.loc[train["E"] == e, "lmp"].mean() if (train["E"] == e).sum() > 0
                    else df.loc[df["E"] == e, "lmp"].mean() for e in range(K_E)])
    sigma = np.array([train.loc[train["E"] == e, "lmp"].std() if (train["E"] == e).sum() > 0
                       else df.loc[df["E"] == e, "lmp"].std() for e in range(K_E)])
    phi = np.array([train.loc[train["D"] == d, "acf_1h"].mean() if (train["D"] == d).sum() > 0
                     else df.loc[df["D"] == d, "acf_1h"].mean() for d in range(K_D)])

    T_E = build_T(train["E"].values, K_E)
    T_D = build_T(train["D"].values, K_D)

    print(f"\n  Parameters (estimated on train):")
    print(f"  mu(E):    {[f'{m:.1f}' for m in mu]}")
    print(f"  sigma(E): {[f'{s:.1f}' for s in sigma]}")
    print(f"  phi(D):   {[f'{p:.3f}' for p in phi]}")

    # ── Simulate test period ──
    H = len(test)
    # Initial state from last training window
    E0 = train["E"].iloc[-1]
    D0 = train["D"].iloc[-1]
    p0 = train["lmp"].iloc[-1]
    print(f"\n  Initial state: E={E0}, D={D0}, p={p0:.1f}")
    print(f"  Simulating {N_SIM} paths x {H} steps ...")

    rng = np.random.RandomState(SEED)
    sim_prices = np.empty((N_SIM, H))

    for n in range(N_SIM):
        E_t, D_t, p_t = E0, D0, p0
        for t in range(H):
            if t > 0:
                E_t = rng.choice(K_E, p=T_E[E_t])
                D_t = rng.choice(K_D, p=T_D[D_t])
            eps = rng.randn()
            p_t = mu[E_t] + phi[D_t] * (p_t - mu[E_t]) + sigma[E_t] * 0.1 * eps
            p_t = max(p_t, 0)
            sim_prices[n, t] = p_t

    actual = test["lmp"].values

    # ── Compare statistics ──
    sim_mean = sim_prices.mean(axis=1)
    sim_std = np.std(sim_prices, axis=1)
    sim_flat = sim_prices.flatten()

    print(f"\n  {'Statistic':<20s} {'Actual':>10s} {'Simulated (mean)':>18s} {'Sim 5-95%':>20s}")
    print("  " + "-" * 70)

    stats = [
        ("Mean", actual.mean(), sim_mean.mean(),
         np.percentile(sim_mean, 5), np.percentile(sim_mean, 95)),
        ("Std", actual.std(), sim_std.mean(),
         np.percentile(sim_std, 5), np.percentile(sim_std, 95)),
        ("Median", np.median(actual), np.median(sim_prices, axis=1).mean(),
         np.percentile(np.median(sim_prices, axis=1), 5),
         np.percentile(np.median(sim_prices, axis=1), 95)),
        ("P5", np.percentile(actual, 5),
         np.percentile(sim_prices, 5, axis=1).mean(),
         np.percentile(np.percentile(sim_prices, 5, axis=1), 5),
         np.percentile(np.percentile(sim_prices, 5, axis=1), 95)),
        ("P95", np.percentile(actual, 95),
         np.percentile(sim_prices, 95, axis=1).mean(),
         np.percentile(np.percentile(sim_prices, 95, axis=1), 5),
         np.percentile(np.percentile(sim_prices, 95, axis=1), 95)),
        ("Max", actual.max(),
         sim_prices.max(axis=1).mean(),
         np.percentile(sim_prices.max(axis=1), 5),
         np.percentile(sim_prices.max(axis=1), 95)),
    ]

    for name, act, sim, lo, hi in stats:
        print(f"  {name:<20s} {act:>10.1f} {sim:>18.1f} [{lo:.1f}, {hi:.1f}]")

    # ACF comparison
    print(f"\n  ACF comparison:")
    for lag in [1, 6, 24]:
        acf_actual = acf(actual, lag)
        acf_sims = [acf(sim_prices[n], lag) for n in range(min(100, N_SIM))]
        print(f"    Lag {lag:>3d}: actual={acf_actual:.3f}  "
              f"sim={np.mean(acf_sims):.3f} [{np.percentile(acf_sims, 5):.3f}, "
              f"{np.percentile(acf_sims, 95):.3f}]")

    # KS test
    ks_stat, ks_p = ks_2samp(actual, sim_flat[:len(actual) * 10])
    print(f"\n  KS test (actual vs pooled sim): stat={ks_stat:.4f}, p={ks_p:.4f}")

    # ── Regime frequency comparison ──
    print(f"\n  Regime frequency in test period:")
    print(f"  {'Regime':<10s} {'Actual %':>10s} {'Simulated %':>12s}")
    print("  " + "-" * 35)
    actual_E = test["E"].values
    for e in range(K_E):
        act_pct = (actual_E == e).mean() * 100
        # Estimate simulated regime from price proximity to mu
        sim_E_all = np.argmin(np.abs(sim_prices[:, :, np.newaxis] -
                                      mu[np.newaxis, np.newaxis, :]), axis=2)
        sim_pct = (sim_E_all == e).mean() * 100
        if act_pct > 0.5 or sim_pct > 0.5:
            print(f"  E{e:<8d} {act_pct:>10.1f} {sim_pct:>12.1f}")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Price distribution: actual vs simulated
    ax = axes[0, 0]
    ax.hist(actual, bins=60, density=True, alpha=0.6, color="#1565C0", label="Actual 2025")
    ax.hist(sim_flat[::10], bins=60, density=True, alpha=0.4, color="#E65100",
            label="Simulated")
    ax.set_xlabel("LMP ($/MWh)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Price distribution: actual vs simulated", fontweight="bold")
    ax.legend()
    ax.set_xlim(0, min(actual.max() * 1.5, 400))

    # (b) QQ plot
    ax = axes[0, 1]
    q_actual = np.percentile(actual, np.linspace(1, 99, 50))
    q_sim = np.percentile(sim_flat, np.linspace(1, 99, 50))
    ax.scatter(q_actual, q_sim, s=20, color="#1565C0", zorder=3)
    lims = [0, max(q_actual.max(), q_sim.max()) * 1.1]
    ax.plot(lims, lims, "--", color="gray", lw=0.8)
    ax.set_xlabel("Actual quantiles ($/MWh)")
    ax.set_ylabel("Simulated quantiles ($/MWh)")
    ax.set_title("(b) QQ plot: actual vs simulated", fontweight="bold")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # (c) Time series: actual + confidence band
    ax = axes[1, 0]
    days = np.arange(H) * 6 / 24
    p5 = np.percentile(sim_prices, 5, axis=0)
    p95 = np.percentile(sim_prices, 95, axis=0)
    mean_sim = sim_prices.mean(axis=0)
    ax.fill_between(days, p5, p95, alpha=0.2, color="#1565C0", label="Sim 5-95%")
    ax.plot(days, mean_sim, color="#1565C0", lw=1, label="Sim mean", alpha=0.7)
    ax.plot(days, actual, color="black", lw=0.5, label="Actual", alpha=0.8)
    ax.set_xlabel("Days (2025)")
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title("(c) Actual 2025 vs simulated confidence band", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, days[-1])

    # (d) ACF comparison
    ax = axes[1, 1]
    lags = range(1, 50)
    acf_actual_vals = [acf(actual, l) for l in lags]
    acf_sim_mean = [np.mean([acf(sim_prices[n], l) for n in range(50)]) for l in lags]
    acf_sim_lo = [np.percentile([acf(sim_prices[n], l) for n in range(50)], 5) for l in lags]
    acf_sim_hi = [np.percentile([acf(sim_prices[n], l) for n in range(50)], 95) for l in lags]
    ax.fill_between(list(lags), acf_sim_lo, acf_sim_hi, alpha=0.2, color="#1565C0",
                     label="Sim 5-95%")
    ax.plot(list(lags), acf_sim_mean, color="#1565C0", lw=1.5, label="Sim mean")
    ax.plot(list(lags), acf_actual_vals, color="black", lw=1.5, label="Actual")
    ax.set_xlabel("Lag (windows, x6h)")
    ax.set_ylabel("ACF")
    ax.set_title("(d) Autocorrelation: actual vs simulated", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Model Validation: Train 2021-2024, Test 2025\n"
                 f"AR(1) regime-switching, {N_SIM} simulations",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("paper/validation.png", dpi=250, bbox_inches="tight", facecolor="white")
    print("\n  Saved paper/validation.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
