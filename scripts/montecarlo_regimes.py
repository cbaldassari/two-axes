"""
montecarlo_regimes.py
=====================
Monte Carlo simulation using the dual-axis regime model.

Two independent Markov chains (economic + dynamic) generate regime paths.
Prices are sampled from the empirical joint distribution P(LMP | econ, dyn).
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

RESULTS = Path("results")
SEED = 42
N_PATHS = 1000
HORIZON = 168  # steps = 168 * 6h = 1008h = 42 days


def build_transition_matrix(labels, K):
    T = np.zeros((K, K))
    for i in range(len(labels) - 1):
        T[labels[i], labels[i + 1]] += 1
    # Normalize rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums
    return T


def simulate_markov(T, initial_state, n_steps, rng):
    """Simulate one Markov chain path."""
    K = T.shape[0]
    path = np.empty(n_steps, dtype=int)
    path[0] = initial_state
    for t in range(1, n_steps):
        path[t] = rng.choice(K, p=T[path[t - 1]])
    return path


def main():
    t0 = time.time()
    print("=" * 70)
    print("  MONTE CARLO SIMULATION — Dual-Axis Regime Model")
    print(f"  Paths: {N_PATHS}  |  Horizon: {HORIZON} steps ({HORIZON * 6}h = {HORIZON * 6 / 24:.0f} days)")
    print("=" * 70)

    # ── Load data ──
    lab_fe = pd.read_parquet(RESULTS / "exp_FE/step04/labels.parquet")
    lab_mom = pd.read_parquet(RESULTS / "exp_C/step04/labels.parquet")
    pre = pd.read_parquet(RESULTS / "preprocessed.parquet")
    for d in [lab_fe, lab_mom, pre]:
        d["datetime"] = pd.to_datetime(d["datetime"])

    df = lab_fe.merge(lab_mom, on="datetime", suffixes=("_fe", "_mom"))
    df = df.merge(pre[["datetime", "lmp"]], on="datetime")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Order regimes
    fe_order = df.groupby("cluster_fe")["lmp"].mean().sort_values().index.tolist()
    fe_rank = {r: i for i, r in enumerate(fe_order)}
    df["econ"] = df["cluster_fe"].map(fe_rank)

    mom_order = df.groupby("cluster_mom")["lmp"].mean().sort_values().index.tolist()
    mom_rank = {r: i for i, r in enumerate(mom_order)}
    df["dyn"] = df["cluster_mom"].map(mom_rank)

    K_econ = len(fe_order)
    K_dyn = len(mom_order)

    # ── Build transition matrices ──
    T_econ = build_transition_matrix(df["econ"].values, K_econ)
    T_dyn = build_transition_matrix(df["dyn"].values, K_dyn)

    print(f"\n  Economic transition matrix: {K_econ}x{K_econ}")
    print(f"  Dynamic transition matrix: {K_dyn}x{K_dyn}")

    # ── Build empirical price distributions P(LMP | econ, dyn) ──
    price_dist = {}
    for e in range(K_econ):
        for d in range(K_dyn):
            mask = (df["econ"] == e) & (df["dyn"] == d)
            prices = df.loc[mask, "lmp"].values
            if len(prices) > 0:
                price_dist[(e, d)] = prices
            else:
                # Fallback: use marginal on economic regime
                price_dist[(e, d)] = df.loc[df["econ"] == e, "lmp"].values

    # Check coverage
    n_joint = sum(1 for e in range(K_econ) for d in range(K_dyn)
                  if ((df["econ"] == e) & (df["dyn"] == d)).sum() > 0)
    print(f"  Joint states with data: {n_joint}/{K_econ * K_dyn}")

    # ── Initial state: last observed ──
    last_econ = df["econ"].iloc[-1]
    last_dyn = df["dyn"].iloc[-1]
    last_lmp = df["lmp"].iloc[-1]
    last_date = df["datetime"].iloc[-1]
    print(f"\n  Initial state (last observed):")
    print(f"    Date: {last_date}")
    print(f"    Economic regime: E{last_econ} (LMP={last_lmp:.1f} $/MWh)")
    print(f"    Dynamic regime: D{last_dyn}")

    # ── Simulate ──
    rng = np.random.RandomState(SEED)
    print(f"\n  Simulating {N_PATHS} paths x {HORIZON} steps ...")

    all_prices = np.empty((N_PATHS, HORIZON))
    all_econ = np.empty((N_PATHS, HORIZON), dtype=int)
    all_dyn = np.empty((N_PATHS, HORIZON), dtype=int)

    for p in range(N_PATHS):
        path_econ = simulate_markov(T_econ, last_econ, HORIZON, rng)
        path_dyn = simulate_markov(T_dyn, last_dyn, HORIZON, rng)
        prices = np.array([
            rng.choice(price_dist[(path_econ[t], path_dyn[t])])
            for t in range(HORIZON)
        ])
        all_prices[p] = prices
        all_econ[p] = path_econ
        all_dyn[p] = path_dyn

    print(f"  Done.")

    # ── Statistics ──
    hours = np.arange(HORIZON) * 6
    days = hours / 24

    mean_price = all_prices.mean(axis=0)
    p5 = np.percentile(all_prices, 5, axis=0)
    p25 = np.percentile(all_prices, 25, axis=0)
    p75 = np.percentile(all_prices, 75, axis=0)
    p95 = np.percentile(all_prices, 95, axis=0)

    # Probability of being in spike regime (econ >= 7, i.e. R7/R8/R9)
    prob_spike = (all_econ >= 7).mean(axis=0)
    # Probability of being in sticky regime (dyn >= 5)
    prob_sticky = (all_dyn >= 5).mean(axis=0)

    print(f"\n  Forecast statistics at key horizons:")
    print(f"  {'Horizon':<12s} {'Mean LMP':>10s} {'P5':>8s} {'P95':>8s} "
          f"{'P(spike)':>10s} {'P(sticky)':>10s}")
    print("  " + "-" * 60)
    for h_idx in [0, 3, 7, 13, 27, 55, 111, 167]:
        if h_idx < HORIZON:
            h = hours[h_idx]
            print(f"  {h:>4d}h ({h/24:>4.0f}d) {mean_price[h_idx]:>10.1f} "
                  f"{p5[h_idx]:>8.1f} {p95[h_idx]:>8.1f} "
                  f"{prob_spike[h_idx]:>10.1%} {prob_sticky[h_idx]:>10.1%}")

    # ── VaR analysis ──
    # What is the worst 5% price over the next 7 days?
    week_prices = all_prices[:, :28]  # 28 steps = 168h = 7 days
    max_price_per_path = week_prices.max(axis=1)
    var_95 = np.percentile(max_price_per_path, 95)
    var_99 = np.percentile(max_price_per_path, 99)
    mean_max = max_price_per_path.mean()

    print(f"\n  7-day VaR analysis (max price per path over 168h):")
    print(f"    Mean max price:  {mean_max:.1f} $/MWh")
    print(f"    VaR 95%:         {var_95:.1f} $/MWh")
    print(f"    VaR 99%:         {var_99:.1f} $/MWh")

    # ── Expected sojourn in spike regime ──
    # Among paths that enter spike, how long do they stay?
    spike_sojourns = []
    for p in range(N_PATHS):
        in_spike = False
        soj = 0
        for t in range(HORIZON):
            if all_econ[p, t] >= 7:
                in_spike = True
                soj += 1
            else:
                if in_spike and soj > 0:
                    spike_sojourns.append(soj * 6)  # hours
                    soj = 0
                in_spike = False
        if in_spike and soj > 0:
            spike_sojourns.append(soj * 6)

    if spike_sojourns:
        print(f"\n  Spike sojourn (simulated, among paths entering spike):")
        print(f"    Episodes: {len(spike_sojourns)}")
        print(f"    Mean duration: {np.mean(spike_sojourns):.0f}h")
        print(f"    Median: {np.median(spike_sojourns):.0f}h")
        print(f"    Max: {np.max(spike_sojourns):.0f}h")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.rcParams["font.family"] = "serif"

    # (a) Price fan chart
    ax = axes[0, 0]
    ax.fill_between(days, p5, p95, alpha=0.15, color="#1565C0", label="5-95%")
    ax.fill_between(days, p25, p75, alpha=0.3, color="#1565C0", label="25-75%")
    ax.plot(days, mean_price, color="#1565C0", lw=2, label="Mean")
    ax.axhline(last_lmp, color="gray", ls="--", lw=0.8, label=f"Current ({last_lmp:.0f})")
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title("(a) Price forecast fan chart", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, days[-1])

    # (b) Spike probability over time
    ax = axes[0, 1]
    ax.plot(days, prob_spike * 100, color="#E65100", lw=2, label="P(spike regime)")
    ax.plot(days, prob_sticky * 100, color="#1565C0", lw=2, ls="--",
            label="P(sticky regime)")
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Probability (%)")
    ax.set_title("(b) Regime probabilities over horizon", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, days[-1])
    ax.set_ylim(0, 100)

    # (c) Distribution of max price over 7 days
    ax = axes[1, 0]
    ax.hist(max_price_per_path, bins=50, color="#1565C0", alpha=0.7, edgecolor="none")
    ax.axvline(var_95, color="red", ls="--", lw=1.5, label=f"VaR 95% = {var_95:.0f}")
    ax.axvline(var_99, color="darkred", ls="--", lw=1.5, label=f"VaR 99% = {var_99:.0f}")
    ax.set_xlabel("Max LMP over 7 days ($/MWh)")
    ax.set_ylabel("Count")
    ax.set_title("(c) 7-day VaR distribution", fontweight="bold")
    ax.legend(fontsize=8)

    # (d) Sample paths (10 random)
    ax = axes[1, 1]
    for p in range(10):
        ax.plot(days, all_prices[p], lw=0.5, alpha=0.6)
    ax.plot(days, mean_price, color="black", lw=2, label="Mean")
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title("(d) Sample simulated paths (10)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, days[-1])

    fig.suptitle("Monte Carlo Simulation: Dual-Axis Regime Model\n"
                 f"({N_PATHS} paths, {HORIZON * 6 / 24:.0f}-day horizon, "
                 f"starting from E{last_econ}/D{last_dyn})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("paper/montecarlo.png", dpi=250, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved paper/montecarlo.png")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
