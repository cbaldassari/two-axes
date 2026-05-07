"""
dual_axis_model.py
==================
Dual-Axis Generative Model: Jump-Diffusion on the 9×8 regime grid.

1. Cross-tabulates Level (9 economic) × Memory (8 dynamic) regimes
2. Estimates per-cell parameters: μ, σ, φ, λ_jump, μ_jump
3. Estimates transition matrices T_E (9×9) and T_D (8×8)
4. Simulates 500 paths on the 2025 test set
5. Compares simulated vs actual distributions

Output:
  results/dual_axis/
    cross_tab.csv         — 9×8 population matrix
    cell_params.csv       — per-cell parameters
    transition_E.csv      — economic transition matrix (9×9)
    transition_D.csv      — dynamic transition matrix (8×8)
    simulation_metrics.csv — OOS validation metrics
    fig_cross_tab.png     — heatmap of regime populations
    fig_validation.png    — simulated vs actual comparison
    fig_qq.png            — QQ plot
    fig_acf.png           — ACF comparison
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
OUT_DIR = RESULTS_DIR / "dual_axis"
SEED = C.RANDOM_STATE
DPI = 150
N_SIM = 1000
TRAIN_END = "2024-12-31"


def load_data():
    """Load labels + LMP + FE features."""
    fe_labels = pd.read_parquet(RESULTS_DIR / "exp_FE" / "step04" / "labels.parquet")
    mom_labels = pd.read_parquet(RESULTS_DIR / "exp_C" / "step04" / "labels.parquet")

    pre = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    lmp_lookup = pre.set_index("datetime")["lmp"]

    fe_feats = pd.read_parquet(RESULTS_DIR / "exp_FE" / "embeddings.parquet")

    df = pd.DataFrame({
        "datetime": pd.to_datetime(fe_labels["datetime"]),
        "E": fe_labels["cluster"].values,
        "D": mom_labels["cluster"].values,
    })

    # Add LMP
    df["lmp"] = lmp_lookup.reindex(df["datetime"]).values

    # Add ACF features for characterization
    feat_names = C.FE_FEATURES
    for fname in feat_names:
        if fname in fe_feats.columns:
            df[fname] = fe_feats[fname].values

    return df


def order_regimes(df):
    """Order E by mean LMP ascending, D by mean ACF_6h ascending."""
    # Order E by LMP
    e_means = df.groupby("E")["lmp"].mean().sort_values()
    e_map = {old: new for new, old in enumerate(e_means.index)}
    df["E"] = df["E"].map(e_map)

    # Order D by ACF_6h
    if "acf_6h" in df.columns:
        d_means = df.groupby("D")["acf_6h"].mean().sort_values()
        d_map = {old: new for new, old in enumerate(d_means.index)}
        df["D"] = df["D"].map(d_map)

    return df


def cross_tabulation(df):
    """9×8 cross-tabulation."""
    ct = pd.crosstab(df["E"], df["D"], margins=True)
    ct_pct = pd.crosstab(df["E"], df["D"], normalize="all") * 100
    return ct, ct_pct


def estimate_cell_params(df):
    """Estimate per-cell (E, D) parameters for jump-diffusion.

    mu(E), sigma(E), lambda(E), mu_jump(E) are estimated per economic regime E
    (pooling across all D), consistent with the paper's equation.
    phi(D) is estimated per dynamic regime D (pooling across all E).
    """
    # ── Step 1: estimate jump params per E (pooling across D) ──
    jump_params_E = {}
    for e in sorted(df["E"].unique()):
        sub_e = df[df["E"] == e]
        lmp = sub_e["lmp"].values
        mu_e = float(np.nanmean(lmp))
        sigma_e = float(np.nanstd(lmp))

        idx = sub_e.index.values
        consecutive = np.where(np.diff(idx) == 1)[0]
        if len(consecutive) > 10 and sigma_e > 0:
            residuals = lmp[consecutive + 1] - (mu_e + 0.5 * (lmp[consecutive] - mu_e))
            threshold = 2.0 * sigma_e
            jumps = np.abs(residuals) > threshold
            lambda_jump = float(jumps.mean())
            if jumps.sum() > 0:
                mu_jump = float(np.abs(residuals[jumps]).mean())
                sigma_jump = float(np.abs(residuals[jumps]).std()) if jumps.sum() > 1 else mu_jump * 0.5
            else:
                mu_jump, sigma_jump = 0.0, 0.0
        else:
            lambda_jump, mu_jump, sigma_jump = 0.0, 0.0, 0.0

        jump_params_E[e] = {
            "mu": round(mu_e, 2), "sigma": round(sigma_e, 2),
            "lambda_jump": round(lambda_jump, 4),
            "mu_jump": round(mu_jump, 2), "sigma_jump": round(sigma_jump, 2),
        }

    # ── Step 2: estimate phi per D (pooling across E) ──
    phi_D = {}
    for d in sorted(df["D"].unique()):
        sub_d = df[df["D"] == d]
        lmp = sub_d["lmp"].values
        mu_d = float(np.nanmean(lmp))
        idx = sub_d.index.values
        consecutive = np.where(np.diff(idx) == 1)[0]
        if len(consecutive) > 10:
            x_t = lmp[consecutive] - mu_d
            x_t1 = lmp[consecutive + 1] - mu_d
            denom = (x_t ** 2).sum()
            phi = float((x_t * x_t1).sum() / max(denom, 1e-12))
            phi = np.clip(phi, 0.0, 0.999)
        else:
            phi = float(sub_d["acf_1h"].mean()) if "acf_1h" in sub_d.columns else 0.9
        phi_D[d] = round(phi, 4)

    # ── Step 3: assemble per-cell records ──
    records = []
    for e in sorted(df["E"].unique()):
        for d in sorted(df["D"].unique()):
            mask = (df["E"] == e) & (df["D"] == d)
            n = int(mask.sum())
            jp = jump_params_E[e]
            records.append({
                "E": e, "D": d, "n": n,
                "mu": jp["mu"], "sigma": jp["sigma"],
                "phi": phi_D[d],
                "lambda_jump": jp["lambda_jump"],
                "mu_jump": jp["mu_jump"],
                "sigma_jump": jp["sigma_jump"],
            })

    return pd.DataFrame(records)


def estimate_transition_matrices(df):
    """Estimate T_E and T_D from consecutive windows."""
    n_E = df["E"].max() + 1
    n_D = df["D"].max() + 1

    T_E = np.zeros((n_E, n_E))
    T_D = np.zeros((n_D, n_D))

    idx = df.index.values
    E = df["E"].values
    D = df["D"].values

    for i in range(len(idx) - 1):
        if idx[i + 1] - idx[i] == 1:  # consecutive
            T_E[E[i], E[i + 1]] += 1
            T_D[D[i], D[i + 1]] += 1

    # Normalize rows
    for i in range(n_E):
        s = T_E[i].sum()
        if s > 0:
            T_E[i] /= s

    for i in range(n_D):
        s = T_D[i].sum()
        if s > 0:
            T_D[i] /= s

    return T_E, T_D


def simulate_paths(T_E, T_D, params_df, n_steps, n_paths, p0, e0, d0, rng):
    """
    Simulate n_paths price paths of n_steps each.

    p_t = mu(E) + phi(D) * (p_{t-1} - mu(E)) + sigma(E) * eps + J * xi

    where:
      E_{t+1} ~ Categorical(T_E[E_t])
      D_{t+1} ~ Categorical(T_D[D_t])
      J ~ Bernoulli(lambda(E))
      xi ~ Exponential(mu_jump) * sign(random)
    """
    n_E = T_E.shape[0]
    n_D = T_D.shape[0]

    # Build param lookup: (E, D) -> (mu, sigma, phi, lambda, mu_jump)
    param_grid = {}
    for _, row in params_df.iterrows():
        e, d = int(row["E"]), int(row["D"])
        param_grid[(e, d)] = (
            row["mu"], row["sigma"], row["phi"],
            row["lambda_jump"], row["mu_jump"], row["sigma_jump"]
        )

    # Global fallbacks
    global_mu = params_df["mu"].mean()
    global_sigma = params_df["sigma"].mean()
    global_phi = params_df["phi"].mean()

    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = p0

    E_paths = np.zeros((n_paths, n_steps), dtype=int)
    D_paths = np.zeros((n_paths, n_steps), dtype=int)
    E_paths[:, 0] = e0
    D_paths[:, 0] = d0

    for t in range(1, n_steps):
        for s in range(n_paths):
            # Transition
            E_paths[s, t] = rng.choice(n_E, p=T_E[E_paths[s, t-1]])
            D_paths[s, t] = rng.choice(n_D, p=T_D[D_paths[s, t-1]])

            e, d = E_paths[s, t], D_paths[s, t]
            if (e, d) in param_grid and not np.isnan(param_grid[(e, d)][0]):
                mu, sigma, phi, lam, mu_j, sig_j = param_grid[(e, d)]
            else:
                mu, sigma, phi = global_mu, global_sigma, global_phi
                lam, mu_j, sig_j = 0.0, 0.0, 0.0

            # AR(1) + jump
            p_prev = paths[s, t-1]
            eps = rng.standard_normal()
            continuous = mu + phi * (p_prev - mu) + sigma * eps

            # Jump component
            if lam > 0 and rng.random() < lam:
                jump_size = rng.exponential(max(mu_j, 1.0))
                jump_sign = rng.choice([-1, 1])
                jump = jump_sign * jump_size
            else:
                jump = 0.0

            paths[s, t] = max(continuous + jump, 0.0)  # price >= 0

    return paths, E_paths, D_paths


def make_plots(df_train, df_test, params_df, sim_paths, T_E, T_D, ct_pct):
    """Generate all figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Cross-tabulation heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    n_E = df_train["E"].max() + 1
    n_D = df_train["D"].max() + 1

    # Counts
    ct_vals = np.zeros((n_E, n_D))
    for e in range(n_E):
        for d in range(n_D):
            mask = (df_train["E"] == e) & (df_train["D"] == d)
            ct_vals[e, d] = mask.sum()

    im = axes[0].imshow(ct_vals, cmap="YlOrRd", aspect="auto")
    for i in range(n_E):
        for j in range(n_D):
            v = int(ct_vals[i, j])
            color = "white" if v > ct_vals.max() * 0.6 else "black"
            axes[0].text(j, i, str(v), ha="center", va="center", fontsize=8, color=color)
    axes[0].set_xlabel("Memory (D) — low to high ACF")
    axes[0].set_ylabel("Level (E) — low to high LMP")
    axes[0].set_xticks(range(n_D)); axes[0].set_xticklabels([f"M{d}" for d in range(n_D)])
    axes[0].set_yticks(range(n_E)); axes[0].set_yticklabels([f"E{e}" for e in range(n_E)])
    axes[0].set_title("Window count per (E, D) cell")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Mean LMP per cell
    lmp_grid = np.full((n_E, n_D), np.nan)
    df_all = pd.concat([df_train, df_test])
    for e in range(n_E):
        for d in range(n_D):
            mask = (df_all["E"] == e) & (df_all["D"] == d)
            if mask.sum() > 0:
                lmp_grid[e, d] = df_all.loc[mask, "lmp"].mean()

    im2 = axes[1].imshow(lmp_grid, cmap="RdYlGn_r", aspect="auto")
    for i in range(n_E):
        for j in range(n_D):
            v = lmp_grid[i, j]
            if not np.isnan(v):
                color = "white" if v > 80 else "black"
                axes[1].text(j, i, f"${v:.0f}", ha="center", va="center", fontsize=8, color=color)
            else:
                axes[1].text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
    axes[1].set_xlabel("Memory (D) — low to high ACF")
    axes[1].set_ylabel("Level (E) — low to high LMP")
    axes[1].set_xticks(range(n_D)); axes[1].set_xticklabels([f"M{d}" for d in range(n_D)])
    axes[1].set_yticks(range(n_E)); axes[1].set_yticklabels([f"E{e}" for e in range(n_E)])
    axes[1].set_title("Mean LMP ($/MWh) per cell")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    fig.suptitle("Dual-Axis Regime Grid: Level × Memory", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_cross_tab.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Validation: time series only ──
    actual = df_test["lmp"].values
    sim_mean = sim_paths.mean(axis=0)
    sim_p5 = np.percentile(sim_paths, 5, axis=0)
    sim_p95 = np.percentile(sim_paths, 95, axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    n_test = len(actual)
    x = np.arange(n_test)
    ax.fill_between(x, sim_p5, sim_p95, alpha=0.3, color="steelblue", label="5--95\% simulated band")
    ax.plot(x, sim_mean, color="steelblue", lw=0.5, alpha=0.7, label="Simulated mean")
    ax.plot(x, actual, color="black", lw=0.5, alpha=0.8, label="Actual 2025")
    ax.set_xlabel("Window index (2025)")
    ax.set_ylabel("LMP ($/MWh)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_timeseries_validation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("  Plots saved.", flush=True)


def main():
    t0 = time.time()
    print("\n" + "=" * 75)
    print("  DUAL-AXIS GENERATIVE MODEL: Jump-Diffusion on 9×8 Regime Grid")
    print("=" * 75)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & order ──
    print("\n  [1/6] Loading data ...", flush=True)
    df = load_data()
    df = order_regimes(df)
    n_E = df["E"].max() + 1
    n_D = df["D"].max() + 1
    print(f"  {len(df)} windows, {n_E} Level × {n_D} Memory = {n_E * n_D} cells")

    # ── 2. Cross-tabulation ──
    print("\n  [2/6] Cross-tabulation ...", flush=True)
    ct, ct_pct = cross_tabulation(df)
    print(ct.to_string())
    ct.to_csv(OUT_DIR / "cross_tab.csv")

    populated = ((ct.iloc[:-1, :-1] > 0).sum().sum())
    empty = n_E * n_D - populated
    print(f"\n  Populated cells: {populated}/{n_E * n_D} ({empty} empty)")

    # ── 3. Train/test split ──
    print("\n  [3/6] Train/test split ...", flush=True)
    train_mask = df["datetime"] <= TRAIN_END
    df_train = df[train_mask].copy().reset_index(drop=True)
    df_test = df[~train_mask].copy().reset_index(drop=True)
    print(f"  Train: {len(df_train)} windows (to {TRAIN_END})")
    print(f"  Test:  {len(df_test)} windows (2025)")

    # ── 4. Estimate parameters ──
    print("\n  [4/6] Estimating per-cell parameters ...", flush=True)
    params = estimate_cell_params(df_train)
    params.to_csv(OUT_DIR / "cell_params.csv", index=False)

    print("\n  Cell parameters (populated cells):")
    populated_params = params[params["n"] >= 5].sort_values(["E", "D"])
    print(populated_params.to_string(index=False))

    # ── 5. Transition matrices ──
    print("\n  [5/6] Transition matrices ...", flush=True)
    T_E, T_D = estimate_transition_matrices(df_train)

    pd.DataFrame(T_E, index=[f"E{i}" for i in range(n_E)],
                 columns=[f"E{i}" for i in range(n_E)]).to_csv(OUT_DIR / "transition_E.csv")
    pd.DataFrame(T_D, index=[f"D{i}" for i in range(n_D)],
                 columns=[f"D{i}" for i in range(n_D)]).to_csv(OUT_DIR / "transition_D.csv")

    print(f"  T_E diagonal: {np.diag(T_E).round(3)}")
    print(f"  T_D diagonal: {np.diag(T_D).round(3)}")

    # ── 6. Simulate ──
    print(f"\n  [6/6] Simulating {N_SIM} paths on 2025 ...", flush=True)
    rng = np.random.default_rng(SEED)
    n_test = len(df_test)
    p0 = df_test["lmp"].iloc[0]
    e0 = df_test["E"].iloc[0]
    d0 = df_test["D"].iloc[0]

    sim_paths, E_paths, D_paths = simulate_paths(
        T_E, T_D, params, n_test, N_SIM, p0, e0, d0, rng
    )

    # ── Metrics ──
    actual = df_test["lmp"].values
    sim_flat = sim_paths.flatten()
    sim_means = sim_paths.mean(axis=1)

    metrics = {
        "actual_mean": round(float(actual.mean()), 2),
        "actual_std": round(float(actual.std()), 2),
        "actual_skew": round(float(stats.skew(actual)), 2),
        "actual_kurt": round(float(stats.kurtosis(actual)), 2),
        "actual_max": round(float(actual.max()), 2),
        "actual_p95": round(float(np.percentile(actual, 95)), 2),
        "sim_mean": round(float(sim_flat.mean()), 2),
        "sim_std": round(float(sim_flat.std()), 2),
        "sim_skew": round(float(stats.skew(sim_flat)), 2),
        "sim_kurt": round(float(stats.kurtosis(sim_flat)), 2),
        "sim_max": round(float(sim_flat.max()), 2),
        "sim_p95": round(float(np.percentile(sim_flat, 95)), 2),
        "coverage_90": round(float(
            ((actual >= np.percentile(sim_paths, 5, axis=0)) &
             (actual <= np.percentile(sim_paths, 95, axis=0))).mean()
        ), 4),
        "ks_statistic": round(float(stats.ks_2samp(actual, sim_flat).statistic), 4),
        "ks_pvalue": float(stats.ks_2samp(actual, sim_flat).pvalue),
        "wasserstein": round(float(stats.wasserstein_distance(actual, sim_flat)), 2),
    }

    print("\n  Validation metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    pd.DataFrame([metrics]).to_csv(OUT_DIR / "simulation_metrics.csv", index=False)

    # ── Plots ──
    print("\n  Generating plots ...", flush=True)
    make_plots(df_train, df_test, params, sim_paths, T_E, T_D, ct_pct)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
