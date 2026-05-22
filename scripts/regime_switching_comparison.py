"""
regime_switching_comparison.py
==============================
Compare regime-switching models: AR(1) with jump component.

Models:
  1. Unconditional AR(1)+J — no regimes
  2. 3-state classical (low / normal / spike by LMP terciles)
  3. Economic-only (9 regimes, all params from E)
  4. Dynamic-only (8 regimes, all params from D)
  5. Dual-axis JOINT (72 states, all params from joint cell)
  6. Dual-axis FACTORED: mu/sigma/lambda from E, phi from D (the paper's equation)

All use the same AR(1)+jump specification:
  p_t = mu(s) + phi(s) * (p_{t-1} - mu(s)) + sigma(s) * eps_t + J_t * xi_t

Train: 2021-2024, Test: 2025. 1000 simulated paths per model.
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

    # 3-state classical
    train_mask = df["datetime"] <= TRAIN_END
    p33 = df.loc[train_mask, "lmp"].quantile(0.33)
    p67 = df.loc[train_mask, "lmp"].quantile(0.67)
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


def estimate_ar1j_params(train, state_col, n_states):
    """Estimate mu, phi, sigma, lambda, mu_jump per state."""
    mu, phi, sigma, lam, mu_j = {}, {}, {}, {}, {}
    for s in range(n_states):
        sub = train[train[state_col] == s]
        lmp = sub["lmp"].values
        n = len(lmp)
        if n >= 5:
            m = float(np.nanmean(lmp))
            sig = float(np.nanstd(lmp))
            idx_s = sub.index.values
            consec = np.where(np.diff(idx_s) == 1)[0]
            if len(consec) > 5 and sig > 0:
                x_t = lmp[consec] - m
                x_t1 = lmp[consec + 1] - m
                denom = (x_t ** 2).sum()
                p = float((x_t * x_t1).sum() / max(denom, 1e-12))
                p = np.clip(p, -0.99, 0.999)
                resid = x_t1 - p * x_t
                threshold = 2.0 * sig
                jumps = np.abs(resid) > threshold
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


def simulate_model(T, mu, phi, sigma, lam, mu_j, s0, p0, n_test, n_sim, seed):
    rng = np.random.default_rng(seed)
    n_states = T.shape[0]
    paths = np.zeros((n_sim, n_test))

    for path in range(n_sim):
        s_prev = s0
        p_prev = p0
        for t in range(n_test):
            s_curr = rng.choice(n_states, p=T[s_prev])
            m = mu[s_curr]
            p_new = m + phi[s_curr] * (p_prev - m) + sigma[s_curr] * rng.standard_normal()
            if lam[s_curr] > 0 and rng.random() < lam[s_curr]:
                p_new += rng.exponential(max(mu_j[s_curr], 1.0)) * rng.choice([-1, 1])
            p_new = max(p_new, 0.0)
            paths[path, t] = p_new
            s_prev = s_curr
            p_prev = p_new

    return paths


def simulate_factored(T_joint, mu_E, sigma_E, lam_E, muj_E, phi_D,
                      s0, p0, n_test, n_sim, seed):
    """Factored model: mu/sigma/lambda from E, phi from D."""
    rng = np.random.default_rng(seed)
    n_states = T_joint.shape[0]
    paths = np.zeros((n_sim, n_test))

    for path in range(n_sim):
        s_prev = s0
        p_prev = p0
        for t in range(n_test):
            s_curr = rng.choice(n_states, p=T_joint[s_prev])
            e_curr = s_curr // n_D
            d_curr = s_curr % n_D
            m = mu_E[e_curr]
            p_new = m + phi_D[d_curr] * (p_prev - m) + sigma_E[e_curr] * rng.standard_normal()
            if lam_E[e_curr] > 0 and rng.random() < lam_E[e_curr]:
                p_new += rng.exponential(max(muj_E[e_curr], 1.0)) * rng.choice([-1, 1])
            p_new = max(p_new, 0.0)
            paths[path, t] = p_new
            s_prev = s_curr
            p_prev = p_new

    return paths


def compute_metrics(test_lmp, sim_paths, rng):
    p5 = np.percentile(sim_paths, 5, axis=0)
    p95 = np.percentile(sim_paths, 95, axis=0)
    coverage = ((test_lmp >= p5) & (test_lmp <= p95)).mean()

    sim_sample = rng.choice(sim_paths.flatten(), size=len(test_lmp), replace=False)
    wass = sp_stats.wasserstein_distance(test_lmp, sim_sample)

    median_path = np.median(sim_paths, axis=0)
    mae = float(np.mean(np.abs(test_lmp - median_path)))

    return {
        "coverage_90": round(coverage, 4),
        "wasserstein": round(wass, 2),
        "mae_median": round(mae, 2),
        "sim_mean": round(sim_paths.mean(), 1),
        "sim_std": round(sim_paths.std(), 1),
    }, p5, p95


def compute_conditional_coverage(test_lmp, sim_paths, test_D):
    """Coverage conditional on D group."""
    p5 = np.percentile(sim_paths, 5, axis=0)
    p95 = np.percentile(sim_paths, 95, axis=0)
    groups = [
        ("Fast (D0-D1)", [0, 1]),
        ("Moderate (D2-D4)", [2, 3, 4]),
        ("Persistent (D5-D7)", [5, 6, 7]),
    ]
    result = {}
    for label, d_vals in groups:
        mask = np.isin(test_D, d_vals)
        if mask.sum() < 10:
            continue
        sub = test_lmp[mask]
        cov = ((sub >= p5[mask]) & (sub <= p95[mask])).mean()
        result[label] = round(cov * 100, 1)
    return result


def main():
    print("Loading data...")
    df = load_and_prepare()

    train = df[df["datetime"] <= TRAIN_END].copy().reset_index(drop=True)
    test = df[df["datetime"] > TRAIN_END].copy().reset_index(drop=True)
    test_lmp = test["lmp"].values
    test_D = test["D"].values
    p0 = float(train["lmp"].iloc[-1])
    print(f"Train: {len(train)}, Test: {len(test)}")
    print(f"Actual 2025: mean={test_lmp.mean():.1f}, std={test_lmp.std():.1f}")

    results = {}
    cond_cov = {}
    rng = np.random.default_rng(SEED)

    # ── Model 1: Unconditional AR(1)+J ──
    print("\n1. Unconditional AR(1)+J...")
    mu_1, phi_1, sig_1, lam_1, muj_1 = estimate_ar1j_params(train, "E", 1)
    # Hack: assign all to state 0
    train_tmp = train.copy()
    train_tmp["_s0"] = 0
    mu_1, phi_1, sig_1, lam_1, muj_1 = estimate_ar1j_params(train_tmp, "_s0", 1)
    T_1 = np.array([[1.0]])
    sim = simulate_model(T_1, mu_1, phi_1, sig_1, lam_1, muj_1, 0, p0, len(test), N_SIM, SEED)
    m, _, _ = compute_metrics(test_lmp, sim, rng)
    m["model"] = "AR(1)+J no regimes"
    m["K"] = 1
    results["ar1"] = m
    cond_cov["ar1"] = compute_conditional_coverage(test_lmp, sim, test_D)
    print(f"   cov={m['coverage_90']*100:.1f}%, wass={m['wasserstein']}, mae={m['mae_median']}")

    # ── Model 2: 3-state classical ──
    print("\n2. 3-state classical (low/normal/spike)...")
    T_c3 = estimate_transition_matrix(train["C3"].values, 3)
    mu_c3, phi_c3, sig_c3, lam_c3, muj_c3 = estimate_ar1j_params(train, "C3", 3)
    s0_c3 = int(train["C3"].iloc[-1])
    sim = simulate_model(T_c3, mu_c3, phi_c3, sig_c3, lam_c3, muj_c3,
                         s0_c3, p0, len(test), N_SIM, SEED)
    m, _, _ = compute_metrics(test_lmp, sim, rng)
    m["model"] = "3-state classical"
    m["K"] = 3
    results["c3"] = m
    cond_cov["c3"] = compute_conditional_coverage(test_lmp, sim, test_D)
    print(f"   cov={m['coverage_90']*100:.1f}%, wass={m['wasserstein']}, mae={m['mae_median']}")

    # ── Model 3: Economic-only (9 regimes) ──
    print("\n3. Economic-only (9 regimes)...")
    T_e = estimate_transition_matrix(train["E"].values, n_E)
    mu_e, phi_e, sig_e, lam_e, muj_e = estimate_ar1j_params(train, "E", n_E)
    s0_e = int(train["E"].iloc[-1])
    sim = simulate_model(T_e, mu_e, phi_e, sig_e, lam_e, muj_e,
                         s0_e, p0, len(test), N_SIM, SEED)
    m, _, _ = compute_metrics(test_lmp, sim, rng)
    m["model"] = "Economic-only (9)"
    m["K"] = 9
    results["econ"] = m
    cond_cov["econ"] = compute_conditional_coverage(test_lmp, sim, test_D)
    print(f"   cov={m['coverage_90']*100:.1f}%, wass={m['wasserstein']}, mae={m['mae_median']}")

    # ── Model 4: Dynamic-only (8 regimes) ──
    print("\n4. Dynamic-only (8 regimes)...")
    T_d = estimate_transition_matrix(train["D"].values, n_D)
    mu_d, phi_d, sig_d, lam_d, muj_d = estimate_ar1j_params(train, "D", n_D)
    s0_d = int(train["D"].iloc[-1])
    sim = simulate_model(T_d, mu_d, phi_d, sig_d, lam_d, muj_d,
                         s0_d, p0, len(test), N_SIM, SEED)
    m, _, _ = compute_metrics(test_lmp, sim, rng)
    m["model"] = "Dynamic-only (8)"
    m["K"] = 8
    results["dyn"] = m
    cond_cov["dyn"] = compute_conditional_coverage(test_lmp, sim, test_D)
    print(f"   cov={m['coverage_90']*100:.1f}%, wass={m['wasserstein']}, mae={m['mae_median']}")

    # ── Model 5: Dual-axis JOINT (72 states) ──
    print("\n5. Dual-axis joint (72 states)...")
    train["state"] = train["E"] * n_D + train["D"]
    test["state"] = test["E"] * n_D + test["D"]
    n_states = n_E * n_D
    T_joint = estimate_transition_matrix(train["state"].values, n_states)
    mu_j, phi_j, sig_j, lam_j, muj_j = estimate_ar1j_params(train, "state", n_states)
    s0_j = int(train["state"].iloc[-1])
    sim = simulate_model(T_joint, mu_j, phi_j, sig_j, lam_j, muj_j,
                         s0_j, p0, len(test), N_SIM, SEED)
    m, _, _ = compute_metrics(test_lmp, sim, rng)
    m["model"] = "Dual-axis joint (72)"
    m["K"] = 72
    results["joint"] = m
    cond_cov["joint"] = compute_conditional_coverage(test_lmp, sim, test_D)
    print(f"   cov={m['coverage_90']*100:.1f}%, wass={m['wasserstein']}, mae={m['mae_median']}")

    # ── Model 6: Dual-axis FACTORED (mu/sigma/lambda from E, phi from D) ──
    print("\n6. Dual-axis factored (E→level, D→persistence)...")
    # Estimate phi per D regime
    phi_D_params = {}
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
                p = float((x_t * x_t1).sum() / max((x_t ** 2).sum(), 1e-12))
                p = np.clip(p, -0.99, 0.999)
            else:
                p = 0.5
        else:
            p = 0.5
        phi_D_params[d] = p

    sim = simulate_factored(T_joint, mu_e, sig_e, lam_e, muj_e, phi_D_params,
                            s0_j, p0, len(test), N_SIM, SEED)
    m, _, _ = compute_metrics(test_lmp, sim, rng)
    m["model"] = "Dual-axis factored"
    m["K"] = "9+8"
    results["factored"] = m
    cond_cov["factored"] = compute_conditional_coverage(test_lmp, sim, test_D)
    print(f"   cov={m['coverage_90']*100:.1f}%, wass={m['wasserstein']}, mae={m['mae_median']}")

    # ── Summary table ──
    print(f"\n{'='*80}")
    print(f"OUT-OF-SAMPLE COMPARISON (2025)")
    print(f"{'='*80}")
    print(f"Actual 2025: mean={test_lmp.mean():.1f}, std={test_lmp.std():.1f}")
    print()
    keys = ["ar1", "c3", "econ", "dyn", "joint", "factored"]
    print(f"{'Model':<25} {'K':>5} {'Cov90%':>7} {'Wass':>7} {'MAE':>7} {'SimMu':>7}")
    print(f"{'-'*25} {'-----':>5} {'-------':>7} {'-------':>7} {'-------':>7} {'-------':>7}")
    for key in keys:
        r = results[key]
        print(f"{r['model']:<25} {str(r['K']):>5} {r['coverage_90']*100:>6.1f}% "
              f"{r['wasserstein']:>7.2f} {r['mae_median']:>7.2f} {r['sim_mean']:>7.1f}")

    # ── Conditional coverage table ──
    print(f"\n{'='*80}")
    print(f"CONDITIONAL 90% COVERAGE BY DYNAMIC REGIME")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Fast':>8} {'Moderate':>10} {'Persistent':>12}")
    print("-" * 58)
    for key in keys:
        r = results[key]
        cc = cond_cov[key]
        fast = cc.get("Fast (D0-D1)", "-")
        mod = cc.get("Moderate (D2-D4)", "-")
        pers = cc.get("Persistent (D5-D7)", "-")
        print(f"{r['model']:<25} {fast:>7}% {mod:>9}% {pers:>11}%")

    # Save
    rows = [results[k] for k in keys]
    pd.DataFrame(rows).to_csv(OUT_DIR / "model_comparison_full.csv", index=False)
    print(f"\nSaved model_comparison_full.csv")

    # ── Figure ──
    print("Generating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    models_short = ["AR(1)+J\nno regimes", "3-state\nclassical", "Economic\nonly (9)",
                    "Dynamic\nonly (8)", "Dual-axis\njoint (72)", "Dual-axis\nfactored"]
    colors = ["#aaaaaa", "#aaaaaa", "#6baed6", "#6baed6", "#c0392b", "#e67e22"]

    # (a) Coverage
    ax = axes[0]
    coverages = [results[k]["coverage_90"] * 100 for k in keys]
    bars = ax.bar(models_short, coverages, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(90, color="black", ls="--", lw=1, label="Nominal 90%")
    ax.set_ylabel("90% coverage (%)")
    ax.set_ylim(70, 100)
    ax.set_title("(a) Prediction interval coverage", fontweight="bold")
    ax.legend(fontsize=9)
    for bar, v in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8)

    # (b) Wasserstein
    ax = axes[1]
    wass_vals = [results[k]["wasserstein"] for k in keys]
    bars = ax.bar(models_short, wass_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Wasserstein distance ($/MWh)")
    ax.set_title("(b) Distributional distance", fontweight="bold")
    for bar, v in zip(bars, wass_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)

    # (c) Conditional coverage — persistent regimes only
    ax = axes[2]
    pers_vals = [cond_cov[k].get("Persistent (D5-D7)", 0) for k in keys]
    bars = ax.bar(models_short, pers_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(90, color="black", ls="--", lw=1, label="Nominal 90%")
    ax.set_ylabel("Coverage in persistent regimes (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(c) Coverage where it matters most", fontweight="bold")
    ax.legend(fontsize=9)
    for bar, v in zip(bars, pers_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model comparison: out-of-sample 2025", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_model_comparison.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
