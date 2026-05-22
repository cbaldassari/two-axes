"""
temporal_metrics.py
===================
Compare models on temporal structure metrics (ACF, volatility, sojourn, conditional coverage).
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import acf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
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


def estimate_ar1(train_df, col, n):
    mu, phi, sigma = {}, {}, {}
    for s in range(n):
        sub = train_df[train_df[col] == s]
        lmp = sub["lmp"].values
        if len(lmp) >= 5:
            m = float(np.nanmean(lmp))
            idx_s = sub.index.values
            consec = np.where(np.diff(idx_s) == 1)[0]
            if len(consec) > 5:
                xt = lmp[consec] - m
                xt1 = lmp[consec + 1] - m
                p = float((xt * xt1).sum() / max((xt ** 2).sum(), 1e-12))
                p = np.clip(p, -0.99, 0.999)
                sig = float(np.std(xt1 - p * xt))
            else:
                p, sig = 0.5, float(np.nanstd(lmp))
        else:
            m = float(train_df["lmp"].mean())
            p, sig = 0.5, float(train_df["lmp"].std())
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
            p_new = m + phi[s_curr] * (p_prev - m) + sigma[s_curr] * rng.standard_normal()
            paths[path, t] = max(p_new, 0.0)
            s_prev, p_prev = s_curr, p_new
    return paths


def main():
    df = load()
    train = df[df["datetime"] <= TRAIN_END].copy().reset_index(drop=True)
    test = df[df["datetime"] > TRAIN_END].copy().reset_index(drop=True)
    test_lmp = test["lmp"].values
    p0 = float(train["lmp"].iloc[-1])
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Build models
    # AR(1) no regimes
    lmp_tr = train["lmp"].values
    mu_all = float(np.mean(lmp_tr))
    xt = lmp_tr[:-1] - mu_all
    xt1 = lmp_tr[1:] - mu_all
    phi_all = float(np.clip((xt * xt1).sum() / (xt ** 2).sum(), -0.99, 0.999))
    sig_all = float(np.std(xt1 - phi_all * xt))
    sim_ar1 = simulate(np.array([[1.0]]), {0: mu_all}, {0: phi_all}, {0: sig_all},
                        0, p0, len(test), N_SIM, SEED)

    # Economic only (9)
    T_e = estimate_T(train["E"].values, n_E)
    mu_e, phi_e, sig_e = estimate_ar1(train, "E", n_E)
    sim_econ = simulate(T_e, mu_e, phi_e, sig_e, int(train["E"].iloc[-1]),
                        p0, len(test), N_SIM, SEED)

    # Dual-axis joint (72)
    train["state"] = train["E"] * n_D + train["D"]
    test["state"] = test["E"] * n_D + test["D"]
    T_j = estimate_T(train["state"].values, n_E * n_D)
    mu_j, phi_j, sig_j = estimate_ar1(train, "state", n_E * n_D)
    sim_joint = simulate(T_j, mu_j, phi_j, sig_j, int(train["state"].iloc[-1]),
                         p0, len(test), N_SIM, SEED)

    models = {"AR(1)": sim_ar1, "Econ-9": sim_econ, "Dual-72": sim_joint}

    # ── 1. ACF comparison ──
    max_lag = 20
    acf_actual = acf(test_lmp, nlags=max_lag, fft=True)

    print("\n=== ACF RMSE (lags 1-20) ===")
    for name, sim in models.items():
        acfs = []
        for i in range(min(200, sim.shape[0])):
            acfs.append(acf(sim[i], nlags=max_lag, fft=True))
        acf_mean = np.mean(acfs, axis=0)
        rmse = float(np.sqrt(np.mean((acf_mean[1:] - acf_actual[1:]) ** 2)))
        print(f"  {name:>10}: RMSE={rmse:.4f}")
        # Print key lags
        for lag in [1, 4, 8, 16]:
            print(f"     lag {lag:>2}: actual={acf_actual[lag]:.3f}  sim={acf_mean[lag]:.3f}  "
                  f"err={abs(acf_mean[lag]-acf_actual[lag]):.3f}")

    # ── 2. Volatility clustering (rolling std) ──
    print("\n=== Rolling volatility RMSE (window=20) ===")
    rstd_actual = pd.Series(test_lmp).rolling(20).std().dropna().values
    for name, sim in models.items():
        rmses = []
        for i in range(min(200, sim.shape[0])):
            rs = pd.Series(sim[i]).rolling(20).std().dropna().values
            if len(rs) == len(rstd_actual):
                rmses.append(np.sqrt(np.mean((rs - rstd_actual) ** 2)))
        print(f"  {name:>10}: RMSE={np.mean(rmses):.2f}")

    # ── 3. Sojourn times above median ──
    print("\n=== Sojourn times above median ===")
    med = np.median(test_lmp)

    def sojourns(series):
        above = (series > med).astype(int)
        runs = []
        cur = 0
        for v in above:
            if v == 1:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur * 6)
                cur = 0
        if cur > 0:
            runs.append(cur * 6)
        return np.array(runs) if runs else np.array([0])

    soj_actual = sojourns(test_lmp)
    print(f"  Actual: mean={soj_actual.mean():.0f}h, median={np.median(soj_actual):.0f}h, "
          f"max={soj_actual.max():.0f}h, n={len(soj_actual)}")
    for name, sim in models.items():
        means, maxes = [], []
        for i in range(min(200, sim.shape[0])):
            s = sojourns(sim[i])
            means.append(s.mean())
            maxes.append(s.max())
        print(f"  {name:>10}: mean={np.mean(means):.0f}h, max={np.mean(maxes):.0f}h")

    # ── 4. Conditional coverage by persistence regime ──
    print("\n=== Conditional 90% coverage by dynamic regime ===")
    test_D = test["D"].values

    groups = [
        ("Fast (D0-D1)", [0, 1]),
        ("Moderate (D2-D4)", [2, 3, 4]),
        ("Persistent (D5-D7)", [5, 6, 7]),
    ]
    for label, d_vals in groups:
        mask = np.isin(test_D, d_vals)
        n_in = mask.sum()
        if n_in < 10:
            continue
        test_sub = test_lmp[mask]
        print(f"  {label} (n={n_in}):")
        for name, sim in models.items():
            p5 = np.percentile(sim, 5, axis=0)[mask]
            p95 = np.percentile(sim, 95, axis=0)[mask]
            cov = ((test_sub >= p5) & (test_sub <= p95)).mean()
            print(f"    {name:>10}: {cov * 100:.1f}%")

    # ── 5. Tail coverage (99%) ──
    print("\n=== 99% tail coverage ===")
    for name, sim in models.items():
        lo = np.percentile(sim, 0.5, axis=0)
        hi = np.percentile(sim, 99.5, axis=0)
        cov = ((test_lmp >= lo) & (test_lmp <= hi)).mean()
        print(f"  {name:>10}: {cov * 100:.1f}%")

    # ── 6. Price change distribution (captures dynamics, not levels) ──
    print("\n=== Price change (delta_p) distribution match ===")
    dp_actual = np.diff(test_lmp)
    for name, sim in models.items():
        wass_dp = []
        for i in range(min(200, sim.shape[0])):
            dp_sim = np.diff(sim[i])
            wass_dp.append(sp_stats.wasserstein_distance(dp_actual, dp_sim))
        print(f"  {name:>10}: Wasserstein(delta_p)={np.mean(wass_dp):.2f}")


if __name__ == "__main__":
    main()
