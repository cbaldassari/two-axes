"""
residual_acf_test.py
====================
Information Leakage Test: do AR(1) residuals retain autocorrelation
when φ is estimated from the wrong axis?

Model 1 (single axis): φ(E) — same φ for all windows in each E
Model 2 (dual axis):   φ(E,D) — different φ for each (E,D) cell

If Model 1 uses the wrong φ, its residuals within each E will show
residual autocorrelation (Ljung-Box rejects H0 of white noise).
If Model 2 uses the correct φ, its residuals should be closer to i.i.d.
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
OUT_DIR = RESULTS_DIR / "dual_axis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
n_E, n_D = 9, 8
MAX_LAG = 12


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


def estimate_phi(sub, mu):
    lmp = sub["lmp"].values
    idx_s = sub.index.values
    consec = np.where(np.diff(idx_s) == 1)[0]
    if len(consec) > 5:
        x_t = lmp[consec] - mu
        x_t1 = lmp[consec + 1] - mu
        return float(np.clip((x_t * x_t1).sum() / max((x_t ** 2).sum(), 1e-12), -0.99, 0.999))
    return 0.5


def compute_residuals(df, mu_E, phi_E, phi_ED):
    """Compute AR(1) residuals for both models on consecutive pairs."""
    idx = df.index.values
    resid_1ax = []  # single axis: φ(E)
    resid_2ax = []  # dual axis: φ(E,D)
    e_list = []
    d_list = []

    for i in range(len(idx) - 1):
        if idx[i + 1] - idx[i] != 1:
            continue
        p_prev = df.loc[idx[i], "lmp"]
        p_curr = df.loc[idx[i + 1], "lmp"]
        e = df.loc[idx[i + 1], "E"]
        d = df.loc[idx[i + 1], "D"]
        mu = mu_E[e]

        r1 = p_curr - mu - phi_E[e] * (p_prev - mu)
        r2 = p_curr - mu - phi_ED.get((e, d), phi_E[e]) * (p_prev - mu)

        resid_1ax.append(r1)
        resid_2ax.append(r2)
        e_list.append(e)
        d_list.append(d)

    return (np.array(resid_1ax), np.array(resid_2ax),
            np.array(e_list), np.array(d_list))


def main():
    df = load().reset_index(drop=True)

    # Estimate parameters
    mu_E = {e: float(df[df["E"] == e]["lmp"].mean()) for e in range(n_E)}

    phi_E = {}
    for e in range(n_E):
        phi_E[e] = estimate_phi(df[df["E"] == e], mu_E[e])

    phi_ED = {}
    for e in range(n_E):
        for d in range(n_D):
            sub = df[(df["E"] == e) & (df["D"] == d)]
            if len(sub) >= 10:
                phi_ED[(e, d)] = estimate_phi(sub, mu_E[e])

    # Compute residuals
    resid_1, resid_2, e_arr, d_arr = compute_residuals(df, mu_E, phi_E, phi_ED)

    E_names = {0: "Off-peak", 1: "Low", 2: "Below-avg", 3: "Baseload",
               4: "Moderate", 5: "Demand", 6: "Stress", 7: "W-spike", 8: "Ext-spike"}

    # ── Analysis per E regime ──
    print("=" * 75)
    print("LJUNG-BOX TEST sui residui condizionati (H0: rumore bianco)")
    print("=" * 75)
    print(f"\n{'E':<20} {'n':>5} | {'LB(E) stat':>10} {'p':>8} {'sig':>4} | "
          f"{'LB(E,D) stat':>12} {'p':>8} {'sig':>4}")
    print("-" * 80)

    results = []
    for e in range(n_E):
        mask = e_arr == e
        if mask.sum() < 50:
            continue
        r1 = resid_1[mask]
        r2 = resid_2[mask]

        # Ljung-Box test at MAX_LAG
        lb1 = acorr_ljungbox(r1, lags=MAX_LAG, return_df=True)
        lb2 = acorr_ljungbox(r2, lags=MAX_LAG, return_df=True)

        # Take the test at the max lag
        stat1 = lb1["lb_stat"].iloc[-1]
        p1 = lb1["lb_pvalue"].iloc[-1]
        stat2 = lb2["lb_stat"].iloc[-1]
        p2 = lb2["lb_pvalue"].iloc[-1]

        sig1 = "***" if p1 < 0.001 else "**" if p1 < 0.01 else "*" if p1 < 0.05 else ""
        sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else ""

        results.append({"E": e, "n": mask.sum(),
                        "LB1": stat1, "p1": p1, "sig1": sig1,
                        "LB2": stat2, "p2": p2, "sig2": sig2})

        print(f"E{e} ({E_names[e]:<12}) {mask.sum():>4} | {stat1:>10.1f} {p1:>8.4f} {sig1:>4} | "
              f"{stat2:>12.1f} {p2:>8.4f} {sig2:>4}")

    # Count
    n_sig1 = sum(1 for r in results if r["p1"] < 0.05)
    n_sig2 = sum(1 for r in results if r["p2"] < 0.05)
    print(f"\nRifiuto H0 (autocorrelazione residua):")
    print(f"  Modello phi(E):   {n_sig1}/{len(results)} regimi")
    print(f"  Modello phi(E,D): {n_sig2}/{len(results)} regimi")

    # ── ACF comparison for key regimes ──
    focus_regimes = [2, 5, 7]
    fig, axes = plt.subplots(len(focus_regimes), 2, figsize=(12, 3.5 * len(focus_regimes)),
                             sharey=True)

    for row, e in enumerate(focus_regimes):
        mask = e_arr == e
        r1 = resid_1[mask]
        r2 = resid_2[mask]
        n_e = mask.sum()

        conf = 1.96 / np.sqrt(n_e)

        for col, (resid, title, color) in enumerate([
            (r1, f"$\\varphi(E)$ — singolo asse", "#e74c3c"),
            (r2, f"$\\varphi(E,D)$ — due assi", "#2ecc71"),
        ]):
            ax = axes[row, col]
            acf_vals = acf(resid, nlags=MAX_LAG, fft=True)
            lags = np.arange(MAX_LAG + 1)

            ax.bar(lags[1:], acf_vals[1:], color=color, alpha=0.7, width=0.6)
            ax.axhline(0, color="black", lw=0.5)
            ax.axhline(conf, color="gray", ls="--", lw=1)
            ax.axhline(-conf, color="gray", ls="--", lw=1)
            ax.set_xlim(0.5, MAX_LAG + 0.5)
            ax.set_ylim(-0.15, 0.25)

            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if row == len(focus_regimes) - 1:
                ax.set_xlabel("Lag (finestre, 6h ciascuna)")
            if col == 0:
                ax.set_ylabel(f"$E_{e}$ ({E_names[e]})\nACF", fontsize=10)

            # Ljung-Box annotation
            lb = acorr_ljungbox(resid, lags=MAX_LAG, return_df=True)
            p_val = lb["lb_pvalue"].iloc[-1]
            ax.text(0.95, 0.95, f"LB p = {p_val:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle("Autocorrelazione dei residui condizionati:\nsingolo asse vs. due assi",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_residual_acf.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved fig_residual_acf.png")

    # ── Global summary ──
    print(f"\n{'=' * 75}")
    print("RIEPILOGO GLOBALE")
    print(f"{'=' * 75}")
    print(f"Residui phi(E):   std={np.std(resid_1):.2f}, "
          f"ACF(1)={acf(resid_1, nlags=1, fft=True)[1]:.4f}")
    print(f"Residui phi(E,D): std={np.std(resid_2):.2f}, "
          f"ACF(1)={acf(resid_2, nlags=1, fft=True)[1]:.4f}")

    lb_all_1 = acorr_ljungbox(resid_1, lags=MAX_LAG, return_df=True)
    lb_all_2 = acorr_ljungbox(resid_2, lags=MAX_LAG, return_df=True)
    print(f"Ljung-Box globale phi(E):   stat={lb_all_1['lb_stat'].iloc[-1]:.1f}, "
          f"p={lb_all_1['lb_pvalue'].iloc[-1]:.2e}")
    print(f"Ljung-Box globale phi(E,D): stat={lb_all_2['lb_stat'].iloc[-1]:.1f}, "
          f"p={lb_all_2['lb_pvalue'].iloc[-1]:.2e}")


if __name__ == "__main__":
    main()
