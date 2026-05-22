"""
residual_by_D.py
================
After removing the price level explained by E (residual = p - mu(E)),
does D explain the structure of what remains?

4 moments of residuals by D regime, on ALL data.
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


def main():
    df = load().reset_index(drop=True)

    # Compute residuals: remove E-level mean
    mu_E = df.groupby("E")["lmp"].mean()
    df["mu_E"] = df["E"].map(mu_E)
    df["resid"] = df["lmp"] - df["mu_E"]

    # Also compute next-step residual change
    df["resid_next"] = df["resid"].shift(-1)
    df["delta_r"] = df["resid_next"] - df["resid"]

    print(f"Total windows: {len(df)}")
    print(f"Residual (p - mu_E): mean={df['resid'].mean():.2f}, std={df['resid'].std():.1f}")

    # ── 4 moments of residuals by D ──
    print("\n" + "=" * 85)
    print("RESIDUALS (p - mu_E) by D regime — 4 moments")
    print("=" * 85)
    print(f"{'D':>3} {'n':>6} {'mean':>8} {'std':>8} {'skew':>8} {'kurt':>8} {'ACF_6h':>8}")
    print("-" * 55)

    records = []
    for d in range(n_D):
        sub = df[df["D"] == d]
        r = sub["resid"].values
        rec = {
            "D": d,
            "n": len(sub),
            "mean": float(np.mean(r)),
            "std": float(np.std(r)),
            "skew": float(sp_stats.skew(r)),
            "kurt": float(sp_stats.kurtosis(r)),
            "acf_6h": float(sub["acf_6h"].mean()),
        }
        records.append(rec)
        print(f"D{d}  {rec['n']:>5} {rec['mean']:>8.1f} {rec['std']:>8.1f} "
              f"{rec['skew']:>8.2f} {rec['kurt']:>8.2f} {rec['acf_6h']:>8.3f}")

    # ── 4 moments of delta_r by D ──
    df_clean = df.dropna(subset=["delta_r"])
    print("\n" + "=" * 85)
    print("RESIDUAL CHANGES (delta_r) by D regime — 4 moments")
    print("=" * 85)
    print(f"{'D':>3} {'n':>6} {'mean':>8} {'std':>8} {'skew':>8} {'kurt':>8}")
    print("-" * 45)

    records_dr = []
    for d in range(n_D):
        sub = df_clean[df_clean["D"] == d]
        dr = sub["delta_r"].values
        rec = {
            "D": d,
            "n": len(sub),
            "mean": float(np.mean(dr)),
            "std": float(np.std(dr)),
            "skew": float(sp_stats.skew(dr)),
            "kurt": float(sp_stats.kurtosis(dr)),
        }
        records_dr.append(rec)
        print(f"D{d}  {rec['n']:>5} {rec['mean']:>8.2f} {rec['std']:>8.1f} "
              f"{rec['skew']:>8.2f} {rec['kurt']:>8.2f}")

    # ── For comparison: same but by E (should be flatter) ──
    print("\n" + "=" * 85)
    print("RESIDUAL CHANGES (delta_r) by E regime — for comparison")
    print("=" * 85)
    print(f"{'E':>3} {'n':>6} {'mean':>8} {'std':>8} {'skew':>8} {'kurt':>8}")
    print("-" * 45)
    for e in range(n_E):
        sub = df_clean[df_clean["E"] == e]
        dr = sub["delta_r"].values
        print(f"E{e}  {len(sub):>5} {np.mean(dr):>8.2f} {np.std(dr):>8.1f} "
              f"{sp_stats.skew(dr):>8.2f} {sp_stats.kurtosis(dr):>8.2f}")

    # ── Eta-squared: how much variance of delta_r does D explain vs E? ──
    print("\n" + "=" * 85)
    print("ETA-SQUARED: fraction of delta_r variance explained")
    print("=" * 85)
    ss_total = np.sum((df_clean["delta_r"] - df_clean["delta_r"].mean()) ** 2)

    for col, label in [("D", "D regimes"), ("E", "E regimes")]:
        group_means = df_clean.groupby(col)["delta_r"].transform("mean")
        ss_between = np.sum((group_means - df_clean["delta_r"].mean()) ** 2)
        eta2 = ss_between / ss_total
        print(f"  {label}: eta^2 = {eta2:.4f}")

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors_d = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, n_D))

    # (a) std of residuals by D
    ax = axes[0, 0]
    stds = [records[d]["std"] for d in range(n_D)]
    ax.bar(range(n_D), stds, color=colors_d, edgecolor="white")
    ax.set_xticks(range(n_D))
    ax.set_xticklabels([f"D{d}" for d in range(n_D)])
    ax.set_ylabel("Std of residuals ($/MWh)")
    ax.set_title("(a) Residual spread by D", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for d, v in enumerate(stds):
        ax.text(d, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)

    # (b) std of delta_r by D
    ax = axes[0, 1]
    stds_dr = [records_dr[d]["std"] for d in range(n_D)]
    ax.bar(range(n_D), stds_dr, color=colors_d, edgecolor="white")
    ax.set_xticks(range(n_D))
    ax.set_xticklabels([f"D{d}" for d in range(n_D)])
    ax.set_ylabel("Std of residual changes ($/MWh)")
    ax.set_title("(b) Residual volatility by D", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for d, v in enumerate(stds_dr):
        ax.text(d, v + 0.3, f"{v:.0f}", ha="center", fontsize=9)

    # (c) kurtosis of delta_r by D
    ax = axes[1, 0]
    kurts = [records_dr[d]["kurt"] for d in range(n_D)]
    ax.bar(range(n_D), kurts, color=colors_d, edgecolor="white")
    ax.set_xticks(range(n_D))
    ax.set_xticklabels([f"D{d}" for d in range(n_D)])
    ax.set_ylabel("Kurtosis of residual changes")
    ax.set_title("(c) Tail heaviness by D", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for d, v in enumerate(kurts):
        ax.text(d, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)

    # (d) Distribution of delta_r for D0 vs D7
    ax = axes[1, 1]
    dr_d0 = df_clean[df_clean["D"] == 0]["delta_r"].values
    dr_d7 = df_clean[df_clean["D"] == 7]["delta_r"].values
    bins = np.linspace(-100, 100, 60)
    ax.hist(dr_d0, bins=bins, density=True, alpha=0.6, color=colors_d[0], label=f"D0 (n={len(dr_d0)})")
    ax.hist(dr_d7, bins=bins, density=True, alpha=0.6, color=colors_d[7], label=f"D7 (n={len(dr_d7)})")
    ax.set_xlabel("Residual change ($/MWh)")
    ax.set_ylabel("Density")
    ax.set_title("(d) D0 vs D7: residual change distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("After removing price level (E): what does D explain in the residuals?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_residual_by_D.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved fig_residual_by_D.png")


if __name__ == "__main__":
    main()
