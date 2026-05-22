"""
fig_d_predictive.py
===================
Three panels showing D's predictive value:

(a) Mean-reversion strength: corr(deviation, next_move) by D
    → D modulates how strongly the price reverts to regime mean

(b) Adaptive forecast: MAE of p_t vs mu(E) vs D-weighted blend, by D
    → D tells you which predictor to trust

(c) P(price continues) by D
    → D predicts whether the current trend continues or reverses
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
    mu_E = df.groupby("E")["lmp"].mean()
    df["mu_E"] = df["E"].map(mu_E)
    df["deviation"] = df["lmp"] - df["mu_E"]
    df["lmp_next"] = df["lmp"].shift(-1)
    df["move"] = df["lmp_next"] - df["lmp"]
    df = df.dropna(subset=["move"]).copy()

    # Errors
    df["err_pt"] = (df["lmp_next"] - df["lmp"]).abs()
    df["err_mu"] = (df["lmp_next"] - df["mu_E"]).abs()
    df["weight"] = df["D"] / 7.0
    df["forecast_adapt"] = df["weight"] * df["lmp"] + (1 - df["weight"]) * df["mu_E"]
    df["err_adapt"] = (df["lmp_next"] - df["forecast_adapt"]).abs()

    # Significant deviations for direction test
    sig = df[df["deviation"].abs() > 5].copy()
    sig["same_sign"] = (np.sign(sig["deviation"]) == np.sign(sig["move"])).astype(int)

    colors_d = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, n_D))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # ── (a) Mean-reversion strength by D ──
    ax = axes[0]
    corrs = []
    for d in range(n_D):
        sub = sig[sig["D"] == d]
        if len(sub) >= 20:
            r, _ = sp_stats.pearsonr(sub["deviation"], sub["move"])
        else:
            r = np.nan
        corrs.append(r)

    bars = ax.bar(range(n_D), [-c for c in corrs], color=colors_d, edgecolor="white")
    ax.set_xticks(range(n_D))
    ax.set_xticklabels([f"D{d}" for d in range(n_D)])
    ax.set_ylabel("Mean-reversion strength\n|corr(deviation, next move)|")
    ax.set_xlabel("Dynamic regime")
    ax.set_title("(a) D modulates mean-reversion", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for d, c in enumerate(corrs):
        if not np.isnan(c):
            ax.text(d, -c + 0.01, f"{-c:.2f}", ha="center", va="bottom", fontsize=9)

    # Annotate
    ax.annotate("D0: strong reversion\n(price snaps back)",
                xy=(0, 0.74), xytext=(2.5, 0.78),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("D7: weak reversion\n(price drifts)",
                xy=(7, 0.37), xytext=(5.5, 0.25),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="#333"))

    # ── (b) Adaptive forecast MAE by D ──
    ax = axes[1]
    mae_pt_d = [df[df["D"] == d]["err_pt"].mean() for d in range(n_D)]
    mae_mu_d = [df[df["D"] == d]["err_mu"].mean() for d in range(n_D)]
    mae_ad_d = [df[df["D"] == d]["err_adapt"].mean() for d in range(n_D)]

    x = np.arange(n_D)
    w = 0.25
    ax.bar(x - w, mae_mu_d, w, color="#2ecc71", edgecolor="white", label="Use regime mean μ(E)")
    ax.bar(x, mae_pt_d, w, color="#3498db", edgecolor="white", label="Use current price p_t")
    ax.bar(x + w, mae_ad_d, w, color="#e67e22", edgecolor="white", label="Adaptive (D-weighted)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}" for d in range(n_D)])
    ax.set_ylabel("MAE ($/MWh)")
    ax.set_xlabel("Dynamic regime")
    ax.set_title("(b) Which forecast to trust?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Mark crossover
    # Find where p_t becomes better than mu(E)
    for d in range(n_D):
        if mae_pt_d[d] < mae_mu_d[d] and d > 0 and mae_pt_d[d-1] >= mae_mu_d[d-1]:
            ax.axvline(d - 0.5, color="black", ls=":", lw=1)
            ax.text(d - 0.5, ax.get_ylim()[1] * 0.95, "← μ(E) wins | p_t wins →",
                    ha="center", va="top", fontsize=8, style="italic")

    # ── (c) P(continue) by D ──
    ax = axes[2]
    p_cont = []
    ns = []
    for d in range(n_D):
        sub = sig[sig["D"] == d]
        if len(sub) >= 20:
            p_cont.append(sub["same_sign"].mean() * 100)
            ns.append(len(sub))
        else:
            p_cont.append(np.nan)
            ns.append(0)

    bars = ax.bar(range(n_D), p_cont, color=colors_d, edgecolor="white")
    ax.axhline(50, color="black", ls="--", lw=1, label="Random (50%)")
    ax.set_xticks(range(n_D))
    ax.set_xticklabels([f"D{d}" for d in range(n_D)])
    ax.set_ylabel("P(price continues direction) %")
    ax.set_xlabel("Dynamic regime")
    ax.set_title("(c) Will the trend continue?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 55)
    for d, (p, n) in enumerate(zip(p_cont, ns)):
        if not np.isnan(p):
            ax.text(d, p + 0.8, f"{p:.0f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("What D predicts: mean-reversion speed, forecast strategy, trend persistence",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_d_predictive.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary stats
    print("Overall MAE comparison:")
    print(f"  Always p_t:     {df['err_pt'].mean():.2f} $/MWh")
    print(f"  Always mu(E):   {df['err_mu'].mean():.2f} $/MWh")
    print(f"  Adaptive (D):   {df['err_adapt'].mean():.2f} $/MWh")
    print(f"  Improvement:    {(1 - df['err_adapt'].mean()/df['err_pt'].mean())*100:.1f}% vs p_t")
    print(f"                  {(1 - df['err_adapt'].mean()/df['err_mu'].mean())*100:.1f}% vs mu(E)")
    print(f"\nCorrelation gradient: D0={corrs[0]:.3f} → D7={corrs[7]:.3f}")
    print(f"P(continue) gradient: D0={p_cont[0]:.0f}% → D7={p_cont[7]:.0f}%")

    print("\nSaved fig_d_predictive.png")


if __name__ == "__main__":
    main()
