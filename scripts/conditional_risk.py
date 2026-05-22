"""
conditional_risk.py
===================
Show that for the same economic regime, the dynamic regime changes the risk profile.
Computes: conditional price stats, next-window price change, and empirical VaR by (E,D).
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
    df = load()
    n = len(df)

    # Next-window price change
    df["lmp_next"] = df["lmp"].shift(-1)
    df["delta_p"] = df["lmp_next"] - df["lmp"]
    df = df.dropna(subset=["delta_p"])

    # Group D into 3 categories for clarity
    df["D_group"] = pd.cut(df["D"], bins=[-1, 1, 4, 7],
                           labels=["Fast (D0-D1)", "Moderate (D2-D4)", "Persistent (D5-D7)"])

    E_names = {
        0: "Off-peak", 1: "Low", 2: "Below-avg", 3: "Baseload",
        4: "Moderate", 5: "Demand", 6: "Stress", 7: "Winter-spike", 8: "Extreme-spike"
    }

    # ── 1. Same E, different D: price statistics ──
    print("=" * 80)
    print("CONDITIONAL RISK: same economic regime, different dynamic regime")
    print("=" * 80)

    records = []
    for e in range(9):
        for dg in ["Fast (D0-D1)", "Moderate (D2-D4)", "Persistent (D5-D7)"]:
            mask = (df["E"] == e) & (df["D_group"] == dg)
            sub = df[mask]
            nn = len(sub)
            if nn < 10:
                continue
            lmp = sub["lmp"].values
            dp = sub["delta_p"].values
            records.append({
                "E": e,
                "E_name": E_names[e],
                "D_group": dg,
                "n": nn,
                "lmp_mean": round(float(np.mean(lmp)), 1),
                "lmp_std": round(float(np.std(lmp)), 1),
                "dp_mean": round(float(np.mean(dp)), 2),
                "dp_std": round(float(np.std(dp)), 2),
                "VaR_95": round(float(np.percentile(dp, 5)), 2),  # 5th pct of delta_p = downside VaR
                "CVaR_95": round(float(np.mean(dp[dp <= np.percentile(dp, 5)])), 2) if (dp <= np.percentile(dp, 5)).sum() > 0 else 0,
                "prob_spike_up": round(float((dp > 20).mean()), 3),   # P(price jump > $20)
                "prob_spike_down": round(float((dp < -20).mean()), 3),
            })

    rdf = pd.DataFrame(records)
    rdf.to_csv(OUT_DIR / "conditional_risk.csv", index=False)

    # Print nicely
    print(f"\n{'E':>2} {'Name':<14} {'D group':<20} {'n':>5} {'LMP mu':>7} {'LMP sig':>7} "
          f"{'dp mu':>7} {'dp sig':>7} {'VaR95':>7} {'P(+20)':>7} {'P(-20)':>7}")
    print("-" * 110)
    for _, r in rdf.iterrows():
        print(f"{r['E']:>2} {r['E_name']:<14} {r['D_group']:<20} {r['n']:>5} "
              f"{r['lmp_mean']:>7.1f} {r['lmp_std']:>7.1f} "
              f"{r['dp_mean']:>7.2f} {r['dp_std']:>7.2f} "
              f"{r['VaR_95']:>7.2f} {r['prob_spike_up']:>7.3f} {r['prob_spike_down']:>7.3f}")

    # ── 2. Focus on key economic regimes: show the D effect ──
    print("\n" + "=" * 80)
    print("KEY INSIGHT: same price level, different risk")
    print("=" * 80)

    for e in [2, 5, 8]:  # Below-avg, Demand, Extreme-spike
        print(f"\n  E{e} ({E_names[e]}):")
        for dg in ["Fast (D0-D1)", "Moderate (D2-D4)", "Persistent (D5-D7)"]:
            sub_rdf = rdf[(rdf["E"] == e) & (rdf["D_group"] == dg)]
            if len(sub_rdf) == 0:
                continue
            r = sub_rdf.iloc[0]
            print(f"    {dg:<22} n={r['n']:>4}  LMP={r['lmp_mean']:>5.1f}+/-{r['lmp_std']:>5.1f}  "
                  f"dp={r['dp_mean']:>+6.2f}+/-{r['dp_std']:>5.2f}  "
                  f"VaR95={r['VaR_95']:>7.2f}  P(+$20)={r['prob_spike_up']:.1%}  P(-$20)={r['prob_spike_down']:.1%}")

    # ── 3. Figure: dp distribution for same E, different D ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    focus_regimes = [(2, "Below-avg ($40)"), (5, "Demand ($72)"), (8, "Extreme spike ($137)")]
    colors = {"Fast (D0-D1)": "#3498db", "Moderate (D2-D4)": "#95a5a6", "Persistent (D5-D7)": "#e74c3c"}

    for ax, (e, title) in zip(axes, focus_regimes):
        for dg in ["Fast (D0-D1)", "Moderate (D2-D4)", "Persistent (D5-D7)"]:
            mask = (df["E"] == e) & (df["D_group"] == dg)
            dp = df.loc[mask, "delta_p"].values
            if len(dp) < 10:
                continue
            # Clip for visualization
            dp_clip = np.clip(dp, -80, 80)
            ax.hist(dp_clip, bins=40, density=True, alpha=0.5, color=colors[dg],
                    label=f"{dg} (n={len(dp)})")
        ax.set_xlabel("Price change ($/MWh)")
        ax.set_title(f"E{e}: {title}")
        ax.axvline(0, color="black", ls="--", lw=0.5)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(-80, 80)

    axes[0].set_ylabel("Density")
    plt.suptitle("Same economic regime, different risk profile", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_conditional_risk.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved fig_conditional_risk.png")

    # ── 4. Summary stat: how much does D change VaR within same E? ──
    print("\n" + "=" * 80)
    print("VaR SPREAD: max VaR - min VaR across D groups, within each E")
    print("=" * 80)
    for e in range(9):
        sub = rdf[rdf["E"] == e]
        if len(sub) >= 2:
            var_range = sub["VaR_95"].max() - sub["VaR_95"].min()
            std_range = sub["dp_std"].max() - sub["dp_std"].min()
            print(f"  E{e} ({E_names[e]:<14}): VaR range = {var_range:>6.1f} $/MWh, "
                  f"dp_std range = {std_range:>5.1f} $/MWh")

    print("\nDone.")


if __name__ == "__main__":
    main()
