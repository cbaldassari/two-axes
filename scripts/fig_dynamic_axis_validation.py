"""
fig_dynamic_axis_validation.py
==============================
D tells you: "can I trust today's price as a forecast for the next hours?"

Compute the naive forecast error |p_{t+k} - p_t| at horizons 6h, 12h, 24h,
grouped by D regime. If D is valid:
  - High D → low error (price stays where it is)
  - Low D  → high error (price moves away)

Also show that E does NOT predict this — the forecast error is similar
across E regimes (once you control for D).
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
STRIDE_H = 6


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

    # Compute naive forecast errors at different horizons
    # Each window is 6h apart, so shift(1)=6h, shift(2)=12h, shift(4)=24h
    horizons = {"6h": 1, "12h": 2, "24h": 4}
    for label, shift in horizons.items():
        df[f"err_{label}"] = (df["lmp"].shift(-shift) - df["lmp"]).abs()

    df = df.dropna(subset=["err_24h"])

    # --- Print results ---
    print("Mean absolute naive forecast error ($/MWh) by D regime:\n")
    print(f"{'D':>3} {'ACF_6h':>7} {'n':>6} {'err 6h':>8} {'err 12h':>8} {'err 24h':>8}")
    print("-" * 45)
    for d in range(n_D):
        sub = df[df["D"] == d]
        acf_m = sub["acf_6h"].mean()
        print(f"D{d}  {acf_m:>6.2f} {len(sub):>6} "
              f"{sub['err_6h'].mean():>8.1f} {sub['err_12h'].mean():>8.1f} {sub['err_24h'].mean():>8.1f}")

    print("\n\nSame metric by E regime (should be noisier / less ordered):\n")
    print(f"{'E':>3} {'n':>6} {'err 6h':>8} {'err 12h':>8} {'err 24h':>8}")
    print("-" * 35)
    for e in range(n_E):
        sub = df[df["E"] == e]
        print(f"E{e}  {len(sub):>6} "
              f"{sub['err_6h'].mean():>8.1f} {sub['err_12h'].mean():>8.1f} {sub['err_24h'].mean():>8.1f}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (label, shift) in enumerate(horizons.items()):
        ax = axes[i]
        col = f"err_{label}"

        # By D
        d_vals = [df[df["D"] == d][col].mean() for d in range(n_D)]
        d_colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, n_D))
        bars_d = ax.bar(np.arange(n_D) - 0.17, d_vals, 0.3, color=d_colors,
                        edgecolor="white", label="By D regime")

        # By E
        e_vals = [df[df["E"] == e][col].mean() for e in range(n_E)]
        bars_e = ax.bar(np.arange(n_E) + 0.17, e_vals, 0.3, color="#cccccc",
                        edgecolor="white", label="By E regime")

        ax.set_xticks(range(max(n_E, n_D)))
        ax.set_xticklabels([str(i) for i in range(max(n_E, n_D))], fontsize=9)
        ax.set_xlabel("Regime index", fontsize=11)
        if i == 0:
            ax.set_ylabel("Mean absolute error ($/MWh)", fontsize=11)
        ax.set_title(f"Forecast horizon: {label}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left" if i < 2 else "upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        '"Can I trust today\'s price as a forecast?"\n'
        'D tells you (colored bars decrease). E does not (gray bars are flat).',
        fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_dynamic_axis_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved fig_dynamic_axis_validation.png")


if __name__ == "__main__":
    main()
