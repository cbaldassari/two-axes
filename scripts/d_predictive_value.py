"""
d_predictive_value.py
=====================
Three tests of D's predictive value:

1. DIRECTION: does D predict whether the price continues or reverts?
   If price is above mu(E), does high D → price stays above?

2. ADAPTIVE FORECAST: does knowing D help choose the better naive forecast?
   High D → use p_t (current price), Low D → use mu(E) (regime mean)
   Compare vs always using p_t, always using mu(E)

3. TIME TO REGIME CHANGE: does D predict how many steps until E changes?
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
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
    df["E_next"] = df["E"].shift(-1)
    df["move"] = df["lmp_next"] - df["lmp"]
    df = df.dropna(subset=["move", "E_next"]).copy()

    # D groups
    df["D_group"] = pd.cut(df["D"], bins=[-1, 1, 4, 7],
                           labels=["Fast", "Moderate", "Persistent"])

    # ═══════════════════════════════════════════════════════
    # TEST 1: Direction — does deviation predict next move, and does D modulate this?
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 1: Does D predict whether the price continues or reverts?")
    print("=" * 70)
    print("\nIf deviation > 0 (price above regime mean):")
    print("  Mean-reverting → next move should be NEGATIVE")
    print("  Persistent     → next move should be POSITIVE or near zero")

    # P(move has same sign as deviation)
    df["same_sign"] = (np.sign(df["deviation"]) == np.sign(df["move"])).astype(int)
    # Exclude near-zero deviations
    significant = df[df["deviation"].abs() > 5].copy()

    print(f"\n{'D group':<15} {'n':>6} {'P(continue)':>12} {'mean move when dev>0':>22} {'mean move when dev<0':>22}")
    print("-" * 80)
    for dg in ["Fast", "Moderate", "Persistent"]:
        sub = significant[significant["D_group"] == dg]
        p_continue = sub["same_sign"].mean()

        above = sub[sub["deviation"] > 0]["move"]
        below = sub[sub["deviation"] < 0]["move"]

        print(f"{dg:<15} {len(sub):>6} {p_continue:>12.1%} "
              f"{above.mean():>+20.2f}  {below.mean():>+20.2f}")

    # Correlation between deviation and next move, by D group
    print(f"\n{'D group':<15} {'corr(dev, move)':>16} {'p-value':>12}")
    print("-" * 45)
    for dg in ["Fast", "Moderate", "Persistent"]:
        sub = significant[significant["D_group"] == dg]
        r, p = sp_stats.pearsonr(sub["deviation"], sub["move"])
        print(f"{dg:<15} {r:>16.4f} {p:>12.2e}")

    # By individual D regime
    print(f"\n{'D':>3} {'n':>6} {'corr(dev,move)':>15} {'P(continue)':>12}")
    print("-" * 40)
    for d in range(n_D):
        sub = significant[significant["D"] == d]
        if len(sub) < 20:
            continue
        r, _ = sp_stats.pearsonr(sub["deviation"], sub["move"])
        pc = sub["same_sign"].mean()
        print(f"D{d}  {len(sub):>5} {r:>15.4f} {pc:>12.1%}")

    # ═══════════════════════════════════════════════════════
    # TEST 2: Adaptive forecast
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: Adaptive forecast — does D tell you which predictor to use?")
    print("=" * 70)

    # Three forecasts:
    # A) Always use p_t (naive random walk)
    # B) Always use mu(E) (regime mean)
    # C) Adaptive: use p_t when D >= 5, mu(E) when D <= 1, blend in between
    df["err_pt"] = (df["lmp_next"] - df["lmp"]).abs()
    df["err_mu"] = (df["lmp_next"] - df["mu_E"]).abs()

    # Adaptive: weight = D/7 (0=all mu, 1=all p_t)
    df["weight"] = df["D"] / 7.0
    df["forecast_adapt"] = df["weight"] * df["lmp"] + (1 - df["weight"]) * df["mu_E"]
    df["err_adapt"] = (df["lmp_next"] - df["forecast_adapt"]).abs()

    print(f"\n{'Forecast':<25} {'MAE overall':>12}")
    print("-" * 40)
    print(f"{'Always p_t (naive)':<25} {df['err_pt'].mean():>12.2f}")
    print(f"{'Always mu(E)':<25} {df['err_mu'].mean():>12.2f}")
    print(f"{'Adaptive (D-weighted)':<25} {df['err_adapt'].mean():>12.2f}")

    # By D group
    print(f"\n{'D group':<15} {'MAE p_t':>8} {'MAE mu(E)':>10} {'MAE adapt':>10} {'best':>8}")
    print("-" * 55)
    for dg in ["Fast", "Moderate", "Persistent"]:
        sub = df[df["D_group"] == dg]
        mae_pt = sub["err_pt"].mean()
        mae_mu = sub["err_mu"].mean()
        mae_ad = sub["err_adapt"].mean()
        best = "p_t" if mae_pt < mae_mu else "mu(E)"
        print(f"{dg:<15} {mae_pt:>8.2f} {mae_mu:>10.2f} {mae_ad:>10.2f} {best:>8}")

    # ═══════════════════════════════════════════════════════
    # TEST 3: Time to regime change
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: Does D predict how long until E changes?")
    print("=" * 70)

    # For each window, count steps until E changes
    E_vals = df["E"].values
    steps_to_change = np.zeros(len(E_vals))
    for i in range(len(E_vals)):
        j = i + 1
        while j < len(E_vals) and E_vals[j] == E_vals[i]:
            j += 1
        steps_to_change[i] = (j - i) * 6  # hours

    df["hours_to_change"] = steps_to_change

    print(f"\n{'D group':<15} {'n':>6} {'mean h to change':>18} {'median':>8}")
    print("-" * 50)
    for dg in ["Fast", "Moderate", "Persistent"]:
        sub = df[df["D_group"] == dg]
        print(f"{dg:<15} {len(sub):>6} {sub['hours_to_change'].mean():>18.0f} "
              f"{sub['hours_to_change'].median():>8.0f}")

    print(f"\n{'D':>3} {'n':>6} {'mean h':>8} {'median h':>9}")
    print("-" * 30)
    for d in range(n_D):
        sub = df[df["D"] == d]
        print(f"D{d}  {len(sub):>5} {sub['hours_to_change'].mean():>8.0f} {sub['hours_to_change'].median():>9.0f}")

    # Correlation
    r, p = sp_stats.pearsonr(df["D"], df["hours_to_change"])
    print(f"\nCorrelation D vs hours_to_change: r={r:.4f}, p={p:.2e}")


if __name__ == "__main__":
    main()
