"""
d_predictive_v2.py
==================
Two new tests for D's predictive value:

1. MULTI-HORIZON: does D's adaptive forecast advantage grow at longer horizons?
   Test at 6h, 12h, 24h, 48h, 72h, 1 week

2. TRANSITION DIRECTION: does D predict whether E goes UP or DOWN?
   P(E increases | D) vs P(E decreases | D)
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

    # ═══════════════════════════════════════════════════════
    # TEST 1: Multi-horizon adaptive forecast
    # ═══════════════════════════════════════════════════════
    print("=" * 75)
    print("TEST 1: Adaptive forecast at multiple horizons")
    print("=" * 75)

    horizons = {
        "6h": 1, "12h": 2, "24h": 4, "48h": 8, "72h": 12, "1wk": 28
    }

    print(f"\n{'Horizon':>8} {'MAE p_t':>8} {'MAE mu(E)':>10} {'MAE adapt':>10} "
          f"{'improv vs best':>15}")
    print("-" * 55)

    for label, shift in horizons.items():
        future = df["lmp"].shift(-shift)
        valid = ~future.isna()
        lmp_now = df.loc[valid, "lmp"].values
        lmp_fut = future[valid].values
        mu_e = df.loc[valid, "mu_E"].values
        d_vals = df.loc[valid, "D"].values

        err_pt = np.abs(lmp_fut - lmp_now)
        err_mu = np.abs(lmp_fut - mu_e)

        weight = d_vals / 7.0
        forecast_adapt = weight * lmp_now + (1 - weight) * mu_e
        err_adapt = np.abs(lmp_fut - forecast_adapt)

        mae_pt = err_pt.mean()
        mae_mu = err_mu.mean()
        mae_ad = err_adapt.mean()
        best = min(mae_pt, mae_mu)
        improv = (1 - mae_ad / best) * 100

        print(f"{label:>8} {mae_pt:>8.2f} {mae_mu:>10.2f} {mae_ad:>10.2f} {improv:>+14.1f}%")

    # Breakdown by D group at key horizons
    print("\n--- Breakdown by D group ---")
    for label, shift in [("6h", 1), ("48h", 8), ("1wk", 28)]:
        future = df["lmp"].shift(-shift)
        valid = ~future.isna()
        tmp = df[valid].copy()
        tmp["lmp_fut"] = future[valid].values
        tmp["err_pt"] = (tmp["lmp_fut"] - tmp["lmp"]).abs()
        tmp["err_mu"] = (tmp["lmp_fut"] - tmp["mu_E"]).abs()
        tmp["D_group"] = pd.cut(tmp["D"], bins=[-1, 1, 4, 7],
                                labels=["Fast", "Moderate", "Persistent"])

        print(f"\n  Horizon: {label}")
        print(f"  {'D group':<15} {'MAE p_t':>8} {'MAE mu(E)':>10} {'winner':>8} {'gap':>8}")
        print(f"  {'-'*50}")
        for dg in ["Fast", "Moderate", "Persistent"]:
            sub = tmp[tmp["D_group"] == dg]
            mpt = sub["err_pt"].mean()
            mmu = sub["err_mu"].mean()
            winner = "p_t" if mpt < mmu else "mu(E)"
            gap = abs(mpt - mmu)
            print(f"  {dg:<15} {mpt:>8.2f} {mmu:>10.2f} {winner:>8} {gap:>8.2f}")

    # ═══════════════════════════════════════════════════════
    # TEST 2: Transition direction
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("TEST 2: Does D predict the direction of E transitions?")
    print("=" * 75)

    # For each window where E changes, record direction and D at departure
    E_vals = df["E"].values
    D_vals = df["D"].values

    transitions = []
    for i in range(len(E_vals) - 1):
        if E_vals[i] != E_vals[i + 1]:
            transitions.append({
                "E_from": E_vals[i],
                "E_to": E_vals[i + 1],
                "D": D_vals[i],
                "direction": "up" if E_vals[i + 1] > E_vals[i] else "down",
                "step": abs(E_vals[i + 1] - E_vals[i]),
            })

    tr = pd.DataFrame(transitions)
    tr["D_group"] = pd.cut(tr["D"], bins=[-1, 1, 4, 7],
                           labels=["Fast", "Moderate", "Persistent"])

    print(f"\nTotal E transitions: {len(tr)}")
    print(f"\n{'D group':<15} {'n trans':>8} {'P(up)':>7} {'P(down)':>8} {'mean step':>10}")
    print("-" * 50)
    for dg in ["Fast", "Moderate", "Persistent"]:
        sub = tr[tr["D_group"] == dg]
        n = len(sub)
        p_up = (sub["direction"] == "up").mean()
        p_down = (sub["direction"] == "down").mean()
        mean_step = sub["step"].mean()
        print(f"{dg:<15} {n:>8} {p_up:>7.1%} {p_down:>8.1%} {mean_step:>10.2f}")

    # By individual D
    print(f"\n{'D':>3} {'n':>5} {'P(up)':>7} {'P(down)':>8} {'mean step':>10}")
    print("-" * 35)
    for d in range(n_D):
        sub = tr[tr["D"] == d]
        if len(sub) < 10:
            continue
        p_up = (sub["direction"] == "up").mean()
        mean_step = sub["step"].mean()
        print(f"D{d}  {len(sub):>4} {p_up:>7.1%} {1-p_up:>8.1%} {mean_step:>10.2f}")

    # Chi-squared test: is direction independent of D_group?
    ct = pd.crosstab(tr["D_group"], tr["direction"])
    chi2, p_chi, _, _ = sp_stats.chi2_contingency(ct)
    print(f"\nChi-squared test (direction vs D_group): chi2={chi2:.2f}, p={p_chi:.4f}")

    # Conditional on current E level (high vs low)
    print("\n--- Transition direction by D, conditional on E level ---")
    for e_label, e_range in [("Low E (0-3)", range(4)), ("High E (4-8)", range(4, 9))]:
        sub_tr = tr[tr["E_from"].isin(e_range)]
        print(f"\n  {e_label} (n={len(sub_tr)}):")
        for dg in ["Fast", "Moderate", "Persistent"]:
            sub = sub_tr[sub_tr["D_group"] == dg]
            if len(sub) < 5:
                continue
            p_up = (sub["direction"] == "up").mean()
            print(f"    {dg:<15} n={len(sub):>3}  P(up)={p_up:.1%}  P(down)={1-p_up:.1%}")


if __name__ == "__main__":
    main()
