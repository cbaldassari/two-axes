"""
transition_coupling.py
======================
Show that P(stay in same E) depends on D.
If the two axes were dynamically independent, the self-persistence
probability of E would be the same regardless of D. It is not.
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

    E_names = {
        0: "Off-peak", 1: "Low", 2: "Below-avg", 3: "Baseload",
        4: "Moderate", 5: "Demand", 6: "Stress", 7: "Winter-spike", 8: "Extreme-spike"
    }

    # Compute P(E_{t+1} = E_t | E_t = e, D_t = d) from consecutive windows
    grid = np.full((n_E, n_D), np.nan)
    counts = np.zeros((n_E, n_D), dtype=int)

    for i in range(len(df) - 1):
        e_now = df.loc[i, "E"]
        d_now = df.loc[i, "D"]
        e_next = df.loc[i + 1, "E"]
        stayed = int(e_now == e_next)
        counts[e_now, d_now] += 1
        if np.isnan(grid[e_now, d_now]):
            grid[e_now, d_now] = stayed
        else:
            # running sum, will divide later
            grid[e_now, d_now] += stayed

    # Divide to get probabilities
    for e in range(n_E):
        for d in range(n_D):
            if counts[e, d] > 0:
                grid[e, d] /= counts[e, d]

    # Print table
    print("P(stay in same E) by (E, D):")
    print(f"{'':>16}", end="")
    for d in range(n_D):
        print(f"  D{d:>1}", end="")
    print("   | marginal")
    print("-" * 80)
    for e in range(n_E):
        print(f"E{e} {E_names[e]:<13}", end="")
        vals = []
        for d in range(n_D):
            if counts[e, d] >= 10:
                print(f" {grid[e, d]:.2f}", end="")
                vals.append(grid[e, d])
            else:
                print(f"    -", end="")
        if vals:
            print(f"   | {np.mean(vals):.2f}  (range {max(vals)-min(vals):.2f})")
        else:
            print()

    # Key examples
    print("\n--- Key examples ---")
    for e in [5, 7, 8]:
        d_fast = grid[e, 0] if counts[e, 0] >= 10 else grid[e, 1]
        d_slow = grid[e, 7] if counts[e, 7] >= 10 else grid[e, 6]
        n_fast = counts[e, 0] if counts[e, 0] >= 10 else counts[e, 1]
        n_slow = counts[e, 7] if counts[e, 7] >= 10 else counts[e, 6]
        if not np.isnan(d_fast) and not np.isnan(d_slow):
            print(f"  E{e} ({E_names[e]}): P(stay|D fast)={d_fast:.0%} vs P(stay|D slow)={d_slow:.0%}")

    # --- Figure: heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): heatmap
    ax = axes[0]
    masked = np.ma.masked_where(counts < 10, grid)
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0,
                   origin="upper")
    ax.set_xticks(range(n_D))
    ax.set_xticklabels([f"D{d}" for d in range(n_D)])
    ax.set_yticks(range(n_E))
    ax.set_yticklabels([f"E{e} {E_names[e]}" for e in range(n_E)], fontsize=9)
    ax.set_xlabel("Dynamic regime (D) — persistence increases →")
    ax.set_ylabel("Economic regime (E)")
    ax.set_title("(a) P(stay in same economic regime)")

    for e in range(n_E):
        for d in range(n_D):
            if counts[e, d] >= 10:
                ax.text(d, e, f"{grid[e, d]:.0%}", ha="center", va="center",
                        fontsize=8, color="black" if grid[e, d] > 0.7 else "white")
            else:
                ax.text(d, e, "-", ha="center", va="center", fontsize=8, color="gray")

    plt.colorbar(im, ax=ax, label="P(stay)", shrink=0.8)

    # Panel (b): line plot for selected E regimes
    ax = axes[1]
    focus = [2, 5, 7, 8]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
    for e, color in zip(focus, colors):
        ds, ps = [], []
        for d in range(n_D):
            if counts[e, d] >= 10:
                ds.append(d)
                ps.append(grid[e, d])
        if ds:
            ax.plot(ds, ps, "o-", color=color, label=f"E{e} ({E_names[e]})", lw=2, markersize=6)

    # Add Kronecker baseline (marginal P(stay in E), independent of D)
    for e, color in zip(focus, colors):
        total_stay = sum(grid[e, d] * counts[e, d] for d in range(n_D) if counts[e, d] >= 10)
        total_n = sum(counts[e, d] for d in range(n_D) if counts[e, d] >= 10)
        if total_n > 0:
            marginal = total_stay / total_n
            ax.axhline(marginal, color=color, ls="--", alpha=0.3, lw=1)

    ax.set_xticks(range(n_D))
    ax.set_xticklabels([f"D{d}" for d in range(n_D)])
    ax.set_xlabel("Dynamic regime (D) — persistence increases →")
    ax.set_ylabel("P(stay in same E)")
    ax.set_title("(b) Self-persistence of E, conditional on D")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Transition coupling: the dynamic regime modulates economic-regime persistence",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_transition_coupling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved fig_transition_coupling.png")


if __name__ == "__main__":
    main()
