"""
fig_why_two_axes.py
===================
Simplest possible: show two REAL price episodes from the same
economic regime — one where the shock passes quickly (fast D),
one where it persists for days (persistent D).
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
    return df, pre


def find_episode(df, pre, e_target, d_low, d_high, want_persistent):
    """Find a stretch where E=e_target and D is in the desired range,
    then show the price before, during, and after."""
    mask = (df["E"] == e_target)
    if want_persistent:
        mask &= (df["D"] >= d_high)
    else:
        mask &= (df["D"] <= d_low)

    candidates = df[mask].copy()
    if len(candidates) == 0:
        return None, None

    # Find a run of at least 3 consecutive windows in this state
    idx = candidates.index.values
    best_start = None
    best_len = 0
    cur_start = idx[0]
    cur_len = 1
    for i in range(1, len(idx)):
        if idx[i] == idx[i-1] + 1:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
            cur_start = idx[i]
            cur_len = 1
    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start

    if best_len < 2:
        return None, None

    # Get the datetime of the middle of the episode
    mid = best_start + best_len // 2
    center_dt = df.loc[mid, "datetime"]

    # Show 5 days of hourly prices around this point
    window_h = 72  # hours each side
    pre_ts = pre.set_index("datetime")["lmp"]
    start_dt = center_dt - pd.Timedelta(hours=window_h)
    end_dt = center_dt + pd.Timedelta(hours=window_h)
    episode = pre_ts[start_dt:end_dt]

    return episode, center_dt


def main():
    df, pre = load()
    df = df.reset_index(drop=True)

    # Find episodes for E=6 (Stress, ~$83) or E=5 (Demand, ~$73)
    # Try E=5 first — more data
    for e_target, e_name, e_price in [(6, "Stress", 83), (5, "Demand", 73), (7, "Winter spike", 92)]:
        ep_fast, dt_fast = find_episode(df, pre, e_target, d_low=1, d_high=6, want_persistent=False)
        ep_pers, dt_pers = find_episode(df, pre, e_target, d_low=1, d_high=6, want_persistent=True)
        if ep_fast is not None and ep_pers is not None and len(ep_fast) > 20 and len(ep_pers) > 20:
            print(f"Using E{e_target} ({e_name}, ~${e_price}/MWh)")
            print(f"  Fast episode around {dt_fast}")
            print(f"  Persistent episode around {dt_pers}")
            break

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Left: fast-reverting
    ax = axes[0]
    hours = np.arange(len(ep_fast)) - len(ep_fast) // 2
    ax.plot(hours, ep_fast.values, color="#3498db", lw=2)
    ax.axhline(e_price, color="gray", ls=":", lw=1, label=f"Avg regime price (~${e_price})")
    ax.axvspan(-12, 12, alpha=0.15, color="#3498db", label="Regime window")
    ax.set_xlabel("Hours from center of episode", fontsize=11)
    ax.set_ylabel("Price ($/MWh)", fontsize=12)
    ax.set_title("Shock passes quickly\n(fast-reverting regime)", fontsize=14,
                 color="#2980b9", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # Right: persistent
    ax = axes[1]
    hours = np.arange(len(ep_pers)) - len(ep_pers) // 2
    ax.plot(hours, ep_pers.values, color="#e74c3c", lw=2)
    ax.axhline(e_price, color="gray", ls=":", lw=1, label=f"Avg regime price (~${e_price})")
    ax.axvspan(-12, 12, alpha=0.15, color="#e74c3c", label="Regime window")
    ax.set_xlabel("Hours from center of episode", fontsize=11)
    ax.set_title("Shock persists for days\n(persistent regime)", fontsize=14,
                 color="#c0392b", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Same economic regime ({e_name}, ~${e_price}/MWh) — two very different behaviors",
                 fontsize=14, fontweight="bold", y=1.03)

    fig.text(0.5, -0.04,
             "A single 'high price' label treats both situations the same.\n"
             "The dynamic axis tells you WHICH one you are in.",
             ha="center", fontsize=12, style="italic")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_why_two_axes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_why_two_axes.png")


if __name__ == "__main__":
    main()
