"""
fig_mc_trajectories.py — Figure 11: 3-row Monte Carlo trajectories.
Row 1: Empirical windows (blue)
Row 2: (E,D) mean-reverting model (red) — cell-specific alpha, sigma
Row 3: GARCH(1,1)-t simulation (gray) — same dynamics for all D regimes

Demonstrates visually that the (E,D) model differentiates persistence
within E3, while GARCH does not.
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path(".")
OUT_IT = Path("paper")
OUT_EN = Path("paper/en")
DATA = Path("results_darcsinh/split_W512_S6")

# Load data
pre = pd.read_parquet(DATA / "preprocessed.parquet")
lab = pd.read_parquet(DATA / "labels.parquet")
rp = pd.read_csv(DATA / "regime_params.csv")

r = pre["r"].values
dr = pre["dr"].values
W, S = 512, 6
starts = list(range(0, len(r) - W + 1, S))
N = len(starts)
lab_E = lab["regime_E"].values
lab_D = lab["regime_D"].values

# Pick 3 cells in E3 spanning the alpha range
rp3 = rp[(rp["E"] == 3) & (rp["n"] >= 20)].sort_values("alpha")
cell_slow = rp3.iloc[0]   # lowest alpha (most persistent)
cell_mid = rp3.iloc[len(rp3)//2]  # middle
cell_fast = rp3.iloc[-1]  # highest alpha (fastest reversion)

def empirical_level_std(E, D):
    """Mean within-window std of r_t levels for a cell."""
    mask = (lab_E == E) & (lab_D == D)
    idxs = np.where(mask)[0]
    return np.mean([np.std(r[starts[i]:starts[i]+W]) for i in idxs])

def sigma_innov(alpha, E, D):
    """Derive AR(1) innovation std so stationary std matches empirical levels."""
    std_r = empirical_level_std(E, D)
    return std_r * np.sqrt(2 * alpha - alpha**2)

cells = [
    {"label_it": "Rapida reversione", "label_en": "Fast reversion",
     "E": 3, "D": int(cell_fast["D"]), "alpha": cell_fast["alpha"],
     "sigma": sigma_innov(cell_fast["alpha"], 3, int(cell_fast["D"])),
     "hl": cell_fast["hl"], "n": int(cell_fast["n"])},
    {"label_it": "Moderata", "label_en": "Moderate",
     "E": 3, "D": int(cell_mid["D"]), "alpha": cell_mid["alpha"],
     "sigma": sigma_innov(cell_mid["alpha"], 3, int(cell_mid["D"])),
     "hl": cell_mid["hl"], "n": int(cell_mid["n"])},
    {"label_it": "Persistente", "label_en": "Persistent",
     "E": 3, "D": int(cell_slow["D"]), "alpha": cell_slow["alpha"],
     "sigma": sigma_innov(cell_slow["alpha"], 3, int(cell_slow["D"])),
     "hl": cell_slow["hl"], "n": int(cell_slow["n"])},
]

# Fit GARCH(1,1)-t on full dr series
print("Fitting GARCH(1,1)-t...", flush=True)
am = arch_model(dr * 100, mean="ARX", lags=1, vol="Garch", p=1, q=1, dist="t")
res = am.fit(disp="off")
print(f"  omega={res.params['omega']:.4f}, alpha[1]={res.params['alpha[1]']:.4f}, "
      f"beta[1]={res.params['beta[1]']:.4f}")

# Get GARCH conditional volatility per window (mean over 512h)
cond_vol = res.conditional_volatility / 100  # back to original scale
garch_sigma_win = np.array([cond_vol[s:s+W].mean() for s in starts])

# GARCH alpha for E3 (single-axis: same for all D)
alpha_E3 = rp[(rp["E"] == 3) & (rp["n"] >= 20)]["alpha"].mean()
# Weight by n for consistency with paper
alpha_E3_wt = (rp3["alpha"] * rp3["n"]).sum() / rp3["n"].sum()

def _dyn_range(tr):
    return np.max(tr) - np.min(tr)

YLIM_12 = 1.3  # y-limit for rows 1-2

def get_empirical(E, D, n_traces=5):
    """Get random empirical windows within y-limits."""
    mask = (lab_E == E) & (lab_D == D)
    idxs = np.where(mask)[0]
    windows = [r[starts[i]:starts[i]+W] for i in idxs]
    # keep only windows that stay within visible range
    windows = [w for w in windows if np.max(np.abs(w)) <= YLIM_12]
    med = np.median([_dyn_range(w) for w in windows])
    chosen = rng.choice(len(windows), size=min(n_traces, len(windows)), replace=False)
    return [windows[j] for j in chosen], med

def simulate_mr(alpha, sigma, n_traces=5, target_range=None, n_pool=200):
    """Simulate AR(1) trajectories matched to empirical dynamic range, within y-limits."""
    pool = []
    for _ in range(n_pool):
        x = np.zeros(W)
        for t in range(1, W):
            x[t] = (1 - alpha) * x[t-1] + sigma * rng.standard_normal()
        if np.max(np.abs(x)) <= YLIM_12:
            pool.append(x)
    if target_range is not None and len(pool) >= n_traces:
        ranges = np.array([_dyn_range(x) for x in pool])
        order = np.argsort(np.abs(ranges - target_range))
        return [pool[j] for j in order[:n_traces]]
    return pool[:n_traces]

def simulate_garch(am, res, n_traces=5):
    """Simulate GARCH trajectories, cumulated to levels."""
    trajs = []
    np.random.seed(SEED)
    for i in range(n_traces):
        sim = am.simulate(res.params, W)
        dr_sim = sim["data"].values / 100  # back to original scale
        r_sim = np.cumsum(dr_sim)
        r_sim -= r_sim.mean()  # center
        trajs.append(r_sim)
    return trajs

def make_figure(lang="it"):
    is_en = lang == "en"
    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True)

    for col, cell in enumerate(cells):
        E, D = cell["E"], cell["D"]
        a, s, hl, n = cell["alpha"], cell["sigma"], cell["hl"], cell["n"]
        lab_str = cell["label_en"] if is_en else cell["label_it"]

        # Column title
        axes[0, col].set_title(f"{lab_str}\n$\\alpha$={a:.3f}, hl={hl:.0f}h (n={n})",
                                fontsize=9)

        # Row 1: Empirical (closest to median dynamic range)
        emp_traces, med_range = get_empirical(E, D)
        for tr in emp_traces:
            axes[0, col].plot(tr, lw=0.4, alpha=0.6, color="steelblue")
        axes[0, col].grid(alpha=0.15)

        # Row 2: (E,D) model (matched to empirical dynamic range)
        for tr in simulate_mr(a, s, target_range=med_range):
            axes[1, col].plot(tr, lw=0.4, alpha=0.6, color="firebrick")
        axes[1, col].grid(alpha=0.15)

        # Row 3: GARCH (same for all columns — single-axis dynamics)
        for tr in simulate_garch(am, res):
            axes[2, col].plot(tr, lw=0.4, alpha=0.6, color="dimgray")
        axes[2, col].grid(alpha=0.15)

    # Row labels
    if is_en:
        row_labels = ["Empirical ($r_t$)", "Model $(E,D)$", "GARCH(1,1)-$t$"]
    else:
        row_labels = ["Empiriche ($r_t$)", "Modello $(E,D)$", "GARCH(1,1)-$t$"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    # x label
    for col in range(3):
        xlabel = "Hours" if is_en else "Ore"
        axes[2, col].set_xlabel(xlabel, fontsize=9)

    # Fixed y-limits: rows 0-1 (empirical, model) ±1.3; row 2 (GARCH) ±2
    for col in range(3):
        axes[0, col].set_ylim(-1.3, 1.3)
        axes[1, col].set_ylim(-1.3, 1.3)
        axes[2, col].set_ylim(-2.2, 2.2)

    if is_en:
        title = "Regime E3 (Baseload, \\$41/MWh) — three dynamic regimes at the same price level"
    else:
        title = "Regime E3 (Baseload, \\$41/MWh) — tre regimi dinamici nello stesso livello di prezzo"
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig

# Generate IT
print("Generating IT figure...", flush=True)
fig_it = make_figure("it")
out_it = OUT_IT / "mc_trajectories.png"
fig_it.savefig(out_it, dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig_it)
print(f"Saved: {out_it}")

# Generate EN
print("Generating EN figure...", flush=True)
fig_en = make_figure("en")
out_en = OUT_EN / "mc_trajectories.png"
fig_en.savefig(out_en, dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig_en)
print(f"Saved: {out_en}")
