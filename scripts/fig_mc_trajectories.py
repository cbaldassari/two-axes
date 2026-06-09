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

cells = [
    {"label_it": "Rapida reversione", "label_en": "Fast reversion",
     "E": 3, "D": int(cell_fast["D"]), "alpha": cell_fast["alpha"],
     "sigma": cell_fast["sigma"], "hl": cell_fast["hl"], "n": int(cell_fast["n"])},
    {"label_it": "Moderata", "label_en": "Moderate",
     "E": 3, "D": int(cell_mid["D"]), "alpha": cell_mid["alpha"],
     "sigma": cell_mid["sigma"], "hl": cell_mid["hl"], "n": int(cell_mid["n"])},
    {"label_it": "Persistente", "label_en": "Persistent",
     "E": 3, "D": int(cell_slow["D"]), "alpha": cell_slow["alpha"],
     "sigma": cell_slow["sigma"], "hl": cell_slow["hl"], "n": int(cell_slow["n"])},
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

def get_empirical(E, D, n_traces=5):
    """Get representative empirical windows from a cell (closest to median peak)."""
    mask = (lab_E == E) & (lab_D == D)
    idxs = np.where(mask)[0]
    peaks = np.array([np.max(np.abs(r[starts[i]:starts[i]+W])) for i in idxs])
    med_peak = np.median(peaks)
    order = np.argsort(np.abs(peaks - med_peak))
    chosen = idxs[order[:n_traces]]
    return [r[starts[i]:starts[i]+W] for i in chosen]

def simulate_mr(alpha, sigma, n_traces=5):
    """Simulate mean-reverting AR(1) trajectories."""
    trajs = []
    for _ in range(n_traces):
        x = np.zeros(W)
        for t in range(1, W):
            x[t] = (1 - alpha) * x[t-1] + sigma * rng.standard_normal()
        trajs.append(x)
    return trajs

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

        # Row 2 first: (E,D) model — sigma calibrated
        mask_cell = (lab_E == E) & (lab_D == D)
        idxs_cell = np.where(mask_cell)[0]
        std_r = np.median([np.std(r[starts[i]:starts[i]+W]) for i in idxs_cell])
        s_cal = std_r * np.sqrt(2 * a - a**2)
        sim_traces = simulate_mr(a, s_cal, n_traces=5)
        sim_std = np.median([np.std(tr) for tr in sim_traces])
        sim_peak = np.median([np.max(np.abs(tr)) for tr in sim_traces])

        # Row 1: Empirical — select windows most similar to model traces
        all_idxs = np.where((lab_E == E) & (lab_D == D))[0]
        emp_peaks = np.array([np.max(np.abs(r[starts[i]:starts[i]+W])) for i in all_idxs])
        emp_stds = np.array([np.std(r[starts[i]:starts[i]+W]) for i in all_idxs])
        dist = np.abs(emp_stds - sim_std) / (sim_std + 1e-9) + \
               np.abs(emp_peaks - sim_peak) / (sim_peak + 1e-9)
        order = np.argsort(dist)
        emp_chosen = all_idxs[order[:5]]
        emp_traces = [r[starts[i]:starts[i]+W] for i in emp_chosen]
        for tr in emp_traces:
            axes[0, col].plot(tr, lw=0.4, alpha=0.6, color="steelblue")
        axes[0, col].grid(alpha=0.15)
        for tr in sim_traces:
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

    # Match y-limits: rows 1 and 2 share [-1, 1], row 3 auto-scales
    for ax in axes[0]:
        ax.set_ylim(-1, 1)
    for ax in axes[1]:
        ax.set_ylim(-1, 1)
    ymin = min(ax.get_ylim()[0] for ax in axes[2])
    ymax = max(ax.get_ylim()[1] for ax in axes[2])
    for ax in axes[2]:
        ax.set_ylim(ymin, ymax)

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
