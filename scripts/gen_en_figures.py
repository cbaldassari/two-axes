"""
gen_en_figures.py -- Generate 8 English-language figures for the paper.
Saves all output to paper/en/.
"""
import sys, warnings, os
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False
    print("[WARN] adjustText not installed, labels may overlap")

SEED = 42
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results_darcsinh" / "split_W512_S6"
OUT  = ROOT / "paper" / "en"
OUT.mkdir(parents=True, exist_ok=True)

W, S, N = 512, 6, 7217

# ── Load data ─────────────────────────────────────────────────────────
labels = pd.read_parquet(DATA / "labels.parquet")
labels["datetime"] = pd.to_datetime(labels["datetime"])
pre    = pd.read_parquet(DATA / "preprocessed.parquet")
pre["datetime"] = pd.to_datetime(pre["datetime"])
eta    = pd.read_csv(DATA / "eta_squared.csv")
params = pd.read_csv(DATA / "regime_params.csv")

# ── Ordering helpers ──────────────────────────────────────────────────
e_order = labels.groupby("regime_E")["lmp_mean"].mean().sort_values().index.tolist()
d_order = labels.groupby("regime_D")["acf_6h"].mean().sort_values().index.tolist()

e_rank = {e: i for i, e in enumerate(e_order)}
d_rank = {d: i for i, d in enumerate(d_order)}

DPI = 250
SAVE_KW = dict(dpi=DPI, facecolor="white", bbox_inches="tight")


# ======================================================================
# 1. eta2_scatter.png
# ======================================================================
def fig_eta2_scatter():
    print("  [1/8] eta2_scatter.png")
    price_feats = {"lmp_mean", "lmp_p95", "lmp_std"}
    acf_feats   = {"acf_1h", "acf_6h", "acf_24h", "acf_168h"}

    display = {
        "mean": "mean", "std": "std \u0394r", "skew": "skew", "kurt": "kurtosis",
        "min": "min", "max": "max", "range": "range", "median": "median",
        "p5": "Q5", "p95": "Q95", "iqr": "IQR", "vol_24h": "vol 24h",
        "lmp_mean": "LMP mean", "lmp_p95": "LMP Q95", "lmp_std": "LMP std",
        "acf_1h": "ACF 1h", "acf_6h": "ACF 6h", "acf_24h": "ACF 24h",
        "acf_168h": "ACF 168h",
    }

    groups = []
    for f in eta["feature"]:
        if f in acf_feats:
            groups.append("ACF diagnostic")
        elif f in price_feats:
            groups.append("FE: price")
        else:
            groups.append("FE: distributional")
    eta["group"] = groups

    style = {
        "FE: distributional": dict(c="tab:blue",   marker="o", label="FE: distributional"),
        "FE: price":          dict(c="tab:green",  marker="s", label="FE: price"),
        "ACF diagnostic":     dict(c="tab:orange", marker="^", label="ACF diagnostic"),
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 0.55], [0, 0.55], "k--", lw=0.8, alpha=0.4)

    texts = []
    for grp, sty in style.items():
        sub = eta[eta["group"] == grp]
        ax.scatter(sub["eta2_FE"], sub["eta2_MOM"], s=50, zorder=3, **sty)
        for _, row in sub.iterrows():
            name = display.get(row["feature"], row["feature"])
            t = ax.annotate(name, (row["eta2_FE"], row["eta2_MOM"]),
                            fontsize=7, ha="center", va="bottom")
            texts.append(t)

    if HAS_ADJUST:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4),
                    expand=(1.8, 1.8), force_text=(0.8, 0.8))

    ax.set_xlim(-0.02, 0.56)
    ax.set_ylim(-0.03, 0.48)
    ax.set_aspect("equal")
    ax.set_xlabel(u"\u03b7\u00b2 with respect to FE partition")
    ax.set_ylabel(u"\u03b7\u00b2 with respect to MOMENT partition")
    ax.set_title("Separation diagnostic: diagonal pattern")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "eta2_scatter.png", **SAVE_KW)
    plt.close(fig)


# ======================================================================
# 2. merge_process.png -- Economic axis boxplot
# ======================================================================
def fig_merge_economic():
    print("  [2/8] merge_process.png")
    cmap = plt.cm.RdYlGn_r
    n_reg = len(e_order)

    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_regime = []
    tick_labels = []
    colors = []

    for pos, e in enumerate(e_order):
        sub = labels[labels["regime_E"] == e]
        data_by_regime.append(sub["lmp_mean"].values)
        mean_lmp = sub["lmp_mean"].mean()
        tick_labels.append(f"E{e}\nn={len(sub)}\n${mean_lmp:.0f}")
        colors.append(cmap(pos / (n_reg - 1)))

    bp = ax.boxplot(data_by_regime, positions=range(n_reg), widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)

    ax.set_xticks(range(n_reg))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("LMP mean ($/MWh)")
    ax.set_xlabel("Economic regime (ordered by LMP)", labelpad=35)
    ax.set_title("9 economic regimes after Tukey HSD merge (from 48 initial modes)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(OUT / "merge_process.png", **SAVE_KW)
    plt.close(fig)


# ======================================================================
# 3. merge_process_dynamic.png -- Dynamic axis boxplot
# ======================================================================
def fig_merge_dynamic():
    print("  [3/8] merge_process_dynamic.png")
    cmap = plt.cm.coolwarm
    n_reg = len(d_order)

    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_regime = []
    tick_labels = []
    colors = []

    for pos, d in enumerate(d_order):
        sub = labels[labels["regime_D"] == d]
        data_by_regime.append(sub["acf_6h"].values)
        mean_acf = sub["acf_6h"].mean()
        tick_labels.append(f"D{d}\nn={len(sub)}\nACF={mean_acf:.2f}")
        colors.append(cmap(pos / (n_reg - 1)))

    bp = ax.boxplot(data_by_regime, positions=range(n_reg), widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)

    ax.set_xticks(range(n_reg))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("ACF at lag 6h")
    ax.set_xlabel("Dynamic regime (increasing persistence)", labelpad=35)
    ax.set_title("9 dynamic regimes after Tukey HSD merge (from 47 initial modes)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(OUT / "merge_process_dynamic.png", **SAVE_KW)
    plt.close(fig)


# ======================================================================
# 4. ari_heatmap.png -- E x D cross-tabulation
# ======================================================================
def fig_ari_heatmap():
    print("  [4/8] ari_heatmap.png")
    ct = pd.crosstab(labels["regime_E"], labels["regime_D"])
    ct = ct.reindex(index=e_order, columns=d_order, fill_value=0)

    grid = ct.values.astype(float)
    mask = grid == 0
    grid_plot = np.where(mask, np.nan, grid)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap_h = plt.cm.YlOrRd.copy()
    cmap_h.set_bad("white")
    im = ax.imshow(grid_plot, cmap=cmap_h, aspect="auto", origin="upper")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = int(grid[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=9,
                        color="white" if val > grid[~mask].max() * 0.7 else "black")

    ax.set_xticks(range(len(d_order)))
    ax.set_xticklabels([f"D{d}" for d in d_order], fontsize=9)
    ax.set_yticks(range(len(e_order)))
    ax.set_yticklabels([f"E{e}" for e in e_order], fontsize=9)
    ax.set_xlabel("Dynamic regime (increasing persistence)")
    ax.set_ylabel("Economic regime (increasing price)")
    ax.set_title(u"E \u00d7 D Grid (ARI = 0.012, 70/81 cells populated)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Number of windows")
    fig.savefig(OUT / "ari_heatmap.png", **SAVE_KW)
    plt.close(fig)


# ======================================================================
# 5. axes_evidence.png -- 2x2 panel
# ======================================================================
def fig_axes_evidence():
    print("  [5/8] axes_evidence.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Color palettes per regime (matching Italian version)
    cmap_e = plt.cm.RdYlGn_r
    colors_e = [cmap_e(i / (len(e_order) - 1)) for i in range(len(e_order))]
    cmap_d = plt.cm.coolwarm
    colors_d = [cmap_d(i / (len(d_order) - 1)) for i in range(len(d_order))]

    def colored_boxplot(ax, data, colors, xlabels):
        bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markersize=2, alpha=0.3))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    # (a) FE regimes vs LMP — top-left
    data_a = [labels[labels["regime_E"] == e]["lmp_mean"].values for e in e_order]
    colored_boxplot(axes[0, 0], data_a, colors_e, [f"E{e}" for e in e_order])
    axes[0, 0].set_ylabel("LMP ($/MWh)")
    axes[0, 0].set_title("(a) FE regimes vs Price: clear separation", fontsize=9)

    # (b) MOMENT regimes vs LMP — top-right
    data_b = [labels[labels["regime_D"] == d]["lmp_mean"].values for d in d_order]
    colored_boxplot(axes[0, 1], data_b, colors_d, [f"D{d}" for d in d_order])
    axes[0, 1].set_title("(b) MOMENT regimes vs Price: no separation", fontsize=9)

    # (c) MOMENT regimes vs ACF — bottom-left
    data_c = [labels[labels["regime_D"] == d]["acf_6h"].values for d in d_order]
    colored_boxplot(axes[1, 0], data_c, colors_d, [f"D{d}" for d in d_order])
    axes[1, 0].set_ylabel("ACF lag 6h")
    axes[1, 0].set_title("(c) MOMENT regimes vs Persistence: clear separation", fontsize=9)

    # (d) FE regimes vs ACF — bottom-right
    data_d = [labels[labels["regime_E"] == e]["acf_6h"].values for e in e_order]
    colored_boxplot(axes[1, 1], data_d, colors_e, [f"E{e}" for e in e_order])
    axes[1, 1].set_title("(d) FE regimes vs Persistence: weak separation", fontsize=9)

    # Column headers
    fig.text(0.28, 0.98, "FE Regimes (E)", ha="center", va="top", fontsize=11, fontweight="bold")
    fig.text(0.73, 0.98, "MOMENT Regimes (D)", ha="center", va="top", fontsize=11, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "axes_evidence.png", **SAVE_KW)
    plt.close(fig)


# ======================================================================
# 6. alpha_heatmap.png
# ======================================================================
def fig_alpha_heatmap():
    print("  [6/8] alpha_heatmap.png")
    big = params[params["n"] >= 20].copy()

    n_e = len(e_order)
    n_d = len(d_order)
    grid = np.full((n_e, n_d), np.nan)

    for _, row in big.iterrows():
        ei = e_order.index(int(row["E"])) if int(row["E"]) in e_order else None
        di = d_order.index(int(row["D"])) if int(row["D"]) in d_order else None
        if ei is not None and di is not None:
            grid[ei, di] = row["alpha"]

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap_a = plt.cm.RdYlGn.copy()
    cmap_a.set_bad("white")
    im = ax.imshow(grid, cmap=cmap_a, aspect="auto", origin="upper")

    for i in range(n_e):
        for j in range(n_d):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="black")

    ax.set_xticks(range(n_d))
    ax.set_xticklabels([f"D{d}" for d in d_order], fontsize=9)
    ax.set_yticks(range(n_e))
    ax.set_yticklabels([f"E{e}" for e in e_order], fontsize=9)
    ax.set_xlabel(u"Dynamic regime (persistence \u2192)")
    ax.set_ylabel(u"Economic regime (price \u2192)")
    ax.set_title(u"Mean-reversion \u03b1 in joint (E, D) space \u2014 cells with n \u2265 20")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(u"\u03b1 (higher = faster reversion)")
    fig.savefig(OUT / "alpha_heatmap.png", **SAVE_KW)
    plt.close(fig)


# ======================================================================
# 7. mc_trajectories.png -- Monte Carlo for E3
# ======================================================================
def fig_mc_trajectories():
    print("  [7/8] mc_trajectories.png")
    # Get E3 cells with n>=20
    e3 = params[(params["E"] == 3) & (params["n"] >= 20)].copy()
    e3 = e3.sort_values("alpha")

    # Pick 3 cells spanning alpha range: lowest, middle, highest
    if len(e3) >= 3:
        idx_low  = e3.index[0]
        idx_mid  = e3.index[len(e3) // 2]
        idx_high = e3.index[-1]
        cells = e3.loc[[idx_high, idx_mid, idx_low]]  # fast, moderate, persistent
    else:
        cells = e3

    # Build r_t lookup
    pre_idx = pre.set_index("datetime")["r"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)

    for col_i, (_, cell) in enumerate(cells.iterrows()):
        e_id, d_id = int(cell["E"]), int(cell["D"])
        alpha, sigma, mu = cell["alpha"], cell["sigma"], cell["mu"]
        hl = cell["hl"]
        n_cell = int(cell["n"])

        # Find windows belonging to this (E,D) cell
        mask = (labels["regime_E"] == e_id) & (labels["regime_D"] == d_id)
        cell_labels = labels[mask]

        # Extract empirical windows (r_t)
        rng = np.random.RandomState(SEED + col_i)
        n_show = min(5, len(cell_labels))
        sample_idx = rng.choice(len(cell_labels), size=n_show, replace=False)
        sample_rows = cell_labels.iloc[sample_idx]

        # Top row: empirical
        ax_top = axes[0, col_i]
        for _, row in sample_rows.iterrows():
            t0 = row["datetime"]
            window_times = pd.date_range(t0, periods=W, freq="h")
            r_vals = pre_idx.reindex(window_times).values
            ax_top.plot(range(W), r_vals, alpha=0.6, lw=0.8, color="tab:blue")
        ax_top.set_title(f"Fast reversion\n\u03b1={alpha:.3f}, hl={hl:.0f}h (n={n_cell})"
                         if col_i == 0 else
                         f"Moderate reversion\n\u03b1={alpha:.3f}, hl={hl:.0f}h (n={n_cell})"
                         if col_i == 1 else
                         f"Persistent\n\u03b1={alpha:.3f}, hl={hl:.0f}h (n={n_cell})",
                         fontsize=9)
        ax_top.grid(True, alpha=0.3)
        if col_i == 0:
            ax_top.set_ylabel("Empirical ($r_t$)", fontsize=9)

        # Bottom row: simulated AR(1)
        ax_bot = axes[1, col_i]
        for _ in range(5):
            x = np.zeros(W)
            x[0] = rng.normal(0, sigma)
            for t in range(1, W):
                x[t] = (1 - alpha) * x[t-1] + mu + rng.normal(0, sigma)
            ax_bot.plot(range(W), x, alpha=0.6, lw=0.8, color="tab:red")
        ax_bot.set_xlabel("Hours")
        ax_bot.grid(True, alpha=0.3)
        if col_i == 0:
            ax_bot.set_ylabel("Monte Carlo simulated", fontsize=9)

    # Overall title -- compute mean LMP for E3
    mean_lmp_e3 = labels[labels["regime_E"] == 3]["lmp_mean"].mean()
    fig.suptitle(f"Regime E3 (Baseload, ${mean_lmp_e3:.0f}/MWh) \u2014 three dynamic regimes at the same price level",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "mc_trajectories.png", **SAVE_KW)
    plt.close(fig)


# ======================================================================
# 8. two_axis_grid.png -- Quadrant scatter
# ======================================================================
def fig_two_axis_grid():
    print("  [8/8] two_axis_grid.png")
    fig, ax = plt.subplots(figsize=(9, 7))

    cmap_tab = plt.cm.tab10
    for i, e in enumerate(e_order):
        sub = labels[labels["regime_E"] == e]
        ax.scatter(sub["lmp_mean"], sub["acf_6h"], s=6, alpha=0.4,
                   color=cmap_tab(i / 9), label=f"E{e}", rasterized=True)

    med_lmp = labels["lmp_mean"].median()
    med_acf = labels["acf_6h"].median()
    ax.axvline(med_lmp, color="black", ls="--", lw=1.2, alpha=0.5)
    ax.axhline(med_acf, color="black", ls="--", lw=1.2, alpha=0.5)

    # Quadrant percentages
    n_total = len(labels)
    tl = ((labels["lmp_mean"] < med_lmp) & (labels["acf_6h"] >= med_acf)).sum()
    tr = ((labels["lmp_mean"] >= med_lmp) & (labels["acf_6h"] >= med_acf)).sum()
    bl = ((labels["lmp_mean"] < med_lmp) & (labels["acf_6h"] < med_acf)).sum()
    br = ((labels["lmp_mean"] >= med_lmp) & (labels["acf_6h"] < med_acf)).sum()

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.text(med_lmp * 0.35, ylim[1] * 0.95,
            f"Calm persistent\n{tl/n_total*100:.0f}%",
            ha="center", va="top", fontsize=9, bbox=bbox_props)
    ax.text(xlim[1] * 0.75, ylim[1] * 0.95,
            f"Persistent stress\n{tr/n_total*100:.0f}%",
            ha="center", va="top", fontsize=9, bbox=bbox_props)
    ax.text(med_lmp * 0.35, ylim[0] + (med_acf - ylim[0]) * 0.15,
            f"Calm transitory\n{bl/n_total*100:.0f}%",
            ha="center", va="bottom", fontsize=9, bbox=bbox_props)
    ax.text(xlim[1] * 0.75, ylim[0] + (med_acf - ylim[0]) * 0.15,
            f"Fast spike\n{br/n_total*100:.0f}%",
            ha="center", va="bottom", fontsize=9, bbox=bbox_props)

    ax.set_xlabel("LMP mean ($/MWh)")
    ax.set_ylabel("ACF at lag 6h")
    ax.set_title("Two-axis regime space")
    ax.legend(loc="upper right", fontsize=7, markerscale=3, ncol=3)
    ax.grid(True, alpha=0.2)
    fig.savefig(OUT / "two_axis_grid.png", **SAVE_KW)
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print(f"Generating 8 English figures in {OUT}")
    fig_eta2_scatter()
    fig_merge_economic()
    fig_merge_dynamic()
    fig_ari_heatmap()
    fig_axes_evidence()
    fig_alpha_heatmap()
    fig_mc_trajectories()
    fig_two_axis_grid()

    # Verify
    expected = [
        "eta2_scatter.png", "merge_process.png", "merge_process_dynamic.png",
        "ari_heatmap.png", "axes_evidence.png", "alpha_heatmap.png",
        "mc_trajectories.png", "two_axis_grid.png",
    ]
    print("\nVerification:")
    all_ok = True
    for f in expected:
        p = OUT / f
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  OK  {f}  ({size_kb:.0f} KB)")
        else:
            print(f"  MISSING  {f}")
            all_ok = False
    if all_ok:
        print(f"\nAll 8 figures saved to {OUT}")
    else:
        print("\nSome figures are missing!")
