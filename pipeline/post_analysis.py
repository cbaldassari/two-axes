"""
post_analysis.py
================
NEPOOL TDA Pipeline — Comprehensive Post-Analysis

Produces:
  Section 1: Comparison table across 3 ToMATo configs (CSV + console)
  Section 2: Regime characterization for B:FE+ToMATo (best config)
  Section 3: ILR analysis plots (6 figures)
  Section 4: Cross-approach pairwise ARI

Output directory: results/post_analysis/

Usage:
  python pipeline/post_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

# ── project root on sys.path ──
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================

RESULTS_DIR = Path(C.RESULTS_DIR)
OUT_DIR = RESULTS_DIR / "post_analysis"
DPI = 200
FACECOLOR = "white"

# Experiment directory -> display name
EXPERIMENTS = ["C", "FE", "COMBO"]
EXP_NAMES = {"C": "A:MOMENT", "FE": "B:FE", "COMBO": "C:COMBO"}

# 3 configurations: 3 representations x ToMATo only
CONFIGS = []
for exp in EXPERIMENTS:
    CONFIGS.append({
        "name": f"{EXP_NAMES[exp]}+ToMATo",
        "exp": exp,
        "labels_dir": "step04",
        "quality_dir": "step05",
    })

# Best config
BEST_EXP = "FE"
BEST_LABELS_DIR = "step04"
BEST_NAME = "B:FE+ToMATo"

SEASONS = {
    "inverno": [12, 1, 2],
    "primavera": [3, 4, 5],
    "estate": [6, 7, 8],
    "autunno": [9, 10, 11],
}

FUEL_COLS = [
    "natural_gas_share", "nuclear_share", "hydro_share", "wind_share",
    "solar_share", "coal_share", "oil_share", "other_share",
]
FUEL_SHORT = ["Gas", "Nuclear", "Hydro", "Wind", "Solar", "Coal", "Oil", "Other"]
FUEL_COLORS = {
    "Gas": "#e07b39", "Nuclear": "#7b52ab", "Hydro": "#4da6e8",
    "Wind": "#6abf69", "Solar": "#f7c948", "Coal": "#5a5a5a",
    "Oil": "#b04040", "Other": "#aaaaaa",
}

ILR_NAMES = {
    "ilr_1": "ilr_1: Dispatchable vs Variable",
    "ilr_2": "ilr_2: Fossil (gas,coal,oil) vs Non-fossil disp. (nuc,other)",
    "ilr_3": "ilr_3: Gas vs Solid fossil (coal,oil)",
    "ilr_4": "ilr_4: Coal vs Oil",
    "ilr_5": "ilr_5: Nuclear vs Other",
    "ilr_6": "ilr_6: Hydro vs Intermittent (wind,solar)",
    "ilr_7": "ilr_7: Solar vs Wind",
}

STRIDE_H = 6
CMAP = plt.get_cmap("tab20")


def regime_colors(n):
    """Return n distinct colors from tab20."""
    return [CMAP(i % 20) for i in range(n)]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_preprocessed() -> pd.DataFrame:
    """Load preprocessed.parquet (hourly)."""
    df = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_original() -> pd.DataFrame:
    """Load isone_dataset.parquet (original with fuel shares)."""
    df = pd.read_parquet(Path(C.DATA_PATH))
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_labels(exp: str, labels_dir: str) -> pd.DataFrame:
    """Load labels.parquet for a given experiment and clustering step."""
    path = RESULTS_DIR / f"exp_{exp}" / labels_dir / "labels.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_quality_report(exp: str, quality_dir: str) -> dict | None:
    """Load quality_report.json for a given experiment."""
    path = RESULTS_DIR / f"exp_{exp}" / quality_dir / "quality_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_bootstrap_ari() -> dict:
    """Load bootstrap_ari.json."""
    path = RESULTS_DIR / "bootstrap_ari.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def assign_season(month_series: pd.Series) -> pd.Series:
    """Map month number to season name."""
    season = pd.Series("", index=month_series.index)
    for sname, months in SEASONS.items():
        season[month_series.isin(months)] = sname
    return season


def regime_order_by_lmp(df_merged: pd.DataFrame) -> list[int]:
    """Return cluster labels sorted by ascending mean LMP."""
    means = df_merged.groupby("cluster")["lmp"].mean().sort_values()
    return list(means.index)


# =============================================================================
# SECTION 1: COMPARISON TABLE
# =============================================================================

def section1_comparison_table():
    """Build comparison table across 3 ToMATo configs and save as CSV."""
    print("\n" + "=" * 70)
    print("  SECTION 1: Comparison table across 3 ToMATo configs")
    print("=" * 70, flush=True)

    bootstrap = load_bootstrap_ari()
    rows = []

    for cfg in CONFIGS:
        q = load_quality_report(cfg["exp"], cfg["quality_dir"])
        if q is None:
            print(f"  [SKIP] {cfg['name']} — quality report not found", flush=True)
            continue

        geo = q.get("geometric", {})
        eco = q.get("economic", {})

        K = len(eco.get("lmp_per_regime", {}))
        eta2 = eco.get("eta_squared", np.nan)
        sil = geo.get("silhouette_avg", np.nan)
        db = geo.get("davies_bouldin", np.nan)
        ch = geo.get("calinski_harabasz", np.nan)
        dunn = geo.get("dunn_index", np.nan)
        trans_rate = eco.get("transition_rate", np.nan)
        sojourn_h = eco.get("sojourn_mean_hours", np.nan)
        cramer_season = eco.get("cramer_v_season", np.nan)
        tukey_sig = eco.get("pairwise_significant", 0)
        tukey_tot = eco.get("pairwise_total", 0)
        tukey_pct = tukey_sig / tukey_tot if tukey_tot > 0 else np.nan

        # Bootstrap ARI
        bari = bootstrap.get(cfg["name"], {})
        bari_mean = bari.get("mean", np.nan)
        bari_std = bari.get("std", np.nan)

        row = {
            "config": cfg["name"],
            "K": K,
            "eta_squared": eta2,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "dunn_index": dunn,
            "transition_rate": trans_rate,
            "sojourn_mean_h": sojourn_h,
            "cramer_v_season": cramer_season,
            "tukey_pct": tukey_pct,
            "bootstrap_ari_mean": bari_mean,
            "bootstrap_ari_std": bari_std,
        }
        rows.append(row)

    df_table = pd.DataFrame(rows)

    # Print formatted
    print()
    header = (f"{'Config':<22s} {'K':>3s} {'eta2':>7s} {'Sil':>7s} "
              f"{'DB':>7s} {'CH':>9s} {'Dunn':>7s} {'Trans%':>7s} "
              f"{'Soj_h':>7s} {'Cramer':>7s} {'Tukey':>7s} "
              f"{'bARI':>12s}")
    print(header)
    print("-" * len(header))

    for _, r in df_table.iterrows():
        bari_s = (f"{r['bootstrap_ari_mean']:.3f}+/-{r['bootstrap_ari_std']:.3f}"
                  if not np.isnan(r['bootstrap_ari_mean']) else "N/A")
        print(f"{r['config']:<22s} {r['K']:>3.0f} {r['eta_squared']:>7.4f} "
              f"{r['silhouette']:>7.4f} {r['davies_bouldin']:>7.4f} "
              f"{r['calinski_harabasz']:>9.1f} {r['dunn_index']:>7.4f} "
              f"{r['transition_rate']:>7.4f} {r['sojourn_mean_h']:>7.0f} "
              f"{r['cramer_v_season']:>7.4f} {r['tukey_pct']:>7.2f} "
              f"{bari_s:>12s}")

    # Save CSV
    df_table.to_csv(OUT_DIR / "comparison_table.csv", index=False, float_format="%.4f")
    print(f"\n  Saved: {OUT_DIR / 'comparison_table.csv'}", flush=True)
    return df_table


# =============================================================================
# SECTION 2: REGIME CHARACTERIZATION (B:FE+ToMATo)
# =============================================================================

def section2_regime_characterization():
    """Detailed regime characterization for the best config."""
    print("\n" + "=" * 70)
    print("  SECTION 2: Regime characterization (B:FE+ToMATo)")
    print("=" * 70, flush=True)

    df_pre = load_preprocessed()
    df_orig = load_original()
    df_labels = load_labels(BEST_EXP, BEST_LABELS_DIR)

    if df_labels is None:
        print("  [ERROR] Labels not found!", flush=True)
        return None

    # Merge labels with preprocessed (for LMP stats + ILR)
    df_m = df_pre.merge(df_labels, on="datetime", how="inner")
    # Merge with original (for fuel shares)
    df_fuel = df_orig[["datetime"] + FUEL_COLS]
    df_m = df_m.merge(df_fuel, on="datetime", how="left")

    print(f"  Merged: {len(df_m)} rows, {df_m['cluster'].nunique()} regimes", flush=True)

    # Order regimes by mean LMP
    order = regime_order_by_lmp(df_m)
    print(f"  Regime order (by mean LMP): {order}", flush=True)

    rows = []
    for rank, k in enumerate(order):
        sub = df_m[df_m["cluster"] == k]
        lmp = sub["lmp"].values
        month = sub["datetime"].dt.month

        # Season distribution
        season = assign_season(month)
        season_pct = {}
        for sname in SEASONS:
            season_pct[f"season_{sname}_pct"] = (season == sname).mean() * 100

        # Fuel shares
        fuel_means = {}
        for fc, short in zip(FUEL_COLS, FUEL_SHORT):
            fuel_means[f"fuel_{short}_pct"] = sub[fc].mean() * 100

        # ILR coordinates
        ilr_means = {}
        for i in range(1, 8):
            col = f"ilr_{i}"
            ilr_means[f"mean_{col}"] = sub[col].mean()

        # Sojourn for this regime (from window-level labels)
        labels_all = df_labels.sort_values("datetime")["cluster"].values
        runs = []
        current_len = 0
        for c in labels_all:
            if c == k:
                current_len += 1
            else:
                if current_len > 0:
                    runs.append(current_len)
                current_len = 0
        if current_len > 0:
            runs.append(current_len)
        sojourn_mean_h = np.mean(runs) * STRIDE_H if runs else 0

        row = {
            "regime_rank": rank,
            "cluster_id": k,
            "n_obs": len(sub),
            "lmp_mean": np.mean(lmp),
            "lmp_median": np.median(lmp),
            "lmp_std": np.std(lmp),
            "lmp_p5": np.percentile(lmp, 5),
            "lmp_p95": np.percentile(lmp, 95),
            "sojourn_mean_h": sojourn_mean_h,
            **season_pct,
            **fuel_means,
            **ilr_means,
        }
        rows.append(row)

    df_char = pd.DataFrame(rows)

    # Print summary
    print()
    for _, r in df_char.iterrows():
        print(f"  Regime R{r['cluster_id']:.0f} (rank={r['regime_rank']:.0f}, "
              f"n={r['n_obs']:.0f})")
        print(f"    LMP: mean={r['lmp_mean']:.2f}  median={r['lmp_median']:.2f}  "
              f"std={r['lmp_std']:.2f}  [p5={r['lmp_p5']:.2f}, p95={r['lmp_p95']:.2f}]")
        print(f"    Sojourn: {r['sojourn_mean_h']:.0f}h")
        print(f"    Seasons: inv={r['season_inverno_pct']:.1f}%  "
              f"pri={r['season_primavera_pct']:.1f}%  "
              f"est={r['season_estate_pct']:.1f}%  "
              f"aut={r['season_autunno_pct']:.1f}%")
        fuel_str = "  ".join([f"{s}={r[f'fuel_{s}_pct']:.1f}%"
                              for s in FUEL_SHORT])
        print(f"    Fuel: {fuel_str}")
        print()

    # Save
    path = OUT_DIR / "regime_characterization_FE_tomato.csv"
    df_char.to_csv(path, index=False, float_format="%.4f")
    print(f"  Saved: {path}", flush=True)
    return df_char


# =============================================================================
# SECTION 3: PLOTS
# =============================================================================

def _get_best_merged():
    """Load and merge data for the best config (B:FE+ToMATo)."""
    df_pre = load_preprocessed()
    df_orig = load_original()
    df_labels = load_labels(BEST_EXP, BEST_LABELS_DIR)
    if df_labels is None:
        return None, None

    df_m = df_pre.merge(df_labels, on="datetime", how="inner")
    df_fuel = df_orig[["datetime"] + FUEL_COLS]
    df_m = df_m.merge(df_fuel, on="datetime", how="left")

    order = regime_order_by_lmp(df_m)
    return df_m, order


def plot01_ilr_by_regime(df_m: pd.DataFrame, order: list[int]):
    """Plot 1: Boxplot of each ILR coordinate by regime."""
    print("  Plotting 01_ilr_by_regime.png ...", flush=True)

    ilr_cols = [f"ilr_{i}" for i in range(1, 8)]
    n_ilr = len(ilr_cols)
    n_regimes = len(order)
    colors = regime_colors(n_regimes)

    fig, axes = plt.subplots(n_ilr, 1, figsize=(12, 3.5 * n_ilr), facecolor=FACECOLOR)

    for idx, col in enumerate(ilr_cols):
        ax = axes[idx]
        data = [df_m.loc[df_m["cluster"] == k, col].dropna().values for k in order]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(0.8)

        ax.set_xticklabels([f"R{k}" for k in order], fontsize=9)
        ax.set_ylabel(col, fontsize=10)
        ax.set_title(ILR_NAMES[col], fontsize=11, fontweight="bold")
        ax.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("ILR Coordinates by Regime (B:FE+ToMATo, ordered by mean LMP)",
                 fontsize=14, fontweight="bold", y=1.002)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_ilr_by_regime.png", dpi=DPI, facecolor=FACECOLOR,
                bbox_inches="tight")
    plt.close(fig)
    print("    Done.", flush=True)


def plot02_fuel_mix_by_regime(df_m: pd.DataFrame, order: list[int]):
    """Plot 2: Stacked bar chart of mean fuel mix per regime."""
    print("  Plotting 02_fuel_mix_by_regime.png ...", flush=True)

    n_regimes = len(order)
    means = {}
    for k in order:
        sub = df_m[df_m["cluster"] == k]
        means[k] = {short: sub[fc].mean() * 100 for fc, short in zip(FUEL_COLS, FUEL_SHORT)}

    fig, ax = plt.subplots(figsize=(max(10, n_regimes * 1.2), 7), facecolor=FACECOLOR)

    x = np.arange(n_regimes)
    bottoms = np.zeros(n_regimes)

    for fuel in FUEL_SHORT:
        vals = np.array([means[k][fuel] for k in order])
        ax.bar(x, vals, bottom=bottoms, label=fuel,
               color=FUEL_COLORS[fuel], edgecolor="white", linewidth=0.5, width=0.7)
        # Label inside bars if large enough
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 3:
                ax.text(i, b + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    lmp_means = [df_m.loc[df_m["cluster"] == k, "lmp"].mean() for k in order]
    ax.set_xticklabels([f"R{k}\n(${m:.0f})" for k, m in zip(order, lmp_means)],
                       fontsize=9)
    ax.set_ylabel("Fuel Share (%)", fontsize=11)
    ax.set_title("Mean Fuel Mix by Regime (B:FE+ToMATo, ordered by mean LMP)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_fuel_mix_by_regime.png", dpi=DPI, facecolor=FACECOLOR,
                bbox_inches="tight")
    plt.close(fig)
    print("    Done.", flush=True)


def plot03_regime_heatmap(df_m: pd.DataFrame, order: list[int]):
    """Plot 3: Heatmap with normalized columns, annotated with actual values."""
    print("  Plotting 03_regime_heatmap.png ...", flush=True)

    # Build data matrix
    cols_display = [
        "LMP_mean", "LMP_std",
        "Gas%", "Nuc%", "Hydro%", "Wind%", "Solar%", "Coal%", "Oil%", "Other%",
        "Winter%", "Summer%", "Sojourn_h",
    ]

    # Compute per-regime stats
    labels_df = load_labels(BEST_EXP, BEST_LABELS_DIR)
    labels_all = labels_df.sort_values("datetime")["cluster"].values

    matrix_raw = []
    for k in order:
        sub = df_m[df_m["cluster"] == k]
        lmp_mean = sub["lmp"].mean()
        lmp_std = sub["lmp"].std()

        fuel_pcts = [sub[fc].mean() * 100 for fc in FUEL_COLS]

        month = sub["datetime"].dt.month
        season = assign_season(month)
        winter_pct = (season == "inverno").mean() * 100
        summer_pct = (season == "estate").mean() * 100

        # Sojourn
        runs = []
        current_len = 0
        for c in labels_all:
            if c == k:
                current_len += 1
            else:
                if current_len > 0:
                    runs.append(current_len)
                current_len = 0
        if current_len > 0:
            runs.append(current_len)
        sojourn_h = np.mean(runs) * STRIDE_H if runs else 0

        row = [lmp_mean, lmp_std] + fuel_pcts + [winter_pct, summer_pct, sojourn_h]
        matrix_raw.append(row)

    matrix_raw = np.array(matrix_raw)
    n_regimes, n_cols = matrix_raw.shape

    # Normalize each column to [0, 1]
    col_min = matrix_raw.min(axis=0)
    col_max = matrix_raw.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1  # avoid division by zero
    matrix_norm = (matrix_raw - col_min) / col_range

    fig, ax = plt.subplots(figsize=(max(14, n_cols * 1.1), max(6, n_regimes * 0.8)),
                           facecolor=FACECOLOR)
    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    # Annotations with actual values
    for i in range(n_regimes):
        for j in range(n_cols):
            val = matrix_raw[i, j]
            norm_val = matrix_norm[i, j]
            color = "white" if norm_val > 0.6 else "black"
            # Format: integers for sojourn/pct, one decimal for fuel, two for LMP
            if j < 2:
                txt = f"{val:.1f}"
            elif j == n_cols - 1:
                txt = f"{val:.0f}"
            else:
                txt = f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color=color, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cols_display, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_regimes))
    ax.set_yticklabels([f"R{k}" for k in order], fontsize=10)
    ax.set_title("Regime Heatmap (B:FE+ToMATo, normalized per column, annotated with actual values)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Normalized [0,1]", shrink=0.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_regime_heatmap.png", dpi=DPI, facecolor=FACECOLOR,
                bbox_inches="tight")
    plt.close(fig)
    print("    Done.", flush=True)


def plot04_transition_matrix(order: list[int]):
    """Plot 4: Transition probability matrix for B:FE+ToMATo."""
    print("  Plotting 04_transition_matrix.png ...", flush=True)

    labels_df = load_labels(BEST_EXP, BEST_LABELS_DIR)
    if labels_df is None:
        print("    [SKIP] Labels not found.", flush=True)
        return

    labels = labels_df.sort_values("datetime")["cluster"].values
    n_regimes = len(order)
    label_to_idx = {k: i for i, k in enumerate(order)}

    # Count transitions
    trans_count = np.zeros((n_regimes, n_regimes), dtype=int)
    for t in range(len(labels) - 1):
        i = label_to_idx.get(labels[t])
        j = label_to_idx.get(labels[t + 1])
        if i is not None and j is not None:
            trans_count[i, j] += 1

    # Normalize rows -> probabilities
    row_sums = trans_count.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans_count / row_sums

    fig, ax = plt.subplots(figsize=(max(8, n_regimes * 0.9 + 2),
                                     max(7, n_regimes * 0.9 + 1)),
                           facecolor=FACECOLOR)
    im = ax.imshow(trans_prob, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    for i in range(n_regimes):
        for j in range(n_regimes):
            val = trans_prob[i, j]
            color = "white" if val > 0.5 else "black"
            txt = f"{val:.2f}" if val >= 0.005 else ""
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold")

    regime_labels = [f"R{k}" for k in order]
    ax.set_xticks(range(n_regimes))
    ax.set_xticklabels(regime_labels, fontsize=9)
    ax.set_yticks(range(n_regimes))
    ax.set_yticklabels(regime_labels, fontsize=9)
    ax.set_xlabel("Next regime", fontsize=11)
    ax.set_ylabel("Current regime", fontsize=11)
    ax.set_title("Transition Probability Matrix (B:FE+ToMATo, regimes ordered by mean LMP)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="P(next | current)", shrink=0.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_transition_matrix.png", dpi=DPI, facecolor=FACECOLOR,
                bbox_inches="tight")
    plt.close(fig)
    print("    Done.", flush=True)


def plot05_comparison_barplot(df_table: pd.DataFrame):
    """Plot 5: Grouped bar chart comparing 3 ToMATo configs on 4 metrics."""
    print("  Plotting 05_comparison_barplot.png ...", flush=True)

    if df_table is None or len(df_table) == 0:
        print("    [SKIP] No comparison data.", flush=True)
        return

    metrics = [
        ("eta_squared", "eta-squared (LMP variance explained)", True),
        ("silhouette", "Silhouette Score", True),
        ("transition_rate", "Transition Rate", False),
        ("bootstrap_ari_mean", "Bootstrap ARI (mean)", True),
    ]

    n_metrics = len(metrics)
    n_configs = len(df_table)
    config_names = df_table["config"].tolist()

    # Colors: one per config
    bar_colors = [
        "#2166ac",  # A:MOMENT ToMATo
        "#b2182b",  # B:FE ToMATo
        "#1b7837",  # C:COMBO ToMATo
    ]
    if n_configs > len(bar_colors):
        bar_colors = [CMAP(i) for i in range(n_configs)]

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6), facecolor=FACECOLOR)

    x = np.arange(n_configs)
    bar_width = 0.7

    for ax_idx, (col, title, higher_better) in enumerate(metrics):
        ax = axes[ax_idx]
        vals = df_table[col].values.astype(float)

        bars = ax.bar(x, vals, width=bar_width, color=bar_colors[:n_configs],
                      edgecolor="black", linewidth=0.5)

        # Value labels on top
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v + max(0.01, abs(v) * 0.02), f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        direction = "(higher=better)" if higher_better else "(lower=better)"
        ax.set_ylabel(f"{col} {direction}", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Configuration Comparison (3 representations, ToMATo clustering)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_comparison_barplot.png", dpi=DPI, facecolor=FACECOLOR,
                bbox_inches="tight")
    plt.close(fig)
    print("    Done.", flush=True)


def plot06_timeline_regimes(df_m: pd.DataFrame, order: list[int]):
    """Plot 6: Timeline of regimes (top: colored strip, bottom: LMP colored by regime)."""
    print("  Plotting 06_timeline_regimes.png ...", flush=True)

    n_regimes = len(order)
    colors = regime_colors(n_regimes)
    # Map cluster id -> ordered index for color
    label_to_idx = {k: i for i, k in enumerate(order)}

    df_sorted = df_m.sort_values("datetime").copy()
    ts = df_sorted["datetime"].values
    labels = df_sorted["cluster"].values
    lmp = df_sorted["lmp"].values
    color_idx = np.array([label_to_idx[c] for c in labels])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 7), facecolor=FACECOLOR,
                                    sharex=True,
                                    gridspec_kw={"height_ratios": [1, 4]})

    # Top: regime strip
    cmap_disc = mcolors.ListedColormap(colors)
    ax1.scatter(ts, np.zeros(len(ts)), c=color_idx, cmap=cmap_disc,
                s=0.3, alpha=0.9, vmin=0, vmax=n_regimes - 1)
    ax1.set_yticks([])
    ax1.set_ylabel("Regime", fontsize=10)
    ax1.set_title("Regime Timeline + LMP (B:FE+ToMATo, 2021-2025)",
                  fontsize=13, fontweight="bold")

    # Legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=colors[i], markersize=8,
               label=f"R{order[i]}")
               for i in range(n_regimes)]
    ax1.legend(handles=handles, loc="upper right", ncol=min(n_regimes, 10),
               fontsize=7, framealpha=0.9)

    # Bottom: LMP colored by regime
    for i, k in enumerate(order):
        mask = labels == k
        ax2.scatter(ts[mask], lmp[mask], s=0.3, alpha=0.4, color=colors[i],
                    rasterized=True)
    ax2.plot(ts, lmp, lw=0.15, color="black", alpha=0.3)
    ax2.set_ylabel("LMP ($/MWh)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_timeline_regimes.png", dpi=DPI, facecolor=FACECOLOR,
                bbox_inches="tight")
    plt.close(fig)
    print("    Done.", flush=True)


def section3_plots(df_table: pd.DataFrame):
    """Generate all 6 plots."""
    print("\n" + "=" * 70)
    print("  SECTION 3: ILR analysis plots")
    print("=" * 70, flush=True)

    df_m, order = _get_best_merged()
    if df_m is None:
        print("  [ERROR] Could not load best config data!", flush=True)
        return

    print(f"  Best config: {BEST_NAME}, K={len(order)}, "
          f"order by LMP: {order}", flush=True)
    print()

    plot01_ilr_by_regime(df_m, order)
    plot02_fuel_mix_by_regime(df_m, order)
    plot03_regime_heatmap(df_m, order)
    plot04_transition_matrix(order)
    plot05_comparison_barplot(df_table)
    plot06_timeline_regimes(df_m, order)


# =============================================================================
# SECTION 4: CROSS-APPROACH ARI
# =============================================================================

def section4_cross_ari():
    """Compute pairwise ARI between the 3 ToMATo configs."""
    print("\n" + "=" * 70)
    print("  SECTION 4: Cross-approach pairwise ARI (ToMATo configs)")
    print("=" * 70, flush=True)

    tomato_configs = [
        ("A:MOMENT+ToMATo", "C", "step04"),
        ("B:FE+ToMATo", "FE", "step04"),
        ("C:COMBO+ToMATo", "COMBO", "step04"),
    ]

    # Load all labels
    label_dfs = {}
    for name, exp, ldir in tomato_configs:
        df = load_labels(exp, ldir)
        if df is not None:
            label_dfs[name] = df
            print(f"  Loaded {name}: {len(df)} windows", flush=True)
        else:
            print(f"  [SKIP] {name}: labels not found", flush=True)

    names = list(label_dfs.keys())
    n = len(names)

    if n < 2:
        print("  Not enough configs for pairwise ARI.", flush=True)
        return

    # Pairwise ARI (align on common datetimes)
    ari_matrix = np.full((n, n), np.nan)
    ari_results = []

    for i in range(n):
        ari_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            df_a = label_dfs[names[i]]
            df_b = label_dfs[names[j]]

            # Inner join on datetime
            merged = df_a.merge(df_b, on="datetime", suffixes=("_a", "_b"))
            if len(merged) == 0:
                print(f"  [WARN] No overlap between {names[i]} and {names[j]}",
                      flush=True)
                continue

            ari = adjusted_rand_score(merged["cluster_a"], merged["cluster_b"])
            ari_matrix[i, j] = ari
            ari_matrix[j, i] = ari
            ari_results.append({
                "config_1": names[i],
                "config_2": names[j],
                "n_common": len(merged),
                "ARI": round(ari, 4),
            })
            print(f"  {names[i]} vs {names[j]}: ARI = {ari:.4f} "
                  f"(n={len(merged)})", flush=True)

    # Print matrix
    print("\n  Pairwise ARI matrix:")
    print(f"  {'':>22s}", end="")
    for name in names:
        print(f" {name:>22s}", end="")
    print()
    for i, name_i in enumerate(names):
        print(f"  {name_i:>22s}", end="")
        for j in range(n):
            v = ari_matrix[i, j]
            print(f" {v:>22.4f}", end="")
        print()

    # Save
    df_ari = pd.DataFrame(ari_results)
    path = OUT_DIR / "cross_approach_ari.csv"
    df_ari.to_csv(path, index=False, float_format="%.4f")
    print(f"\n  Saved: {path}", flush=True)

    # Also save the matrix
    df_matrix = pd.DataFrame(ari_matrix, index=names, columns=names)
    matrix_path = OUT_DIR / "cross_approach_ari_matrix.csv"
    df_matrix.to_csv(matrix_path, float_format="%.4f")
    print(f"  Saved: {matrix_path}", flush=True)

    return df_ari


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  NEPOOL TDA — Post-Analysis")
    print(f"  Output: {OUT_DIR}")
    print("=" * 70, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Section 1
    df_table = section1_comparison_table()

    # Section 2
    df_char = section2_regime_characterization()

    # Section 3
    section3_plots(df_table)

    # Section 4
    section4_cross_ari()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  Post-analysis complete in {elapsed:.1f}s")
    print(f"  All outputs saved to: {OUT_DIR}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
