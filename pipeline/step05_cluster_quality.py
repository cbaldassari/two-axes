"""
step05_cluster_quality.py
=========================
NEPOOL TDA Pipeline — Step 05: Cluster Quality Evaluation

Valuta i cluster ToMATo da due prospettive indipendenti:

A) GEOMETRICA — qualità nello spazio PCA dove opera ToMATo
   1. Silhouette score           separazione inter-cluster / coesione intra-cluster
   2. Davies-Bouldin index       rapporto dispersione/distanza tra centroidi (lower=better)
   3. Calinski-Harabasz index    varianza inter/intra (higher=better)
   4. Dunn index                 min dist inter-cluster / max diam intra-cluster

B) ECONOMICA — i cluster hanno significato nel mercato elettrico?
   1. η² (eta-squared)           ANOVA: quanta varianza LMP è spiegata dai regimi
   2. Pairwise LMP separation    Tukey HSD: coppie di cluster con prezzi distinguibili
   3. Sojourn time               permanenza media in un regime (in finestre di 6h)
   4. Transition rate            frequenza di cambio regime (transizioni / finestre)
   5. Cramér's V (season)        associazione regime ↔ stagione
   6. Cramér's V (hour block)    associazione regime ↔ fascia oraria

Input  : results/exp_{X}/step04/labels.parquet
         results/exp_{X}/step03/pca_embeddings.parquet
         results/preprocessed.parquet
Output : results/exp_{X}/step05/
           quality_report.json
           01_silhouette.png          silhouette per cluster
           02_geometric_summary.png   barplot metriche geometriche
           03_lmp_by_regime.png       boxplot LMP per regime
           04_sojourn.png             distribuzione permanenza
           05_timeline_lmp.png        timeline regime + LMP sovrapposto
           06_seasonal_heatmap.png    heatmap regime × mese
           07_hourly_heatmap.png      heatmap regime × fascia oraria

Uso
---
  python pipeline/step05_cluster_quality.py --exp A
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, chi2_contingency
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

_parser = argparse.ArgumentParser(description="Step 05 — Cluster quality")
_parser.add_argument("--exp", default="A", choices=["A", "B", "C", "D", "E", "FE", "FT", "COMBO", "D2"])
_parser.add_argument("--labels-dir", default=None,
                     help="Override labels directory (e.g. step04_ward). Default: step04")
_parser.add_argument("--out-suffix", default=None,
                     help="Suffix for output dir (e.g. _ward -> step05_ward). Default: none")
_args = _parser.parse_args()
EXPERIMENT = _args.exp

RESULTS_DIR = Path("results")
_labels_dir = _args.labels_dir or "step04"
_out_suffix = _args.out_suffix or ""
PCA_PATH    = RESULTS_DIR / f"exp_{EXPERIMENT}" / "step03" / "pca_embeddings.parquet"
LABELS_PATH = RESULTS_DIR / f"exp_{EXPERIMENT}" / _labels_dir / "labels.parquet"
PREPROC_PATH = RESULTS_DIR / "preprocessed.parquet"
OUT_DIR     = RESULTS_DIR / f"exp_{EXPERIMENT}" / f"step05{_out_suffix}"

SEED = 42
DPI  = 150
CMAP = plt.get_cmap("tab20")

HOUR_BLOCKS = {
    "notte (0-6)": (0, 6),
    "mattina (6-12)": (6, 12),
    "pomeriggio (12-18)": (12, 18),
    "sera (18-24)": (18, 24),
}

SEASONS = {
    "inverno": [12, 1, 2],
    "primavera": [3, 4, 5],
    "estate": [6, 7, 8],
    "autunno": [9, 10, 11],
}


def regime_colors(n):
    return [CMAP(i % 20) for i in range(n)]


# =============================================================================
# LOAD
# =============================================================================

def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Carica PCA embeddings, etichette, e dati preprocessati."""
    # PCA
    df_pca = pd.read_parquet(PCA_PATH)
    ts_pca = pd.to_datetime(df_pca["datetime"])
    E = df_pca.drop(columns=["datetime"]).values.astype(np.float32)

    # Labels
    df_lab = pd.read_parquet(LABELS_PATH)
    labels = df_lab["cluster"].values

    # Preprocessed (per LMP e timestamp orari)
    df_pre = pd.read_parquet(PREPROC_PATH)
    df_pre["datetime"] = pd.to_datetime(df_pre["datetime"])

    # Join labels con preprocessed via timestamp
    df_lab["datetime"] = pd.to_datetime(df_lab["datetime"])
    df_merged = df_pre.merge(df_lab, on="datetime", how="inner")

    print(f"  PCA: {E.shape}  |  Labels: {len(labels)}  |  Merged: {len(df_merged)}",
          flush=True)
    return E, labels, df_merged


# =============================================================================
# A) METRICHE GEOMETRICHE
# =============================================================================

def geometric_metrics(E: np.ndarray, labels: np.ndarray) -> dict:
    """Calcola metriche geometriche nello spazio PCA."""
    print("\n  [A] Metriche geometriche...", flush=True)

    n_clusters = len(np.unique(labels))

    # Silhouette
    sil_avg = float(silhouette_score(E, labels, random_state=SEED))
    sil_samples = silhouette_samples(E, labels)
    sil_per_cluster = {}
    for k in sorted(np.unique(labels)):
        sil_per_cluster[int(k)] = float(np.mean(sil_samples[labels == k]))

    # Davies-Bouldin (lower = better)
    db = float(davies_bouldin_score(E, labels))

    # Calinski-Harabasz (higher = better)
    ch = float(calinski_harabasz_score(E, labels))

    # Dunn index: min(inter-cluster dist) / max(intra-cluster diameter)
    dunn = _dunn_index(E, labels)

    geo = {
        "silhouette_avg": round(sil_avg, 4),
        "silhouette_per_cluster": {str(k): round(v, 4) for k, v in sil_per_cluster.items()},
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 4),
        "dunn_index": round(dunn, 6),
    }

    print(f"    Silhouette avg : {sil_avg:.4f}", flush=True)
    print(f"    Davies-Bouldin : {db:.4f}", flush=True)
    print(f"    Calinski-Harab.: {ch:.1f}", flush=True)
    print(f"    Dunn index     : {dunn:.6f}", flush=True)

    return geo


def _dunn_index(E: np.ndarray, labels: np.ndarray) -> float:
    """Dunn index approssimato con centroidi (per efficienza)."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return 0.0

    # Centroidi e diametri
    centroids = np.array([E[labels == k].mean(axis=0) for k in unique_labels])
    diameters = []
    for k in unique_labels:
        pts = E[labels == k]
        if len(pts) < 2:
            diameters.append(0.0)
            continue
        # Diametro approssimato: max distanza dal centroide × 2
        dists = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
        diameters.append(float(2 * np.max(dists)))

    max_diam = max(diameters) if diameters else 1.0
    if max_diam == 0:
        return 0.0

    # Min distanza inter-cluster (tra centroidi)
    min_inter = float("inf")
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            if d < min_inter:
                min_inter = d

    return min_inter / max_diam


# =============================================================================
# B) METRICHE ECONOMICHE
# =============================================================================

def economic_metrics(labels: np.ndarray, df: pd.DataFrame) -> dict:
    """Calcola metriche economiche sui dati di mercato.

    labels : array dalla sequenza originale di finestre (step04 labels.parquet)
             — usato per sojourn/transition (risoluzione finestra, stride 6h)
    df     : merge labels×preprocessed — usato per LMP, stagione, ora
    """
    print("\n  [B] Metriche economiche...", flush=True)

    n_clusters = len(np.unique(labels))
    lmp = df["lmp"].values
    clusters_merged = df["cluster"].values

    # --- 1. η² (eta-squared): varianza LMP spiegata dai regimi ---
    groups = [lmp[clusters_merged == k] for k in sorted(np.unique(clusters_merged))]
    ss_total = np.sum((lmp - lmp.mean()) ** 2)
    ss_between = sum(len(g) * (g.mean() - lmp.mean()) ** 2 for g in groups)
    eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0

    # ANOVA F-test
    f_stat, f_pval = f_oneway(*groups)
    # Kruskal-Wallis (non-parametrico)
    h_stat, h_pval = kruskal(*groups)

    print(f"    eta2 (LMP)     : {eta2:.4f}", flush=True)
    print(f"    ANOVA F        : {f_stat:.1f}  (p={f_pval:.2e})", flush=True)
    print(f"    Kruskal-Wallis : {h_stat:.1f}  (p={h_pval:.2e})", flush=True)

    # --- 2. LMP per regime (summary stats) ---
    lmp_stats = {}
    for k in sorted(np.unique(clusters_merged)):
        g = lmp[clusters_merged == k]
        lmp_stats[int(k)] = {
            "mean": round(float(g.mean()), 2),
            "median": round(float(np.median(g)), 2),
            "std": round(float(g.std()), 2),
            "p5": round(float(np.percentile(g, 5)), 2),
            "p95": round(float(np.percentile(g, 95)), 2),
            "n": int(len(g)),
        }
    print(f"    LMP per regime :", flush=True)
    for k, s in lmp_stats.items():
        print(f"      R{k}: mean={s['mean']:>7.2f}  std={s['std']:>6.2f}  "
              f"[p5={s['p5']:>7.2f}, p95={s['p95']:>7.2f}]  n={s['n']:,}", flush=True)

    # --- 3. Pairwise Tukey HSD ---
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(lmp, clusters_merged, alpha=0.05)
        n_pairs = len(tukey.reject)
        n_significant = int(np.sum(tukey.reject))
        pairwise_sep = n_significant / n_pairs if n_pairs > 0 else 0.0
    except ImportError:
        pairwise_sep = None
        n_significant = None
        n_pairs = None

    if pairwise_sep is not None:
        print(f"    Tukey HSD      : {n_significant}/{n_pairs} coppie significative "
              f"({pairwise_sep:.1%})", flush=True)

    # --- 4. Sojourn time (calcolato sulla sequenza di FINESTRE, non righe orarie) ---
    # labels è la sequenza originale delle finestre (stride 6h, ~7064 punti)
    STRIDE_H = 6
    transitions = int(np.sum(labels[1:] != labels[:-1]))
    n_windows = len(labels)
    transition_rate = float(transitions / (n_windows - 1)) if n_windows > 1 else 0.0

    # Run lengths in finestre
    sojourns = []
    current_len = 1
    for i in range(1, n_windows):
        if labels[i] == labels[i - 1]:
            current_len += 1
        else:
            sojourns.append(current_len)
            current_len = 1
    sojourns.append(current_len)
    sojourns = np.array(sojourns)
    sojourn_mean = float(sojourns.mean())
    sojourn_median = float(np.median(sojourns))

    # Sojourn per cluster
    sojourn_per_cluster = {}
    for k in sorted(np.unique(labels)):
        runs = []
        current_len = 0
        for c in labels:
            if c == k:
                current_len += 1
            else:
                if current_len > 0:
                    runs.append(current_len)
                current_len = 0
        if current_len > 0:
            runs.append(current_len)
        if runs:
            sojourn_per_cluster[int(k)] = {
                "mean_windows": round(float(np.mean(runs)), 1),
                "mean_hours": round(float(np.mean(runs)) * STRIDE_H, 0),
                "median_windows": round(float(np.median(runs)), 1),
                "max_windows": int(np.max(runs)),
            }

    print(f"    Transition rate: {transition_rate:.4f} "
          f"({transitions} transizioni su {n_windows} finestre)", flush=True)
    print(f"    Sojourn medio  : {sojourn_mean:.1f} finestre "
          f"({sojourn_mean * STRIDE_H:.0f}h)", flush=True)

    # --- 5. Cramér's V con stagione ---
    df_tmp = df.copy()
    month = df_tmp["datetime"].dt.month
    df_tmp["season"] = "inverno"
    for sname, months in SEASONS.items():
        df_tmp.loc[month.isin(months), "season"] = sname

    ct_season = pd.crosstab(df_tmp["cluster"], df_tmp["season"])
    cramer_season = _cramers_v(ct_season.values)

    print(f"    Cramér's V (season): {cramer_season:.4f}", flush=True)

    # --- 6. Cramér's V con fascia oraria ---
    hour = df_tmp["datetime"].dt.hour
    df_tmp["hour_block"] = "notte (0-6)"
    for bname, (h0, h1) in HOUR_BLOCKS.items():
        df_tmp.loc[(hour >= h0) & (hour < h1), "hour_block"] = bname

    ct_hour = pd.crosstab(df_tmp["cluster"], df_tmp["hour_block"])
    cramer_hour = _cramers_v(ct_hour.values)

    print(f"    Cramér's V (hour)  : {cramer_hour:.4f}", flush=True)

    econ = {
        "eta_squared": round(eta2, 4),
        "anova_F": round(float(f_stat), 2),
        "anova_p": float(f_pval),
        "kruskal_H": round(float(h_stat), 2),
        "kruskal_p": float(h_pval),
        "lmp_per_regime": lmp_stats,
        "pairwise_separation": round(pairwise_sep, 4) if pairwise_sep is not None else None,
        "pairwise_significant": n_significant,
        "pairwise_total": n_pairs,
        "transition_rate": round(transition_rate, 4),
        "transitions": int(transitions),
        "sojourn_mean_windows": round(sojourn_mean, 1),
        "sojourn_mean_hours": round(sojourn_mean * STRIDE_H, 0),
        "sojourn_median_windows": round(sojourn_median, 1),
        "sojourn_per_cluster": sojourn_per_cluster,
        "cramer_v_season": round(cramer_season, 4),
        "cramer_v_hour_block": round(cramer_hour, 4),
    }

    return econ


def _cramers_v(contingency: np.ndarray) -> float:
    """Cramér's V con correzione di Bergsma (bias-corrected)."""
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum()
    r, k = contingency.shape
    phi2 = chi2 / n
    # Bias correction
    phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    k_corr = k - (k - 1) ** 2 / (n - 1)
    r_corr = r - (r - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


# =============================================================================
# PLOTS
# =============================================================================

def plot_silhouette(E: np.ndarray, labels: np.ndarray):
    """Silhouette plot per cluster."""
    sil_vals = silhouette_samples(E, labels)
    sil_avg = float(np.mean(sil_vals))
    unique_labels = sorted(np.unique(labels))
    n_clusters = len(unique_labels)
    colors = regime_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 0
    for i, k in enumerate(unique_labels):
        cluster_sils = np.sort(sil_vals[labels == k])
        size = len(cluster_sils)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sils,
                         facecolor=colors[i], alpha=0.7, label=f"R{k}")
        y_lower = y_upper + 5

    ax.axvline(sil_avg, color="red", ls="--", lw=1.5,
               label=f"Media={sil_avg:.3f}")
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Punti (ordinati)")
    ax.set_title(f"Silhouette — Exp {EXPERIMENT} (K={n_clusters})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_silhouette.png", dpi=DPI)
    plt.close(fig)
    print(f"  01_silhouette.png", flush=True)


def plot_geometric_summary(geo: dict):
    """Barplot riassuntivo metriche geometriche."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Silhouette per cluster
    sil = geo["silhouette_per_cluster"]
    ks = sorted(sil.keys(), key=int)
    vals = [sil[k] for k in ks]
    colors = regime_colors(len(ks))
    axes[0].bar([f"R{k}" for k in ks], vals, color=colors, edgecolor="k", lw=0.5)
    axes[0].axhline(geo["silhouette_avg"], color="red", ls="--", lw=1)
    axes[0].set_title(f"Silhouette (avg={geo['silhouette_avg']:.3f})")
    axes[0].set_ylabel("Silhouette")

    # Davies-Bouldin
    axes[1].bar(["DB"], [geo["davies_bouldin"]], color="#e07b39", edgecolor="k")
    axes[1].set_title(f"Davies-Bouldin ({geo['davies_bouldin']:.3f})")
    axes[1].set_ylabel("DB (lower=better)")

    # Calinski-Harabasz
    axes[2].bar(["CH"], [geo["calinski_harabasz"]], color="#4a90d9", edgecolor="k")
    axes[2].set_title(f"Calinski-Harabasz ({geo['calinski_harabasz']:.0f})")
    axes[2].set_ylabel("CH (higher=better)")

    # Dunn
    axes[3].bar(["Dunn"], [geo["dunn_index"]], color="#6abf69", edgecolor="k")
    axes[3].set_title(f"Dunn ({geo['dunn_index']:.4f})")
    axes[3].set_ylabel("Dunn (higher=better)")

    fig.suptitle(f"Metriche geometriche — Exp {EXPERIMENT}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_geometric_summary.png", dpi=DPI)
    plt.close(fig)
    print(f"  02_geometric_summary.png", flush=True)


def plot_lmp_by_regime(df: pd.DataFrame):
    """Boxplot LMP per regime."""
    n_clusters = len(df["cluster"].unique())
    colors = regime_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df.loc[df["cluster"] == k, "lmp"].values for k in sorted(df["cluster"].unique())]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels([f"R{k}" for k in sorted(df["cluster"].unique())])
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title(f"LMP per regime — Exp {EXPERIMENT}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_lmp_by_regime.png", dpi=DPI)
    plt.close(fig)
    print(f"  03_lmp_by_regime.png", flush=True)


def plot_sojourn(df: pd.DataFrame):
    """Distribuzione delle permanenze per regime."""
    clusters = df["cluster"].values
    unique_labels = sorted(np.unique(clusters))
    n_clusters = len(unique_labels)
    colors = regime_colors(n_clusters)

    # Calcola run lengths per cluster
    runs_per_cluster = {k: [] for k in unique_labels}
    current_label = clusters[0]
    current_len = 1
    for i in range(1, len(clusters)):
        if clusters[i] == current_label:
            current_len += 1
        else:
            runs_per_cluster[current_label].append(current_len)
            current_label = clusters[i]
            current_len = 1
    runs_per_cluster[current_label].append(current_len)

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = []
    for i, k in enumerate(unique_labels):
        runs = np.array(runs_per_cluster[k]) * 6  # converti in ore
        if len(runs) > 0:
            bp = ax.boxplot([runs], positions=[i], patch_artist=True,
                           showfliers=True, widths=0.6)
            bp["boxes"][0].set_facecolor(colors[i])
            bp["boxes"][0].set_alpha(0.7)
        positions.append(i)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"R{k}" for k in unique_labels])
    ax.set_ylabel("Permanenza (ore)")
    ax.set_title(f"Sojourn time per regime — Exp {EXPERIMENT}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_sojourn.png", dpi=DPI)
    plt.close(fig)
    print(f"  04_sojourn.png", flush=True)


def plot_timeline_lmp(df: pd.DataFrame):
    """Timeline regime con LMP sovrapposto."""
    n_clusters = len(df["cluster"].unique())
    colors = regime_colors(n_clusters)
    cmap_disc = mcolors.ListedColormap(colors)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 3]})

    ts = df["datetime"].values
    labels = df["cluster"].values
    lmp = df["lmp"].values

    # Timeline regimi
    ax1.scatter(ts, np.zeros_like(labels), c=labels, cmap=cmap_disc,
                s=0.5, alpha=0.8, vmin=0, vmax=n_clusters - 1)
    ax1.set_yticks([])
    ax1.set_title(f"Timeline regimi + LMP — Exp {EXPERIMENT}")

    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=colors[k], markersize=8, label=f"R{k}")
               for k in range(n_clusters)]
    ax1.legend(handles=handles, loc="upper right", ncol=n_clusters, fontsize=7)

    # LMP
    ax2.plot(ts, lmp, lw=0.3, color="black", alpha=0.5)
    # Colora sfondo per regime
    for k in range(n_clusters):
        mask = labels == k
        ax2.scatter(ts[mask], lmp[mask], s=0.3, alpha=0.3, color=colors[k])
    ax2.set_ylabel("LMP ($/MWh)")
    ax2.set_xlabel("Data")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_timeline_lmp.png", dpi=DPI)
    plt.close(fig)
    print(f"  05_timeline_lmp.png", flush=True)


def plot_seasonal_heatmap(df: pd.DataFrame):
    """Heatmap regime × mese."""
    df_tmp = df.copy()
    df_tmp["month"] = df_tmp["datetime"].dt.month
    ct = pd.crosstab(df_tmp["cluster"], df_tmp["month"], normalize="columns")

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(ct.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
                         "Lug", "Ago", "Set", "Ott", "Nov", "Dic"])
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels([f"R{k}" for k in ct.index])
    ax.set_title(f"Distribuzione regime per mese (normalizzata) — Exp {EXPERIMENT}")
    plt.colorbar(im, ax=ax, label="Proporzione")

    # Annota celle
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = ct.values[i, j]
            color = "white" if val > 0.5 * ct.values.max() else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_seasonal_heatmap.png", dpi=DPI)
    plt.close(fig)
    print(f"  06_seasonal_heatmap.png", flush=True)


def plot_hourly_heatmap(df: pd.DataFrame):
    """Heatmap regime × ora del giorno."""
    df_tmp = df.copy()
    df_tmp["hour"] = df_tmp["datetime"].dt.hour
    ct = pd.crosstab(df_tmp["cluster"], df_tmp["hour"], normalize="columns")

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(ct.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels([f"R{k}" for k in ct.index])
    ax.set_xlabel("Ora")
    ax.set_title(f"Distribuzione regime per ora (normalizzata) — Exp {EXPERIMENT}")
    plt.colorbar(im, ax=ax, label="Proporzione")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_hourly_heatmap.png", dpi=DPI)
    plt.close(fig)
    print(f"  07_hourly_heatmap.png", flush=True)


# =============================================================================
# SAVE
# =============================================================================

def save_report(geo: dict, econ: dict):
    report = {
        "experiment": EXPERIMENT,
        "geometric": geo,
        "economic": econ,
    }
    report_path = OUT_DIR / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Salvato: {report_path}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 65)
    print(f"  NEPOOL TDA — Step 05: Cluster Quality Evaluation")
    print(f"  Experiment : {EXPERIMENT}")
    print("=" * 65, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    E, labels, df_merged = load_data()

    # 2. Geometric
    geo = geometric_metrics(E, labels)

    # 3. Economic
    econ = economic_metrics(labels, df_merged)

    # 4. Plots
    print(f"\n  Plots...", flush=True)
    plot_silhouette(E, labels)
    plot_geometric_summary(geo)
    plot_lmp_by_regime(df_merged)
    plot_sojourn(df_merged)
    plot_timeline_lmp(df_merged)
    plot_seasonal_heatmap(df_merged)
    plot_hourly_heatmap(df_merged)

    # 5. Save
    print(f"\n  Salvataggio...", flush=True)
    save_report(geo, econ)

    elapsed = time.time() - t0
    print(f"\n  Completato in {elapsed:.1f}s")
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
