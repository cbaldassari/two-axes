"""
make_overview.py
================
Genera un'immagine di riepilogo dell'intera pipeline NEPOOL TDA.

  python pipeline/make_overview.py

Output: results/overview.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

RESULTS = Path("results")
EXPS = ["A", "B", "C", "D"]
DPI = 180

# Colors
C_BG       = "#f7f7f7"
C_HEADER   = "#1a1a2e"
C_STEP     = ["#16213e", "#0f3460", "#533483", "#e94560"]
C_EXP      = {"A": "#4a90d9", "B": "#e07b39", "C": "#6abf69", "D": "#b04040"}
C_REGIME   = plt.get_cmap("tab20")


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    # ── Collect data ──────────────────────────────────────────────────────
    pca_reports = {}
    tomato_reports = {}
    quality_reports = {}
    diag_reports = {}
    labels_data = {}
    preproc = None

    if (RESULTS / "preprocessed.parquet").exists():
        preproc = pd.read_parquet(RESULTS / "preprocessed.parquet")
        preproc["datetime"] = pd.to_datetime(preproc["datetime"])

    for exp in EXPS:
        pca_reports[exp] = load_json(RESULTS / f"exp_{exp}" / "step03" / "pca_report.json")
        tomato_reports[exp] = load_json(RESULTS / f"exp_{exp}" / "step04" / "tomato_report.json")
        quality_reports[exp] = load_json(RESULTS / f"exp_{exp}" / "step05" / "quality_report.json")
        diag_reports[exp] = load_json(RESULTS / f"exp_{exp}" / "step06" / "diagnostics_report.json")
        lab_path = RESULTS / f"exp_{exp}" / "step04" / "labels.parquet"
        if lab_path.exists():
            labels_data[exp] = pd.read_parquet(lab_path)
            labels_data[exp]["datetime"] = pd.to_datetime(labels_data[exp]["datetime"])

    has_pca = any(v is not None for v in pca_reports.values())
    has_tomato = any(v is not None for v in tomato_reports.values())
    has_quality = any(v is not None for v in quality_reports.values())
    has_diag = any(v is not None for v in diag_reports.values())
    has_labels = len(labels_data) > 0

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 28), facecolor=C_BG)

    # Main grid: 6 rows
    gs = gridspec.GridSpec(7, 2, figure=fig, hspace=0.35, wspace=0.25,
                           left=0.05, right=0.95, top=0.95, bottom=0.03,
                           height_ratios=[0.6, 2.0, 1.8, 1.8, 1.8, 1.8, 1.5])

    # ── ROW 0: Title + pipeline diagram ───────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(C_HEADER)
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis("off")

    ax_title.text(0.5, 0.7, "NEPOOL TDA — Regime Detection Pipeline Overview",
                  ha="center", va="center", fontsize=22, fontweight="bold",
                  color="white", family="monospace")
    ax_title.text(0.5, 0.25,
                  "ISO New England  ·  2021–2025  ·  43,794h  ·  "
                  "Chronos-2 → PCA (Marchenko-Pastur) → ToMATo (TDA)",
                  ha="center", va="center", fontsize=11, color="#aaaaaa",
                  family="monospace")

    # ── ROW 1: Pipeline flow + experiment design ──────────────────────────
    ax_pipe = fig.add_subplot(gs[1, 0])
    ax_pipe.axis("off")
    ax_pipe.set_title("Pipeline Flow", fontsize=14, fontweight="bold", loc="left")

    steps = [
        ("STEP 01", "Preprocessing",
         "LMP → arcsinh → log_return\n"
         "MSTL(24h+168h) → resid_lr, resid_arcsinh\n"
         "Fuel(8) → ILR(7) → MSTL → resid_ilr_*"),
        ("STEP 02", "Chronos-2 Embedding  [GPU]",
         "720h window, 6h stride → ~7064 finestre\n"
         "768D (price) or 775D (price ‖ ILR detrended)"),
        ("STEP 03", "PCA Reduction",
         "Marchenko-Pastur threshold\n"
         "768D → 34-48 PC (~89-90% var)"),
        ("STEP 04", "ToMATo Clustering",
         "logDTM density + k-NN graph\n"
         "K via max-gap in persistence barcode"),
        ("STEP 05", "Quality Evaluation",
         "Geometric: Sil, DB, CH, Dunn\n"
         "Economic: η², Tukey, sojourn, Cramér V"),
        ("STEP 06", "Embedding Diagnostics",
         "Opacity: PC↔features, RF importance\n"
         "Transfer: Mantel, NN coherence"),
    ]

    box_h = 0.13
    gap = 0.025
    y_start = 0.92
    colors_step = ["#2c3e50", "#2c3e50", "#8e44ad", "#c0392b", "#27ae60", "#2980b9"]

    for i, (label, title, desc) in enumerate(steps):
        y = y_start - i * (box_h + gap)
        rect = mpatches.FancyBboxPatch(
            (0.02, y - box_h), 0.96, box_h,
            boxstyle="round,pad=0.01",
            facecolor=colors_step[i], alpha=0.12,
            edgecolor=colors_step[i], linewidth=1.5,
        )
        ax_pipe.add_patch(rect)
        ax_pipe.text(0.04, y - 0.01, label,
                     fontsize=8, fontweight="bold", color=colors_step[i],
                     va="top", family="monospace")
        ax_pipe.text(0.16, y - 0.01, title,
                     fontsize=10, fontweight="bold", color="#1a1a1a", va="top")
        ax_pipe.text(0.16, y - 0.045, desc,
                     fontsize=7.5, color="#555555", va="top", linespacing=1.4)

        if i < len(steps) - 1:
            ax_pipe.annotate("", xy=(0.5, y - box_h - gap * 0.1),
                           xytext=(0.5, y - box_h + gap * 0.1),
                           arrowprops=dict(arrowstyle="->", color="#999999", lw=1.5))

    # ── Experiment design table ───────────────────────────────────────────
    ax_exp = fig.add_subplot(gs[1, 1])
    ax_exp.axis("off")
    ax_exp.set_title("Experiment Design (2×2 Factorial)", fontsize=14,
                     fontweight="bold", loc="left")

    table_data = [
        ["Exp", "Chronos-2 input", "ILR detrended", "Dim"],
        ["A", "mstl_resid_lr", "—", "768D"],
        ["B", "mstl_resid_lr", "concat", "775D"],
        ["C", "mstl_resid_arcsinh", "—", "768D"],
        ["D", "mstl_resid_arcsinh", "concat", "775D"],
    ]

    if has_pca:
        table_data[0].append("PC (MP)")
        for i, exp in enumerate(EXPS):
            r = pca_reports[exp]
            table_data[i + 1].append(
                f"{r['n_components']}  ({r['variance_explained']:.0%})" if r else "—")

    if has_tomato:
        table_data[0].append("K (ToMATo)")
        for i, exp in enumerate(EXPS):
            r = tomato_reports[exp]
            table_data[i + 1].append(str(r["n_clusters"]) if r else "—")

    tbl = ax_exp.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="upper center",
        cellLoc="center",
        bbox=[0.0, 0.45, 1.0, 0.45],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Color header
    for j in range(len(table_data[0])):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Color experiment rows
    for i, exp in enumerate(EXPS):
        tbl[(i + 1, 0)].set_facecolor(C_EXP[exp])
        tbl[(i + 1, 0)].set_text_props(color="white", fontweight="bold")

    # Design rationale
    rationale = (
        "Design rationale:\n"
        "• Factor 1: shock type — log-return (rendimento) vs arcsinh (livello)\n"
        "• Factor 2: fuel mix — without vs with ILR detrended (MSTL residuals)\n"
        "• ILR destagionalizzate: catturano anomalie strutturali, non cicli stagionali\n"
        "• Chronos-2: foundation model, 720h context, normalizzazione interna per finestra"
    )
    ax_exp.text(0.0, 0.35, rationale, fontsize=8, color="#333333", va="top",
                linespacing=1.6)

    # ── ROW 2: LMP timeseries + PCA scree ─────────────────────────────────
    # LMP
    ax_lmp = fig.add_subplot(gs[2, 0])
    if preproc is not None:
        ax_lmp.plot(preproc["datetime"], preproc["lmp"], lw=0.2, color="#2c6fad",
                    alpha=0.6)
        p99 = np.percentile(preproc["lmp"], 99)
        ax_lmp.axhline(p99, color="red", ls="--", lw=0.8, alpha=0.7)
        ax_lmp.set_title("LMP Time Series (MA Hub, $/MWh)", fontsize=12,
                         fontweight="bold", loc="left")
        ax_lmp.set_ylabel("$/MWh")
        ax_lmp.tick_params(labelsize=8)
    else:
        ax_lmp.text(0.5, 0.5, "preprocessed.parquet not found",
                    ha="center", va="center")
        ax_lmp.set_title("LMP Time Series", fontsize=12, fontweight="bold", loc="left")

    # PCA variance
    ax_pca = fig.add_subplot(gs[2, 1])
    if has_pca:
        for exp in EXPS:
            r = pca_reports[exp]
            if r and "variance_explained_cumulative" in r:
                cumvar = r["variance_explained_cumulative"]
                ax_pca.plot(range(1, len(cumvar) + 1), cumvar, "o-", ms=3,
                           color=C_EXP[exp], label=f"Exp {exp} ({r['n_components']} PC)",
                           lw=1.5)
        ax_pca.axhline(0.9, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax_pca.set_xlabel("Componenti")
        ax_pca.set_ylabel("Varianza cumulata")
        ax_pca.set_title("PCA — Varianza cumulata spiegata", fontsize=12,
                         fontweight="bold", loc="left")
        ax_pca.legend(fontsize=8)
        ax_pca.set_ylim(0, 1.05)
        ax_pca.tick_params(labelsize=8)
    else:
        ax_pca.text(0.5, 0.5, "PCA reports not found", ha="center", va="center")
        ax_pca.set_title("PCA", fontsize=12, fontweight="bold", loc="left")

    # ── ROW 3: ToMATo persistence + cluster sizes ─────────────────────────
    ax_bar = fig.add_subplot(gs[3, 0])
    if has_tomato:
        bar_width = 0.18
        for i, exp in enumerate(EXPS):
            r = tomato_reports[exp]
            if r:
                p = np.array(r["persistences"])[:15]
                x = np.arange(len(p)) + i * bar_width
                ax_bar.bar(x, p, bar_width, color=C_EXP[exp], alpha=0.8,
                          label=f"Exp {exp} (K={r['n_clusters']})", edgecolor="k", lw=0.3)
        ax_bar.set_xlabel("Componente (ordinata per persistenza)")
        ax_bar.set_ylabel("Persistenza")
        ax_bar.set_title("ToMATo — Persistence Barcode (top 15)", fontsize=12,
                         fontweight="bold", loc="left")
        ax_bar.legend(fontsize=8)
        ax_bar.tick_params(labelsize=8)
    else:
        ax_bar.text(0.5, 0.5, "ToMATo reports not found", ha="center", va="center")
        ax_bar.set_title("ToMATo Persistence", fontsize=12, fontweight="bold", loc="left")

    # Cluster sizes
    ax_sizes = fig.add_subplot(gs[3, 1])
    if has_tomato:
        for i, exp in enumerate(EXPS):
            r = tomato_reports[exp]
            if r:
                sizes = r["cluster_sizes"]
                ks = sorted(sizes.keys(), key=int)
                total = sum(sizes.values())
                pcts = [sizes[k] / total * 100 for k in ks]
                x = np.arange(len(ks)) + i * bar_width
                ax_sizes.bar(x, pcts, bar_width, color=C_EXP[exp], alpha=0.8,
                            label=f"Exp {exp}", edgecolor="k", lw=0.3)
        ax_sizes.set_xlabel("Cluster")
        ax_sizes.set_ylabel("% finestre")
        ax_sizes.set_title("Cluster Size Distribution (%)", fontsize=12,
                           fontweight="bold", loc="left")
        ax_sizes.legend(fontsize=8)
        ax_sizes.tick_params(labelsize=8)
    else:
        ax_sizes.text(0.5, 0.5, "ToMATo reports not found", ha="center", va="center")

    # ── ROW 4: Timelines ──────────────────────────────────────────────────
    if has_labels:
        for idx, exp in enumerate(EXPS):
            ax_tl = fig.add_subplot(gs[4, 0] if idx < 2 else gs[4, 1])
            if idx == 1 or idx == 3:
                continue  # Use subgrid
        # Redo with proper subgrid
        gs_tl = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[4, :],
                                                  hspace=0.4)
        for idx, exp in enumerate(EXPS):
            ax_tl = fig.add_subplot(gs_tl[idx])
            if exp in labels_data:
                df_l = labels_data[exp]
                n_cl = df_l["cluster"].nunique()
                colors = [C_REGIME(k % 20) for k in range(n_cl)]
                cmap_disc = matplotlib.colors.ListedColormap(colors)
                ax_tl.scatter(df_l["datetime"].values, np.zeros(len(df_l)),
                             c=df_l["cluster"].values, cmap=cmap_disc, s=0.3,
                             alpha=0.8, vmin=0, vmax=n_cl - 1)
                ax_tl.set_yticks([])
                ax_tl.set_title(f"Exp {exp} (K={n_cl})", fontsize=9, loc="left",
                               fontweight="bold", color=C_EXP[exp])
                ax_tl.tick_params(labelsize=7)
                if idx < 3:
                    ax_tl.set_xticklabels([])
            else:
                ax_tl.text(0.5, 0.5, f"Exp {exp}: no labels", ha="center", va="center")
                ax_tl.set_yticks([])
        fig.text(0.5, gs[4, :].get_position(fig).y1 + 0.01,
                "Regime Timelines", ha="center", fontsize=13, fontweight="bold")
    else:
        ax_notl = fig.add_subplot(gs[4, :])
        ax_notl.text(0.5, 0.5, "Labels not found — run step04", ha="center", va="center")
        ax_notl.set_title("Regime Timelines", fontsize=13, fontweight="bold")
        ax_notl.axis("off")

    # ── ROW 5: Quality metrics ────────────────────────────────────────────
    ax_geo = fig.add_subplot(gs[5, 0])
    ax_eco = fig.add_subplot(gs[5, 1])

    if has_quality:
        # Geometric
        geo_metrics = ["silhouette_avg", "davies_bouldin", "dunn_index"]
        geo_labels = ["Silhouette ↑", "Davies-Bouldin ↓", "Dunn ↑"]
        x = np.arange(len(geo_metrics))
        bar_w = 0.18
        for i, exp in enumerate(EXPS):
            q = quality_reports[exp]
            if q:
                g = q["geometric"]
                vals = [g.get(m, 0) for m in geo_metrics]
                ax_geo.bar(x + i * bar_w, vals, bar_w, color=C_EXP[exp],
                          label=f"Exp {exp}", edgecolor="k", lw=0.3)
        ax_geo.set_xticks(x + 1.5 * bar_w)
        ax_geo.set_xticklabels(geo_labels, fontsize=9)
        ax_geo.set_title("Geometric Quality", fontsize=12, fontweight="bold", loc="left")
        ax_geo.legend(fontsize=8)
        ax_geo.tick_params(labelsize=8)

        # Economic
        eco_metrics = ["eta_squared", "transition_rate", "cramer_v_season"]
        eco_labels = ["η² (LMP) ↑", "Transition rate ↓", "Cramér V (season)"]
        for i, exp in enumerate(EXPS):
            q = quality_reports[exp]
            if q:
                e = q["economic"]
                vals = [e.get(m, 0) for m in eco_metrics]
                ax_eco.bar(x + i * bar_w, vals, bar_w, color=C_EXP[exp],
                          label=f"Exp {exp}", edgecolor="k", lw=0.3)
        ax_eco.set_xticks(x + 1.5 * bar_w)
        ax_eco.set_xticklabels(eco_labels, fontsize=9)
        ax_eco.set_title("Economic Quality", fontsize=12, fontweight="bold", loc="left")
        ax_eco.legend(fontsize=8)
        ax_eco.tick_params(labelsize=8)
    else:
        ax_geo.text(0.5, 0.5, "Quality reports not found — run step05",
                   ha="center", va="center")
        ax_geo.set_title("Geometric Quality", fontsize=12, fontweight="bold", loc="left")
        ax_geo.axis("off")
        ax_eco.text(0.5, 0.5, "Quality reports not found — run step05",
                   ha="center", va="center")
        ax_eco.set_title("Economic Quality", fontsize=12, fontweight="bold", loc="left")
        ax_eco.axis("off")

    # ── ROW 6: Embedding diagnostics ──────────────────────────────────────
    ax_diag1 = fig.add_subplot(gs[6, 0])
    ax_diag2 = fig.add_subplot(gs[6, 1])

    if has_diag:
        # Mantel + NN coherence
        exps_with_diag = [e for e in EXPS if diag_reports[e] is not None]
        mantel_vals = [diag_reports[e]["transferability"]["mantel_r"]
                      for e in exps_with_diag]
        nn_vals = [diag_reports[e]["transferability"]["nn_ratio_mean"]
                  for e in exps_with_diag]
        kendall_vals = [diag_reports[e]["transferability"]["nn_kendall_tau"]
                       for e in exps_with_diag]

        x = np.arange(len(exps_with_diag))
        ax_diag1.bar(x - 0.15, mantel_vals, 0.3, color="#4a90d9",
                    label="Mantel r", edgecolor="k", lw=0.5)
        ax_diag1.bar(x + 0.15, kendall_vals, 0.3, color="#e07b39",
                    label="Kendall τ (NN)", edgecolor="k", lw=0.5)
        ax_diag1.set_xticks(x)
        ax_diag1.set_xticklabels([f"Exp {e}" for e in exps_with_diag])
        ax_diag1.set_title("Transferability: Mantel r & Kendall τ",
                          fontsize=12, fontweight="bold", loc="left")
        ax_diag1.legend(fontsize=8)
        ax_diag1.tick_params(labelsize=8)

        # Reconstruction R² (best per experiment)
        for e in exps_with_diag:
            r2 = diag_reports[e]["opacity"]["reconstruction"]
            sorted_r2 = sorted(r2.items(), key=lambda x: x[1], reverse=True)[:8]
        # Show last experiment's reconstruction as example
        if exps_with_diag:
            e_show = exps_with_diag[0]
            r2 = diag_reports[e_show]["opacity"]["reconstruction"]
            sorted_r2 = sorted(r2.items(), key=lambda x: x[1], reverse=True)[:10]
            names = [x[0] for x in sorted_r2]
            vals = [x[1] for x in sorted_r2]
            colors = ["#4a90d9" if v > 0.5 else "#e07b39" if v > 0.1 else "#d9534f"
                     for v in vals]
            ax_diag2.barh(range(len(names)), vals, color=colors, edgecolor="k", lw=0.3)
            ax_diag2.set_yticks(range(len(names)))
            ax_diag2.set_yticklabels(names, fontsize=8)
            ax_diag2.invert_yaxis()
            ax_diag2.set_xlabel("R² (Ridge CV)")
            ax_diag2.set_title(f"Opacity: Reconstruction R² (Exp {e_show})",
                              fontsize=12, fontweight="bold", loc="left")
            ax_diag2.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
            ax_diag2.tick_params(labelsize=8)
    else:
        ax_diag1.text(0.5, 0.5, "Diagnostics not found — run step06",
                     ha="center", va="center")
        ax_diag1.set_title("Transferability", fontsize=12, fontweight="bold", loc="left")
        ax_diag1.axis("off")
        ax_diag2.text(0.5, 0.5, "Diagnostics not found — run step06",
                     ha="center", va="center")
        ax_diag2.set_title("Opacity", fontsize=12, fontweight="bold", loc="left")
        ax_diag2.axis("off")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = RESULTS / "overview.png"
    fig.savefig(out_path, dpi=DPI, facecolor=C_BG)
    plt.close(fig)
    print(f"  Salvato: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
