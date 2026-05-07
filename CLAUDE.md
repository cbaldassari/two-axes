# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**This is a new project. No pipeline scripts exist yet — they need to be written from scratch.**

The `pipeline/` and `utils/` directories are empty placeholders. `RESEARCH_LOG.md` describes the target architecture. The dataset (`isone_dataset.parquet`) lives in the sibling repo `nepool-regime-detection-main/`.

## What This Project Is

Research pipeline for **electricity market regime detection** on ISO New England (ISONE), using:
- **MOMENT** (time-series foundation model, AutonLab/CMU, ICML 2024) for embedding
- **ToMATo** (Topological Mode Analysis Tool, Chazal et al. 2013) for clustering — K emerges from persistent homology, no parametric assumptions
- **Ward dendrogram** as a traditional baseline for comparison

Dataset: `isone_dataset.parquet` — 43,794 hourly rows, 2021–2025, Massachusetts Hub LMP + EIA fuel generation mix (8 types).

## What This Project Is NOT

This is NOT the old pipeline (`nepool-regime-detection-main/`) which used GMM, UMAP, TOPSIS, and 15 experiments. That approach was abandoned because:
- GMM hit K=20 (ceiling) suggesting no natural discrete structure for Gaussian clusters
- 15 experiments produced divergent rankings with no convergence
- UMAP introduced circularity in evaluation

## Target Pipeline (to be implemented)

```
step01  Preprocessing
        LMP → log_return → MSTL(24h+168h) → mstl_resid_lr
        Fuel mix → ILR(8→7) → MSTL(24h+168h) → mstl_resid_ilr_*

step02  MOMENT embedding (AutonLab/MOMENT-1-large, ICML 2024)
        Single input: mstl_resid_lr (univariate, 512h window, 6h stride)
        Concat: mstl_resid_ilr_* (7D) at window timestamp
        Output: 1031D per window (~7200 windows)

step03  PCA (Marchenko-Pastur) → ToMATo clustering
        PCA: keep eigenvalues > λ_max = (1+√(p/n))²
        ToMATo: k-NN density, persistence diagram → K automatic
        Baseline: Ward dendrogram on same PCA space
        Compare: ARI, NMI, economic metrics

step04  Validation (TBD)

step05  Markov chain analysis on identified regimes
```

## Key Design Decisions

- **No UMAP**: ToMATo operates directly on PCA space. UMAP only for 2D visualization if needed.
- **K from data**: persistence diagram determines number of regimes — no BIC sweep.
- **Marchenko-Pastur for PCA**: data-driven threshold, not arbitrary 95% variance.
- **Ward baseline**: if ToMATo and Ward agree → robust. If they diverge → informative.
- **MOMENT over Chronos-2**: Chronos-2 embeddings were opaque (negative R² on all market features, Mantel r ~0.15). MOMENT is designed for representation learning (not just forecasting) and trained on the Time-Series Pile.

## Key Dependencies (to install)

- `momentfm` — MOMENT embedding
- `gudhi` or `giotto-tda` — ToMATo + persistence diagrams
- `scikit-learn` — PCA, metrics
- `scipy` — Ward dendrogram
