# Two-Axis Regime Detection in Electricity Markets

**Quasi-orthogonal regimes in ISO New England: price level and temporal persistence as independent dimensions**

This repository contains the code and data pipeline for the paper by C. Mari and C. Baldassari (Università della Tuscia, 2026).

## Key Finding

Electricity market regimes are conventionally described along a single dimension — price level. We show that this view misses an entire axis of market structure. Using a fully unsupervised, data-driven pipeline, we discover **two statistically independent axes** of regime organization:

- **Economic axis** (Feature Engineering on Δr_t): 9 regimes separated by price level ($30–$151/MWh)
- **Dynamic axis** (MOMENT embeddings on r_t): 9 regimes separated by temporal persistence (ACF 0.38–0.94)

The two axes are nearly orthogonal (ARI = 0.012). Within the same price regime, the speed of mean-reversion varies by a factor of 5×.

## Pipeline

```
LMP hourly prices
    │
    ▼
Preprocessing: arcsinh → MSTL(24h, 168h, 8760h) → residual r_t
    │
    ├── Δr_t (stationary) ──► Feature Engineering (15 features)
    │                              │
    │                              ▼
    │                         Diffusion Maps (d=2)
    │                              │
    │                              ▼
    │                         ToMATo clustering (48 modes)
    │                              │
    │                              ▼
    │                         Tukey HSD merge → 9 Economic regimes (E)
    │
    └── r_t (persistent) ──► MOMENT-1-large embedding (1024D, zero-shot)
                                   │
                                   ▼
                              Diffusion Maps (d=5)
                                   │
                                   ▼
                              ToMATo clustering (47 modes)
                                   │
                                   ▼
                              Tukey HSD merge → 9 Dynamic regimes (D)
```

## Repository Structure

```
├── config.py                    # Central configuration
├── isone_dataset.parquet        # ISONE LMP + fuel mix (43,814 hourly obs, 2021-2025)
├── pipeline_darcsinh/
│   └── run_pipeline.py          # Full pipeline script (preprocessing → regimes → validation)
├── notebooks/
│   └── pipeline.ipynb           # Step-by-step notebook version of the pipeline
├── results_darcsinh/
│   └── split_W512_S6/           # Output: labels, features, embeddings, diagnostics
├── paper/
│   ├── paper_v10_it.tex         # Paper (Italian)
│   └── references.bib           # Bibliography
└── scripts/                     # Analysis and visualization scripts
```

## Quick Start

### Requirements

```bash
pip install numpy pandas torch scikit-learn statsmodels momentfm gudhi scipy
```

### Run the pipeline

```bash
python pipeline_darcsinh/run_pipeline.py
```

This runs the full pipeline in `split` mode (default): Feature Engineering on Δr_t, MOMENT on r_t. Results are saved to `results_darcsinh/split_W512_S6/`.

### Notebook

For a step-by-step walkthrough, see `notebooks/pipeline.ipynb`.

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `momentfm` | MOMENT-1-large foundation model for time series embedding |
| `gudhi` | ToMATo clustering via persistent homology |
| `scikit-learn` | StandardScaler, PCA, KMeans, silhouette, ARI |
| `statsmodels` | MSTL seasonal-trend decomposition |
| `torch` | GPU-accelerated diffusion maps and MOMENT inference |

## Data

ISONE day-ahead LMP for Massachusetts Hub, downloaded from the [EIA Open Data API](https://api.eia.gov/v2/electricity/rto/wholesale-data) (`respondent=ISNE`, `type=D`). The dataset includes hourly LMP and EIA fuel generation mix (8 fuel types) from January 2021 to December 2025.

## Citation

```bibtex
@article{mari2026twoaxes,
  title={Quasi-orthogonal regimes in {ISO New England}: price level and
         temporal persistence as independent dimensions},
  author={Mari, Carlo and Baldassari, Cristiano},
  year={2026}
}
```

## License

This project is for academic research purposes.
