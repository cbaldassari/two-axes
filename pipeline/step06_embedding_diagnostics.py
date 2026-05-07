"""
step06_embedding_diagnostics.py
================================
NEPOOL TDA Pipeline — Step 06: Embedding Opacity & Transferability Tests

Verifica se gli embedding Chronos-2 sono interpretabili e catturano
struttura reale del mercato elettrico.

A) OPACITY — quanto sono interpretabili gli embedding?
   1. Correlazione PC <-> variabili fisiche (LMP, volatilita, quote fuel)
   2. Feature importance: Random Forest predice LMP medio dalla PCA embedding
      -> importanza per componente = "cosa codifica ogni PC"
   3. R² per variabile: quanto ciascuna variabile fisica e predicibile
      dall'embedding (Ridge regression)

B) TRANSFERABILITY — Chronos-2 cattura struttura del mercato?
   1. Mantel test: correlazione tra distanze nello spazio embedding e
      distanze nello spazio feature manuali (Pearson su matrici distanza)
   2. Reconstruction test: decoder lineare (Ridge) predice feature manuali
      dall'embedding -> R² per feature
   3. Nearest-neighbor coherence: i k-NN nello spazio embedding hanno
      condizioni di mercato simili? (Kendall ? su distanze)

Input  : results/exp_{X}/step03/pca_embeddings.parquet
         results/preprocessed.parquet
Output : results/exp_{X}/step06/
           diagnostics_report.json
           01_pc_correlation.png        heatmap PC <-> variabili fisiche
           02_feature_importance.png    importanza PC per predire LMP
           03_reconstruction_r2.png     R² ricostruzione feature manuali
           04_mantel_scatter.png        distanze embedding vs feature
           05_nn_coherence.png          coherenza nearest-neighbor

Uso
---
  python pipeline/step06_embedding_diagnostics.py --exp A
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
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

_parser = argparse.ArgumentParser(description="Step 06 — Embedding diagnostics")
_parser.add_argument("--exp", default="A", choices=["A", "B", "C", "D"])
_args = _parser.parse_args()
EXPERIMENT = _args.exp

RESULTS_DIR  = Path("results")
PCA_PATH     = RESULTS_DIR / f"exp_{EXPERIMENT}" / "step03" / "pca_embeddings.parquet"
PREPROC_PATH = RESULTS_DIR / "preprocessed.parquet"
OUT_DIR      = RESULTS_DIR / f"exp_{EXPERIMENT}" / "step06"

SEED       = 42
DPI        = 150
N_SAMPLE   = 2000   # subsample per Mantel test (distanze N² troppo grandi)
K_NN       = 20     # vicini per coherence test
CV_FOLDS   = 5


# =============================================================================
# LOAD & BUILD FEATURES
# =============================================================================

def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    """Carica PCA embeddings e costruisce feature manuali per ogni finestra."""
    # PCA embeddings
    df_pca = pd.read_parquet(PCA_PATH)
    timestamps = pd.to_datetime(df_pca["datetime"])
    E = df_pca.drop(columns=["datetime"]).values.astype(np.float32)

    # Preprocessed
    pre = pd.read_parquet(PREPROC_PATH)
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    pre = pre.set_index("datetime")

    # Calcola feature rolling sulla serie oraria completa (vettorizzato),
    # poi campiona ai timestamp delle finestre.
    WINDOW = 512
    lmp_s = pre["lmp"]

    # Rolling stats su finestra 720h
    r = lmp_s.rolling(WINDOW, min_periods=int(WINDOW * 0.9))
    roll_feat = pd.DataFrame({
        "lmp_mean":   r.mean(),
        "lmp_std":    r.std(),
        "lmp_skew":   r.skew(),
        "lmp_kurt":   r.kurt(),
        "lmp_min":    r.min(),
        "lmp_max":    r.max(),
        "lmp_median": r.median(),
    }, index=pre.index)
    roll_feat["lmp_range"] = roll_feat["lmp_max"] - roll_feat["lmp_min"]
    # Percentili — no rolling nativo, calcolo con quantile
    roll_feat["lmp_p5"]  = r.quantile(0.05)
    roll_feat["lmp_p95"] = r.quantile(0.95)

    # ACF a vari lag: calcolato come prodotto shiftato / varianza rolling
    for lag in [1, 6, 24, 168]:
        cov_lag = (lmp_s * lmp_s.shift(lag)).rolling(WINDOW, min_periods=int(WINDOW * 0.9)).mean() \
                  - roll_feat["lmp_mean"] * lmp_s.shift(lag).rolling(WINDOW, min_periods=int(WINDOW * 0.9)).mean()
        var_roll = roll_feat["lmp_std"] ** 2
        roll_feat[f"lmp_acf_{lag}h"] = (cov_lag / var_roll.clip(lower=1e-10))

    # Volatilita' 24h media nella finestra
    lr = np.diff(np.arcsinh(lmp_s.values), prepend=np.nan)
    vol_24h = pd.Series(lr, index=pre.index).rolling(24, min_periods=2).std()
    roll_feat["vol_24h_mean"] = vol_24h.rolling(WINDOW, min_periods=int(WINDOW * 0.9)).mean()

    # ILR (ultima osservazione della finestra = valore al timestamp)
    ilr_cols = [c for c in pre.columns if c.startswith("ilr_") and "resid" not in c]
    for col in ilr_cols:
        roll_feat[col] = pre[col].fillna(0.0).values

    # Temporali
    roll_feat["month"] = pre.index.month

    # Campiona ai timestamp delle finestre
    ts_set = set(timestamps)
    pre_ts_idx = {t: i for i, t in enumerate(pre.index)}
    valid_idx = []
    feat_rows = []
    for emb_i, ts in enumerate(timestamps):
        if ts in pre_ts_idx:
            row_i = pre_ts_idx[ts]
            row = roll_feat.iloc[row_i]
            if row.isna().sum() < 5:  # non troppi NaN
                valid_idx.append(emb_i)
                feat_rows.append(row)

    df_feat = pd.DataFrame(feat_rows).reset_index(drop=True)
    # Aggiungi temporali dal timestamp
    df_feat["month"]     = [timestamps[i].month for i in valid_idx]
    df_feat["hour"]      = [timestamps[i].hour for i in valid_idx]
    df_feat["dayofweek"] = [timestamps[i].dayofweek for i in valid_idx]
    # Riempi NaN residui con 0
    df_feat = df_feat.fillna(0.0)

    E_valid = E[valid_idx]

    print(f"  Embedding: {E_valid.shape}  |  Feature manuali: {df_feat.shape}", flush=True)
    return E_valid, df_feat


# =============================================================================
# A) OPACITY TESTS
# =============================================================================

def test_pc_correlation(E: np.ndarray, df_feat: pd.DataFrame) -> dict:
    """Correlazione Pearson tra ogni PC e ogni variabile fisica."""
    print("\n  [A1] Correlazione PC <-> variabili fisiche...", flush=True)

    n_pc = min(E.shape[1], 20)  # prime 20 PC
    feat_cols = [c for c in df_feat.columns if c not in ["month", "hour", "dayofweek"]]
    n_feat = len(feat_cols)

    corr_matrix = np.zeros((n_pc, n_feat))
    pval_matrix = np.zeros((n_pc, n_feat))

    for i in range(n_pc):
        for j, col in enumerate(feat_cols):
            r, p = pearsonr(E[:, i], df_feat[col].values)
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

    # Max correlazione per PC
    max_corr_per_pc = {}
    for i in range(n_pc):
        best_j = int(np.argmax(np.abs(corr_matrix[i])))
        max_corr_per_pc[f"PC{i}"] = {
            "best_feature": feat_cols[best_j],
            "r": round(float(corr_matrix[i, best_j]), 4),
        }
    top5 = sorted(max_corr_per_pc.items(), key=lambda x: abs(x[1]["r"]), reverse=True)[:5]
    for pc, info in top5:
        print(f"    {pc}: r={info['r']:+.3f} con {info['best_feature']}", flush=True)

    return {
        "corr_matrix": corr_matrix,
        "feat_cols": feat_cols,
        "n_pc": n_pc,
        "max_corr_per_pc": max_corr_per_pc,
    }


def test_feature_importance(E: np.ndarray, df_feat: pd.DataFrame) -> dict:
    """Random Forest: quali PC sono importanti per predire LMP medio?

    Nota: R² negativo in CV significa che il modello e' peggio della media
    costante — l'embedding PCA (che codifica residui MSTL destagionalizzati)
    non predice direttamente il livello LMP raw. Questo e' atteso.
    """
    print("\n  [A2] Feature importance (RF -> LMP medio)...", flush=True)

    y = df_feat["lmp_mean"].values
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(E)

    rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                                min_samples_leaf=20,
                                random_state=SEED, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=CV_FOLDS, scoring="r2")
    r2_cv = float(np.mean(scores))
    r2_cv_clamped = max(0.0, r2_cv)

    rf.fit(X, y)
    importances = rf.feature_importances_

    print(f"    R² CV (LMP mean): {r2_cv:.4f}"
          f"{'  [clamped to 0]' if r2_cv < 0 else ''}", flush=True)
    if r2_cv < 0:
        print(f"    (R²<0: embedding destagionalizzato non predice LMP raw — atteso)",
              flush=True)
    top_idx = np.argsort(importances)[::-1][:5]
    for idx in top_idx:
        print(f"    PC{idx}: importance={importances[idx]:.4f}", flush=True)

    return {
        "r2_cv_lmp_mean": round(r2_cv, 4),
        "r2_cv_lmp_mean_clamped": round(r2_cv_clamped, 4),
        "importances": importances,
    }


def test_reconstruction(E: np.ndarray, df_feat: pd.DataFrame) -> dict:
    """Ridge regression: quanto ciascuna feature e' predicibile dall'embedding?

    R² negativo = il modello lineare e' peggio della media costante.
    Viene riportato tal quale ma clampato a 0 nei plot e nel summary.
    Cause tipiche: embedding codifica residui destagionalizzati, feature
    manuali sono su LMP raw -> disallineamento di scala/contenuto.
    """
    print("\n  [A3] Reconstruction R² (embedding -> feature manuali)...", flush=True)

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(E)

    feat_cols = [c for c in df_feat.columns if c not in ["month", "hour", "dayofweek"]]
    r2_per_feature = {}
    r2_clamped = {}

    for col in feat_cols:
        y = df_feat[col].values
        if np.std(y) < 1e-10:
            continue
        # Standardizza anche y per migliorare la stabilita' di Ridge
        y_mean, y_std = y.mean(), y.std()
        y_s = (y - y_mean) / y_std

        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=CV_FOLDS)
        ridge.fit(X, y_s)
        scores = cross_val_score(ridge, X, y_s, cv=CV_FOLDS, scoring="r2")
        r2 = float(np.mean(scores))
        r2_per_feature[col] = round(r2, 4)
        r2_clamped[col] = round(max(0.0, r2), 4)

    # Ordina per R²
    sorted_feats = sorted(r2_per_feature.items(), key=lambda x: x[1], reverse=True)
    for col, r2 in sorted_feats[:8]:
        flag = "" if r2 > 0 else "  [<0: non predicibile]"
        print(f"    {col:.<25s} R²={r2:.4f}{flag}", flush=True)
    n_negative = sum(1 for _, r in sorted_feats if r < 0)
    n_good = sum(1 for _, r in sorted_feats if r > 0.3)
    print(f"    Riepilogo: {n_good} feature con R²>0.3, "
          f"{n_negative} con R²<0", flush=True)

    return {
        "r2_per_feature": r2_per_feature,
        "r2_per_feature_clamped": r2_clamped,
    }


# =============================================================================
# B) TRANSFERABILITY TESTS
# =============================================================================

def test_mantel(E: np.ndarray, df_feat: pd.DataFrame) -> dict:
    """Mantel test: correlazione tra matrici di distanza embedding vs feature."""
    print("\n  [B1] Mantel test (distanze embedding vs feature)...", flush=True)

    n = len(E)
    rng = np.random.default_rng(SEED)

    # Subsample per contenere la memoria
    if n > N_SAMPLE:
        idx = rng.choice(n, N_SAMPLE, replace=False)
        E_sub = E[idx]
        feat_cols = [c for c in df_feat.columns if c not in ["month", "hour", "dayofweek"]]
        F_sub = df_feat[feat_cols].values[idx]
    else:
        E_sub = E
        feat_cols = [c for c in df_feat.columns if c not in ["month", "hour", "dayofweek"]]
        F_sub = df_feat[feat_cols].values

    # Standardizza entrambi
    E_sub = StandardScaler().fit_transform(E_sub)
    F_sub = StandardScaler().fit_transform(F_sub)

    # Distanze condensate
    d_emb = pdist(E_sub, metric="euclidean")
    d_feat = pdist(F_sub, metric="euclidean")

    # Pearson sulle distanze
    r_mantel, p_mantel = pearsonr(d_emb, d_feat)
    print(f"    Mantel r = {r_mantel:.4f}  (p = {p_mantel:.2e})", flush=True)
    print(f"    Interpretazione: ", end="", flush=True)
    if r_mantel > 0.7:
        print("forte — embedding e feature manuali concordano molto", flush=True)
    elif r_mantel > 0.4:
        print("moderata — embedding cattura buona parte della struttura", flush=True)
    elif r_mantel > 0.2:
        print("debole — embedding cattura informazione parziale", flush=True)
    else:
        print("molto debole — embedding diverge dalle feature manuali", flush=True)

    return {
        "mantel_r": round(float(r_mantel), 4),
        "mantel_p": float(p_mantel),
        "n_sample": len(E_sub),
        "d_emb_sample": d_emb,
        "d_feat_sample": d_feat,
    }


def test_nn_coherence(E: np.ndarray, df_feat: np.ndarray) -> dict:
    """Per ogni punto, i k-NN nell'embedding hanno feature simili?"""
    print(f"\n  [B2] Nearest-neighbor coherence (k={K_NN})...", flush=True)

    n = len(E)
    rng = np.random.default_rng(SEED)

    # Subsample
    if n > N_SAMPLE:
        idx = rng.choice(n, N_SAMPLE, replace=False)
        E_sub = E[idx]
        feat_cols = [c for c in df_feat.columns if c not in ["month", "hour", "dayofweek"]]
        F_sub = df_feat[feat_cols].values[idx]
    else:
        E_sub = E
        feat_cols = [c for c in df_feat.columns if c not in ["month", "hour", "dayofweek"]]
        F_sub = df_feat[feat_cols].values

    E_sub = StandardScaler().fit_transform(E_sub)
    F_sub = StandardScaler().fit_transform(F_sub)

    # Distanze complete
    D_emb = squareform(pdist(E_sub, "euclidean"))
    D_feat = squareform(pdist(F_sub, "euclidean"))

    # Per ogni punto: ranking dei vicini in embedding vs feature
    taus = []
    for i in range(len(E_sub)):
        # Rank dei K_NN vicini nell'embedding
        nn_emb = np.argsort(D_emb[i])[1:K_NN + 1]  # escludi se stesso
        # Distanza feature media dei k-NN vs random
        d_feat_nn = np.mean(D_feat[i, nn_emb])
        # Distanza feature media globale
        d_feat_all = np.mean(D_feat[i])
        # Ratio: < 1 = i vicini nell'embedding sono anche vicini nelle feature
        taus.append(d_feat_nn / d_feat_all if d_feat_all > 0 else 1.0)

    taus = np.array(taus)
    mean_ratio = float(np.mean(taus))
    median_ratio = float(np.median(taus))

    print(f"    Ratio medio  d_feat(k-NN) / d_feat(all) = {mean_ratio:.4f}", flush=True)
    print(f"    Ratio mediano = {median_ratio:.4f}", flush=True)
    print(f"    (< 1.0 = vicini nell'embedding sono anche vicini nelle feature)", flush=True)
    if mean_ratio < 0.6:
        print(f"    -> Forte coerenza", flush=True)
    elif mean_ratio < 0.8:
        print(f"    -> Buona coerenza", flush=True)
    elif mean_ratio < 0.95:
        print(f"    -> Coerenza moderata", flush=True)
    else:
        print(f"    -> Coerenza debole", flush=True)

    # Kendall tau sui ranking completi (subsample più piccolo)
    n_kendall = min(500, len(E_sub))
    tau_vals = []
    for i in range(n_kendall):
        rank_emb = np.argsort(np.argsort(D_emb[i]))
        rank_feat = np.argsort(np.argsort(D_feat[i]))
        tau, _ = kendalltau(rank_emb, rank_feat)
        tau_vals.append(tau)
    kendall_mean = float(np.mean(tau_vals))
    print(f"    Kendall tau medio (ranking): {kendall_mean:.4f}", flush=True)

    return {
        "ratio_mean": round(mean_ratio, 4),
        "ratio_median": round(median_ratio, 4),
        "kendall_tau_mean": round(kendall_mean, 4),
        "n_sample": len(E_sub),
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_pc_correlation(corr_info: dict):
    """Heatmap correlazione PC <-> variabili fisiche."""
    corr = corr_info["corr_matrix"]
    feat_cols = corr_info["feat_cols"]
    n_pc = corr_info["n_pc"]
    n_show = min(n_pc, 15)
    n_feat_show = min(len(feat_cols), 20)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(corr[:n_show, :n_feat_show], aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"PC{i}" for i in range(n_show)])
    ax.set_xticks(range(n_feat_show))
    ax.set_xticklabels(feat_cols[:n_feat_show], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Correlazione PC <-> variabili fisiche — Exp {EXPERIMENT}")
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

    # Annota celle con |r| > 0.3
    for i in range(n_show):
        for j in range(n_feat_show):
            if abs(corr[i, j]) > 0.3:
                color = "white" if abs(corr[i, j]) > 0.6 else "black"
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_pc_correlation.png", dpi=DPI)
    plt.close(fig)
    print(f"  01_pc_correlation.png", flush=True)


def plot_feature_importance(imp_info: dict):
    """Barplot importanza PC per predire LMP medio."""
    importances = imp_info["importances"]
    n_show = min(20, len(importances))
    top_idx = np.argsort(importances)[::-1][:n_show]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(n_show), importances[top_idx], color="#4a90d9", edgecolor="k", lw=0.5)
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"PC{i}" for i in top_idx], rotation=45)
    ax.set_ylabel("Importanza (RF)")
    ax.set_title(f"PC importance per LMP medio — Exp {EXPERIMENT}  "
                 f"(R² CV={imp_info['r2_cv_lmp_mean']:.3f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_feature_importance.png", dpi=DPI)
    plt.close(fig)
    print(f"  02_feature_importance.png", flush=True)


def plot_reconstruction_r2(recon_info: dict):
    """Barplot R² ricostruzione per feature."""
    r2 = recon_info["r2_per_feature"]
    sorted_items = sorted(r2.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    vals = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4a90d9" if v > 0.5 else "#e07b39" if v > 0.1 else "#d9534f" for v in vals]
    ax.barh(range(len(names)), vals, color=colors, edgecolor="k", lw=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("R² (Ridge CV)")
    ax.set_title(f"Ricostruzione feature manuali dall'embedding — Exp {EXPERIMENT}")
    ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="R²=0.5")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_reconstruction_r2.png", dpi=DPI)
    plt.close(fig)
    print(f"  03_reconstruction_r2.png", flush=True)


def plot_mantel(mantel_info: dict):
    """Scatter distanze embedding vs feature (subsample)."""
    d_emb = mantel_info["d_emb_sample"]
    d_feat = mantel_info["d_feat_sample"]

    # Subsample per il plot (troppi punti)
    rng = np.random.default_rng(SEED)
    n_plot = min(50000, len(d_emb))
    idx = rng.choice(len(d_emb), n_plot, replace=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(d_emb[idx], d_feat[idx], s=1, alpha=0.05, color="#4a90d9")
    ax.set_xlabel("Distanza embedding (PCA)")
    ax.set_ylabel("Distanza feature manuali")
    ax.set_title(f"Mantel test — Exp {EXPERIMENT}  "
                 f"(r={mantel_info['mantel_r']:.3f})")
    # Fit line
    z = np.polyfit(d_emb[idx], d_feat[idx], 1)
    x_line = np.linspace(d_emb[idx].min(), d_emb[idx].max(), 100)
    ax.plot(x_line, z[0] * x_line + z[1], color="red", lw=2, alpha=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_mantel_scatter.png", dpi=DPI)
    plt.close(fig)
    print(f"  04_mantel_scatter.png", flush=True)


def plot_nn_coherence(nn_info: dict):
    """Placeholder — info gia nel report JSON."""
    # Creiamo un summary plot
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = {
        f"d_feat ratio\n(< 1 = buono)": nn_info["ratio_mean"],
        f"Kendall tau\n(> 0 = buono)": nn_info["kendall_tau_mean"],
    }
    colors = []
    for k, v in metrics.items():
        if "ratio" in k:
            colors.append("#4a90d9" if v < 0.8 else "#e07b39" if v < 0.95 else "#d9534f")
        else:
            colors.append("#4a90d9" if v > 0.3 else "#e07b39" if v > 0.1 else "#d9534f")

    ax.bar(list(metrics.keys()), list(metrics.values()), color=colors,
           edgecolor="k", lw=0.5, width=0.4)
    for i, (k, v) in enumerate(metrics.items()):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=12)
    ax.set_title(f"NN coherence — Exp {EXPERIMENT} (k={K_NN})")
    ax.set_ylim(0, max(1.2, max(metrics.values()) + 0.1))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_nn_coherence.png", dpi=DPI)
    plt.close(fig)
    print(f"  05_nn_coherence.png", flush=True)


# =============================================================================
# SAVE
# =============================================================================

def save_report(corr_info, imp_info, recon_info, mantel_info, nn_info):
    report = {
        "experiment": EXPERIMENT,
        "opacity": {
            "pc_correlation": {
                "max_corr_per_pc": corr_info["max_corr_per_pc"],
            },
            "feature_importance": {
                "r2_cv_lmp_mean": imp_info["r2_cv_lmp_mean"],
            },
            "reconstruction": recon_info["r2_per_feature"],
        },
        "transferability": {
            "mantel_r": mantel_info["mantel_r"],
            "mantel_p": mantel_info["mantel_p"],
            "nn_ratio_mean": nn_info["ratio_mean"],
            "nn_ratio_median": nn_info["ratio_median"],
            "nn_kendall_tau": nn_info["kendall_tau_mean"],
        },
    }
    report_path = OUT_DIR / "diagnostics_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Salvato: {report_path}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 65)
    print(f"  NEPOOL TDA — Step 06: Embedding Opacity & Transferability")
    print(f"  Experiment : {EXPERIMENT}")
    print("=" * 65, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & build features
    print("\n  Loading data & building manual features...", flush=True)
    E, df_feat = load_data()

    # 2. Opacity tests
    print("\n" + "-" * 65)
    print("  A) OPACITY TESTS")
    print("-" * 65, flush=True)
    corr_info = test_pc_correlation(E, df_feat)
    imp_info = test_feature_importance(E, df_feat)
    recon_info = test_reconstruction(E, df_feat)

    # 3. Transferability tests
    print("\n" + "-" * 65)
    print("  B) TRANSFERABILITY TESTS")
    print("-" * 65, flush=True)
    mantel_info = test_mantel(E, df_feat)
    nn_info = test_nn_coherence(E, df_feat)

    # 4. Plots
    print(f"\n  Plots...", flush=True)
    plot_pc_correlation(corr_info)
    plot_feature_importance(imp_info)
    plot_reconstruction_r2(recon_info)
    plot_mantel(mantel_info)
    plot_nn_coherence(nn_info)

    # 5. Save
    print(f"\n  Salvataggio...", flush=True)
    save_report(corr_info, imp_info, recon_info, mantel_info, nn_info)

    elapsed = time.time() - t0
    print(f"\n  Completato in {elapsed:.1f}s")
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
