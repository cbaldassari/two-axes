"""
utils/pca_dim_selector.py
=========================
Selezione data-driven del numero di componenti PCA via distribuzione
di Marchenko-Pastur (Random Matrix Theory).

Metodo
------
Per una matrice n×p con entries di varianza σ², gli autovalori della
matrice di covarianza campionaria seguono asintoticamente la legge di
Marchenko-Pastur (1967) con bordo superiore:

    λ_max = σ² · (1 + √(p/n))²

Tutti gli autovalori λᵢ > λ_max contengono segnale reale (non sono
compatibili con il rumore puro). Tutti quelli ≤ λ_max sono compatibili
con una matrice puramente casuale e vengono scartati.

Per dati standardizzati (StandardScaler: σ² = 1) la formula diventa:

    λ_max = (1 + √(p/n))²

Riferimento: V.A. Marchenko & L.A. Pastur, "Distribution of eigenvalues
for some sets of random matrices", Mathematics of the USSR-Sbornik, 1967.
"""

from __future__ import annotations
import numpy as np


def marchenko_pastur_n_components(
    eigenvalues,
    n_samples: int,
    n_features: int,
    min_comp: int = 2,
) -> tuple[int, float]:
    """Numero di componenti PCA via soglia di Marchenko-Pastur.

    Parameters
    ----------
    eigenvalues : array-like
        Autovalori della matrice di covarianza campionaria
        (``PCA().explained_variance_``), in ordine decrescente.
    n_samples : int
        Numero di osservazioni (righe della matrice di input).
    n_features : int
        Numero di feature (colonne della matrice di input,
        dimensione originale prima di PCA).
    min_comp : int
        Numero minimo di componenti restituito (default 2).

    Returns
    -------
    n_comp : int
        Numero di componenti con autovalore > λ_max.
    lambda_max : float
        Soglia di Marchenko-Pastur usata.
    """
    gamma = n_features / n_samples
    lambda_max = (1.0 + np.sqrt(gamma)) ** 2
    eigs = np.asarray(eigenvalues)
    n_comp = max(min_comp, int(np.sum(eigs > lambda_max)))
    return n_comp, lambda_max


def spectral_gap_n_components(
    eigenvalues,
    min_comp: int = 3,
    max_comp: int = 60,
) -> int:
    """Select number of components via spectral gap.

    Finds the largest ratio drop eigenvalue[i] / eigenvalue[i+1]
    for i >= min_comp-1.  The cut is placed at the index with the
    maximum ratio (biggest relative drop).

    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues in descending order (e.g., from diffusion maps).
    min_comp : int
        Minimum number of components returned (default 3).
    max_comp : int
        Maximum number of components considered (default 60).

    Returns
    -------
    n_comp : int
    """
    eigs = np.abs(np.asarray(eigenvalues, dtype=np.float64))[:max_comp]
    if len(eigs) < min_comp + 1:
        return min_comp

    ratios = eigs[:-1] / np.clip(eigs[1:], 1e-12, None)

    best_idx = min_comp - 1
    best_ratio = 0.0
    for i in range(min_comp - 1, len(ratios)):
        if ratios[i] > best_ratio:
            best_ratio = ratios[i]
            best_idx = i

    n_comp = max(best_idx + 1, min_comp)
    return n_comp
