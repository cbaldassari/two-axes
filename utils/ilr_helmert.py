"""
ilr_helmert.py
==============
ILR (Isometric Log-Ratio) transform with Helmert basis.

Helmert contrast matrix is a standard orthonormal basis that does not
require an arbitrary partition of the composition (unlike SBP).
Row k contrasts part k+1 against the geometric mean of parts 1..k.

References:
  - Egozcue et al. (2003), "Isometric Logratio Transformations for Compositional Data Analysis"
  - Pawlowsky-Glahn, Egozcue & Tolosana-Delgado (2015), "Modeling and Analysis of Compositional Data"
"""

import numpy as np


def helmert_basis(D: int) -> np.ndarray:
    """
    Helmert contrast matrix for D-part composition.

    Returns (D-1, D) orthonormal matrix V such that ILR = log(X) @ V.T

    Row k (0-indexed) contrasts part k+1 against geometric mean of parts 0..k:
      V[k, j] = 1/sqrt(k*(k+1))  for j < k
      V[k, k] = -k/sqrt(k*(k+1))
      V[k, j] = 0                for j > k
    """
    V = np.zeros((D - 1, D))
    for k in range(D - 1):
        # k is 0-indexed row; contrasts part (k+1) against parts 0..k
        scale = np.sqrt((k + 1) * (k + 2))
        V[k, :k + 1] = 1.0 / scale
        V[k, k + 1] = -(k + 1) / scale
    return V


def ilr_transform(compositions: np.ndarray, delta: float = 0.0001) -> np.ndarray:
    """
    Forward ILR transform: (N, D) compositions -> (N, D-1) ILR coordinates.

    Steps:
      1. Multiplicative zero replacement (Martin-Fernandez et al., 2003)
      2. Closure (re-normalize to sum=1)
      3. Log transform
      4. Project onto Helmert basis

    Parameters
    ----------
    compositions : (N, D) array, rows sum to ~1 (or ~100)
    delta : zero replacement value

    Returns
    -------
    ilr_coords : (N, D-1) array
    """
    X = np.array(compositions, dtype=np.float64)
    N, D = X.shape

    # Zero replacement
    X[X <= 0] = delta

    # Closure
    X = X / X.sum(axis=1, keepdims=True)

    # Log
    logX = np.log(X)

    # Helmert basis
    V = helmert_basis(D)

    # ILR = logX @ V.T
    return logX @ V.T


def inverse_ilr_transform(ilr_coords: np.ndarray, D: int = 8) -> np.ndarray:
    """
    Inverse ILR transform: (N, D-1) ILR coordinates -> (N, D) compositions.

    Parameters
    ----------
    ilr_coords : (N, D-1) array
    D : number of parts in the composition (default 8 for fuel types)

    Returns
    -------
    compositions : (N, D) array, rows sum to 1
    """
    V = helmert_basis(D)

    # Inverse: logX = ilr @ V (since V is orthonormal, V.T @ V = I)
    logX = ilr_coords @ V

    # Exponentiate and close
    X = np.exp(logX)
    X = X / X.sum(axis=1, keepdims=True)

    return X
