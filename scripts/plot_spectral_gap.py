"""
Generate spectral gap figure for Diffusion Maps (FE and MOMENT).
Produces paper/spectral_gap.png showing eigenvalue decay and gap selection.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

SEED = C.RANDOM_STATE
W = C.DARCSINH["W"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent / "results_darcsinh" / "split_W512_S6"
PAPER = Path(__file__).parent.parent / "paper"


def get_evals(X, label):
    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > 50:
        Xs = PCA(n_components=50, random_state=SEED).fit_transform(Xs)
    Xt = torch.tensor(Xs, dtype=torch.float64, device=DEVICE)
    dists = torch.cdist(Xt, Xt)
    eps = float(torch.median(dists[dists > 0]).item()) ** 2
    K = torch.exp(-dists ** 2 / eps)
    P = torch.diag(1.0 / K.sum(dim=1)) @ K
    evals = torch.linalg.eigvalsh(P).flip(0).cpu().numpy()
    print(f"{label}: top 12 eigenvalues = {evals[1:13].round(5)}")
    ratios = evals[1:12] / evals[2:13]
    print(f"{label}: ratios lam_i/lam_(i+1) = {ratios.round(3)}")
    return evals


# Load data
fe = pd.read_parquet(OUT / "fe_features.parquet").drop(columns=["datetime"]).values
mom = pd.read_parquet(OUT / "moment_embeddings.parquet").drop(columns=["datetime"]).values
print(f"FE: {fe.shape}, MOMENT: {mom.shape}")

# Compute eigenvalues
evals_fe = get_evals(StandardScaler().fit_transform(fe), "FE")
evals_mom = get_evals(mom, "MOMENT")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

n_show = 12
idx = np.arange(1, n_show + 1)

# FE panel
ax = axes[0]
ax.bar(idx, evals_fe[1:n_show+1], color="steelblue", alpha=0.8, edgecolor="navy", linewidth=0.5)
ax.axvline(x=2.5, color="red", linestyle="--", linewidth=1.5, label="gap: $d=2$")
ax.set_xlabel("Eigenvalue index $k$", fontsize=11)
ax.set_ylabel("$\\lambda_k$", fontsize=12)
ax.set_title("Feature Engineering", fontsize=12)
ax.set_xticks(idx)
ax.legend(fontsize=10)

# Annotate ratio
r_fe = evals_fe[2] / evals_fe[3]
ax.annotate(f"$\\lambda_2/\\lambda_3 = {r_fe:.2f}$",
            xy=(2.5, evals_fe[2]), xytext=(5, evals_fe[2]*0.95),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="red"),
            color="red")

# MOMENT panel
ax = axes[1]
ax.bar(idx, evals_mom[1:n_show+1], color="darkorange", alpha=0.8, edgecolor="saddlebrown", linewidth=0.5)
ax.axvline(x=5.5, color="red", linestyle="--", linewidth=1.5, label="$d=5$")
# Highlight plateau (lambda2-lambda4)
for i in [2, 3, 4]:
    ax.bar(i, evals_mom[i], color="gold", alpha=0.9, edgecolor="saddlebrown", linewidth=0.5)
ax.set_xlabel("Eigenvalue index $k$", fontsize=11)
ax.set_ylabel("$\\lambda_k$", fontsize=12)
ax.set_title("MOMENT", fontsize=12)
ax.set_xticks(idx)
ax.legend(fontsize=10)

# Annotate plateau and gap
ax.annotate("plateau\n$\\lambda_2$--$\\lambda_4$",
            xy=(3, evals_mom[3]), xytext=(7, evals_mom[2]*1.02),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="goldenrod"),
            color="goldenrod", ha="center")
r_mom_main = evals_mom[4] / evals_mom[5]
ax.annotate(f"$\\lambda_4/\\lambda_5 = {r_mom_main:.2f}$",
            xy=(4.5, evals_mom[4]), xytext=(7.5, evals_mom[4]*1.1),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="darkred"),
            color="darkred")
r_mom_5 = evals_mom[5] / evals_mom[6]
ax.annotate(f"$\\lambda_5/\\lambda_6 = {r_mom_5:.2f}$",
            xy=(5.5, evals_mom[5]), xytext=(8.5, evals_mom[5]*0.85),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="red"),
            color="red")

plt.tight_layout()
plt.savefig(PAPER / "spectral_gap.png", dpi=200, bbox_inches="tight")
print(f"\nSaved to {PAPER / 'spectral_gap.png'}")
plt.show()
