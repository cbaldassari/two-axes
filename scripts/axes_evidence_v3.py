"""
axes_evidence_v3.py — Clean version of the 2x2 orthogonality figure.
Row labels on the left: "Price level" and "Persistence".
Column labels on top: "FE regimes (E)" and "MOMENT regimes (D)".
Clean x-axis: E0-E8 and D0-D7.
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10

RESULTS = Path("results")

# Load
lab_fe = pd.read_parquet(RESULTS / "exp_FE/step04/labels.parquet")
lab_mom = pd.read_parquet(RESULTS / "exp_C/step04/labels_acf_merge.parquet")
lab_fe["datetime"] = pd.to_datetime(lab_fe["datetime"])
lab_mom["datetime"] = pd.to_datetime(lab_mom["datetime"])

fe_emb = pd.read_parquet(RESULTS / "exp_FE/embeddings.parquet")
fe_emb["datetime"] = pd.to_datetime(fe_emb["datetime"])

pre = pd.read_parquet(RESULTS / "preprocessed.parquet")
pre["datetime"] = pd.to_datetime(pre["datetime"])

df = lab_fe.merge(lab_mom, on="datetime", suffixes=("_fe", "_mom"))
df = df.merge(pre[["datetime", "lmp"]], on="datetime")
df = df.merge(fe_emb[["datetime", "acf_6h"]], on="datetime")

# Order: FE by LMP, MOMENT by ACF
fe_order = df.groupby("cluster_fe")["lmp"].mean().sort_values().index.tolist()
fe_rank = {r: i for i, r in enumerate(fe_order)}
df["E"] = df["cluster_fe"].map(fe_rank)

mom_order = df.groupby("cluster_mom")["acf_6h"].mean().sort_values().index.tolist()
mom_rank = {r: i for i, r in enumerate(mom_order)}
df["D"] = df["cluster_mom"].map(mom_rank)

K_E = len(fe_order)
K_D = len(mom_order)

colors_E = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, K_E))
colors_D = plt.cm.coolwarm(np.linspace(0.15, 0.85, K_D))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# --- (a) E regimes vs LMP — clean separation ---
ax = axes[0, 0]
data = [df.loc[df["E"] == i, "lmp"].values for i in range(K_E)]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(bp["boxes"], colors_E):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels([f"$E_{{{i}}}$" for i in range(K_E)], fontsize=9)
ax.set_ylabel("LMP ($/MWh)", fontsize=11)
ax.set_title("(a) Clear gradient", fontsize=12, fontweight="bold")

# --- (b) D regimes vs LMP — overlap ---
ax = axes[0, 1]
data = [df.loc[df["D"] == i, "lmp"].values for i in range(K_D)]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(bp["boxes"], colors_D):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels([f"$D_{{{i}}}$" for i in range(K_D)], fontsize=9)
ax.set_ylabel("LMP ($/MWh)", fontsize=11)
ax.set_title("(b) No separation", fontsize=12, fontweight="bold")

# --- (c) D regimes vs ACF_6h — clean separation ---
ax = axes[1, 0]
data = [df.loc[df["D"] == i, "acf_6h"].values for i in range(K_D)]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(bp["boxes"], colors_D):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels([f"$D_{{{i}}}$" for i in range(K_D)], fontsize=9)
ax.set_ylabel("ACF at lag 6h", fontsize=11)
ax.set_title("(c) Clear gradient", fontsize=12, fontweight="bold")

# --- (d) E regimes vs ACF_6h — no separation ---
ax = axes[1, 1]
data = [df.loc[df["E"] == i, "acf_6h"].values for i in range(K_E)]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(bp["boxes"], colors_E):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels([f"$E_{{{i}}}$" for i in range(K_E)], fontsize=9)
ax.set_ylabel("ACF at lag 6h", fontsize=11)
ax.set_title("(d) No separation", fontsize=12, fontweight="bold")

# Row labels on the right
fig.text(1.02, 0.73, "Price level", fontsize=14, fontweight="bold",
         rotation=270, va="center", ha="left")
fig.text(1.02, 0.28, "Persistence", fontsize=14, fontweight="bold",
         rotation=270, va="center", ha="left")

# Column labels on top
fig.text(0.27, 0.97, "Economic axis (E)", fontsize=13, fontweight="bold",
         ha="center", va="bottom")
fig.text(0.73, 0.97, "Dynamic axis (D)", fontsize=13, fontweight="bold",
         ha="center", va="bottom")

# Diagonal markers
axes[0, 0].patch.set_edgecolor("#2ecc71")
axes[0, 0].patch.set_linewidth(3)
axes[1, 0].patch.set_edgecolor("#2ecc71")
axes[1, 0].patch.set_linewidth(3)

fig.subplots_adjust(top=0.93, right=0.96, hspace=0.3, wspace=0.3)
fig.savefig("paper/axes_evidence.png", dpi=250, bbox_inches="tight", facecolor="white")
print("Saved paper/axes_evidence.png")
