"""
axes_evidence_v2.py — Regenerate axes_evidence with ACF-merged MOMENT labels.
Also recompute quadrant populations.
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10

RESULTS = Path("results")

# Load
lab_fe = pd.read_parquet(RESULTS / "exp_FE/step04/labels.parquet")
lab_mom = pd.read_parquet(RESULTS / "exp_C/step04/labels_acf_merge.parquet")  # NEW
lab_fe["datetime"] = pd.to_datetime(lab_fe["datetime"])
lab_mom["datetime"] = pd.to_datetime(lab_mom["datetime"])

fe_emb = pd.read_parquet(RESULTS / "exp_FE/embeddings.parquet")
fe_emb["datetime"] = pd.to_datetime(fe_emb["datetime"])

pre = pd.read_parquet(RESULTS / "preprocessed.parquet")
pre["datetime"] = pd.to_datetime(pre["datetime"])

df = lab_fe.merge(lab_mom, on="datetime", suffixes=("_fe", "_mom"))
df = df.merge(pre[["datetime", "lmp"]], on="datetime")
df = df.merge(fe_emb[["datetime", "acf_6h"]], on="datetime")

# Order: FE by LMP, MOMENT by ACF (each on its natural variable)
fe_order = df.groupby("cluster_fe")["lmp"].mean().sort_values().index.tolist()
fe_rank = {r: i for i, r in enumerate(fe_order)}
df["fe_rank"] = df["cluster_fe"].map(fe_rank)

mom_order = df.groupby("cluster_mom")["acf_6h"].mean().sort_values().index.tolist()
mom_rank = {r: i for i, r in enumerate(mom_order)}
df["mom_rank"] = df["cluster_mom"].map(mom_rank)

K_fe = len(fe_order)
K_mom = len(mom_order)

# =====================================================
# FIGURE: Why each axis separates what the other cannot
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

colors_fe = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, K_fe))
colors_mom = plt.cm.coolwarm(np.linspace(0.15, 0.85, K_mom))

# (a) FE regimes vs LMP — clean separation
ax = axes[0, 0]
data = [df.loc[df["fe_rank"] == i, "lmp"].values for i in range(K_fe)]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(bp["boxes"], colors_fe):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels([f"F{i}" for i in range(K_fe)], fontsize=8)
ax.set_ylabel("LMP ($/MWh)")
ax.set_title("(a) FE regimes separate price levels cleanly", fontweight="bold")

# (b) MOMENT regimes vs LMP — overlap
ax = axes[0, 1]
data = [df.loc[df["mom_rank"] == i, "lmp"].values for i in range(K_mom)]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(bp["boxes"], colors_mom):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels([f"M{i}" for i in range(K_mom)], fontsize=8)
ax.set_ylabel("LMP ($/MWh)")
ax.set_title("(b) MOMENT regimes: price distributions overlap", fontweight="bold")

# (c) MOMENT regimes vs ACF_6h — clean separation
ax = axes[1, 0]
data = [df.loc[df["mom_rank"] == i, "acf_6h"].values for i in range(K_mom)]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(bp["boxes"], colors_mom):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels([f"M{i}" for i in range(K_mom)], fontsize=8)
ax.set_ylabel("ACF at 6h lag")
ax.set_title("(c) MOMENT regimes separate persistence cleanly", fontweight="bold")

# (d) FE regimes vs ACF_6h — no separation
ax = axes[1, 1]
data = [df.loc[df["fe_rank"] == i, "acf_6h"].values for i in range(K_fe)]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(bp["boxes"], colors_fe):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels([f"F{i}" for i in range(K_fe)], fontsize=8)
ax.set_ylabel("ACF at 6h lag")
ax.set_title("(d) FE regimes do NOT separate persistence", fontweight="bold")

fig.suptitle("Each axis separates what the other cannot",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("paper/axes_evidence.png", dpi=250, bbox_inches="tight", facecolor="white")
print("Saved paper/axes_evidence.png")

# Quadrant stats
high_p = df["lmp"] > 60
high_a = df["acf_6h"] > 0.6
print("\n=== QUADRANT POPULATIONS ===")
print(f"Low price + Fast-reverting:  {(~high_p & ~high_a).sum():>5d} ({(~high_p & ~high_a).mean()*100:.1f}%)")
print(f"Low price + Sticky:          {(~high_p & high_a).sum():>5d} ({(~high_p & high_a).mean()*100:.1f}%)")
print(f"High price + Fast-reverting: {(high_p & ~high_a).sum():>5d} ({(high_p & ~high_a).mean()*100:.1f}%)")
print(f"High price + Sticky:         {(high_p & high_a).sum():>5d} ({(high_p & high_a).mean()*100:.1f}%)")
