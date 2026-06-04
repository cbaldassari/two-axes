"""
fig_eta2_scatter.py — Regenerate Figure 5: eta-squared separation diagnostic.
Labels use adjustText to avoid overlap.
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

try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False
    print("adjustText not installed, using manual offsets")

OUT = Path("results_darcsinh/split_W512_S6")
PAPER = Path("paper")

eta = pd.read_csv(OUT / "eta_squared.csv")

# Classify features
price_features = {"lmp_mean", "lmp_p95", "lmp_std"}
acf_features = {"acf_1h", "acf_6h", "acf_24h", "acf_168h"}

groups = []
for _, row in eta.iterrows():
    f = row["feature"]
    if f in acf_features:
        groups.append("ACF diagnostiche")
    elif f in price_features:
        groups.append("FE: prezzo")
    else:
        groups.append("FE: distribuzionali")
eta["group"] = groups

# Display names
display = {
    "mean": "media", "std": "std Δr", "skew": "asimm.", "kurt": "curtosi",
    "min": "min", "max": "max", "range": "range", "median": "mediana",
    "p5": "Q5", "p95": "Q95", "iqr": "IQR", "vol_24h": "vol 24h",
    "lmp_mean": "LMP medio", "lmp_p95": "LMP Q95", "lmp_std": "LMP std",
    "acf_1h": "ACF 1h", "acf_6h": "ACF 6h", "acf_24h": "ACF 24h", "acf_168h": "ACF 168h",
}

fig, ax = plt.subplots(figsize=(8, 7))

style = {
    "FE: distribuzionali": ("o", "steelblue", 50),
    "FE: prezzo": ("s", "forestgreen", 70),
    "ACF diagnostiche": ("^", "darkorange", 70),
}

for grp, (marker, color, size) in style.items():
    mask = eta["group"] == grp
    ax.scatter(eta.loc[mask, "eta2_FE"], eta.loc[mask, "eta2_MOM"],
               c=color, marker=marker, s=size, label=grp,
               alpha=0.85, edgecolors="white", lw=0.5, zorder=3)

# Diagonal
ax.plot([0, 0.55], [0, 0.55], "k--", lw=0.5, alpha=0.3, zorder=1)

# Labels
texts = []
for _, row in eta.iterrows():
    name = display.get(row["feature"], row["feature"])
    t = ax.annotate(name, (row["eta2_FE"], row["eta2_MOM"]),
                    fontsize=7.5, color="#333333", zorder=4)
    texts.append(t)

if HAS_ADJUST:
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4),
                expand=(1.8, 1.8), force_text=(0.8, 0.8))

ax.set_xlabel("$\\eta^2$ rispetto alla partizione FE", fontsize=11)
ax.set_ylabel("$\\eta^2$ rispetto alla partizione MOMENT", fontsize=11)
ax.set_title("Diagnostica di separazione: pattern diagonale", fontsize=12)
ax.legend(frameon=True, fontsize=9, loc="center")
ax.set_xlim(-0.02, 0.55)
ax.set_ylim(-0.02, 0.48)
ax.set_aspect("equal")
ax.grid(alpha=0.1)

fig.tight_layout()
out = PAPER / "eta2_scatter.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close(fig)
