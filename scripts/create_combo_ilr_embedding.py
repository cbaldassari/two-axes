"""
create_combo_ilr_embedding.py
=============================
Approccio D aggiornato: FE(19D) + MOMENT PCA-MP(55D) + ILR detrended annuale(7D) = 81D

Lanciare DOPO che step01 ha completato il MSTL annuale per ILR.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
sys.path.insert(0, ".")
from utils.pca_dim_selector import marchenko_pastur_n_components

# MOMENT -> MP PCA
emb_m = pd.read_parquet("results/exp_C/embeddings.parquet")
ts = pd.to_datetime(emb_m["datetime"])
E_m = StandardScaler().fit_transform(emb_m.drop(columns=["datetime"]).values)
n_mp, _ = marchenko_pastur_n_components(
    PCA(random_state=42).fit(E_m).explained_variance_, *E_m.shape
)
E_m_pca = PCA(n_components=n_mp, random_state=42).fit_transform(E_m)
print(f"MOMENT PCA: {n_mp}D")

# Features
emb_f = pd.read_parquet("results/exp_FE/embeddings.parquet")
E_f = emb_f.drop(columns=["datetime"]).values.astype(np.float32)
print(f"Features: {E_f.shape[1]}D")

# ILR detrended with annual MSTL
pre = pd.read_parquet("results/preprocessed.parquet")
pre["datetime"] = pd.to_datetime(pre["datetime"])
ilr_cols = [f"mstl_resid_ilr_{i}" for i in range(1, 8)]
ilr_lookup = pre.set_index("datetime")[ilr_cols]
ilr_vals = ilr_lookup.reindex(ts).values.astype(np.float32)
for j in range(ilr_vals.shape[1]):
    mask = np.isnan(ilr_vals[:, j])
    if mask.any():
        ilr_vals[mask, j] = np.nanmedian(ilr_vals[:, j])
print(f"ILR detrended (annual): {ilr_vals.shape[1]}D")

# Concatenate + single StandardScaler
E_all = np.concatenate([E_f, E_m_pca, ilr_vals], axis=1)
E_final = StandardScaler().fit_transform(E_all)
print(f"Combined: {E_f.shape[1]} + {n_mp} + {ilr_vals.shape[1]} = {E_final.shape[1]}D")

out_dir = Path("results/exp_D2")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "step03").mkdir(exist_ok=True)
(out_dir / "step04").mkdir(exist_ok=True)
cols = [f"emb_{i}" for i in range(E_final.shape[1])]
df = pd.DataFrame(E_final, columns=cols)
df.insert(0, "datetime", ts.values)
df.to_parquet(out_dir / "embeddings.parquet", index=False)
print(f"Saved: {out_dir}/embeddings.parquet")
