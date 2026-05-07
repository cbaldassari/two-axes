"""
create_ilr_embedding.py
=======================
Approccio E: ILR-only embedding.

7 canali ILR destagionalizzati (MSTL 24h+168h+8760h) -> statistiche rolling
sulla finestra 512h -> 28D per finestra.

Per ogni canale ILR: mean, std, skew sulla finestra + valore puntuale a fine finestra.
7 canali x 4 statistiche = 28 feature.

Input  : results/preprocessed.parquet (con MSTL annuale)
Output : results/exp_ILR/embeddings.parquet

Lanciare DOPO che step01 ha completato il MSTL annuale per ILR.
"""

import numpy as np
import pandas as pd
from pathlib import Path

pre = pd.read_parquet("results/preprocessed.parquet")
pre["datetime"] = pd.to_datetime(pre["datetime"])

ilr_cols = [f"mstl_resid_ilr_{i}" for i in range(1, 8)]
dt = pre["datetime"].values
T = len(pre)

CONTEXT = 512
STRIDE = 6
GAP_FROM = pd.Timestamp("2023-06-14 23:00:00")
GAP_TO = pd.Timestamp("2023-06-16 00:00:00")

windows = []
timestamps = []

for i in range(0, T - CONTEXT + 1, STRIDE):
    end = i + CONTEXT - 1
    t_s = pd.Timestamp(dt[i])
    t_e = pd.Timestamp(dt[end])
    if t_s <= GAP_FROM and t_e >= GAP_TO:
        continue
    if abs((t_e - t_s).total_seconds() / 3600 - (CONTEXT - 1)) > 2:
        continue

    feat = {}
    for col in ilr_cols:
        w = pre[col].values[i:i + CONTEXT].astype(np.float64)
        w_clean = np.nan_to_num(w, nan=0.0)
        short = col.replace("mstl_resid_", "")

        # Rolling statistics on the window
        feat[f"{short}_mean"] = np.mean(w_clean)
        feat[f"{short}_std"] = np.std(w_clean)
        feat[f"{short}_skew"] = float(pd.Series(w_clean).skew())

        # Point-in-time value at window end
        val = pre[col].values[end]
        feat[f"{short}_end"] = val if not np.isnan(val) else 0.0

    windows.append(feat)
    timestamps.append(t_e)

feat_df = pd.DataFrame(windows)
feat_df.insert(0, "datetime", pd.to_datetime(timestamps))

# Rename columns for step03 compatibility
for col in list(feat_df.columns[1:]):
    feat_df = feat_df.rename(columns={col: f"emb_{col}"})

out_dir = Path("results/exp_ILR")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "step03").mkdir(exist_ok=True)
(out_dir / "step04").mkdir(exist_ok=True)
feat_df.to_parquet(out_dir / "embeddings.parquet", index=False)

N = len(feat_df)
D = feat_df.shape[1] - 1
print(f"ILR-only embedding: {N} windows x {D} features")
print(f"Saved: {out_dir}/embeddings.parquet")
