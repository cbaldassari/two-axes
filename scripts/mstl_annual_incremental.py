"""
mstl_annual_incremental.py
==========================
Esegue MSTL(24h+168h+8760h) un canale alla volta, salvando dopo ogni canale.
Se interrotto, riparte dal canale non ancora completato.

Usage: python scripts/mstl_annual_incremental.py
"""

import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from statsmodels.tsa.seasonal import MSTL

PREPROC_PATH = Path("results/preprocessed.parquet")
PROGRESS_PATH = Path("results/mstl_annual_progress.json")
PERIODS = [24, 168, 8760]
WINDOWS = [25, 169, 8761]

# All channels to process
CHANNELS = [
    ("arcsinh_lmp", "mstl_resid_arcsinh"),
    ("log_return",  "mstl_resid_lr"),
] + [
    (f"ilr_{i}", f"mstl_resid_ilr_{i}") for i in range(1, 8)
]


def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed": []}


def save_progress(progress):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    print("MSTL Annual Incremental")
    print(f"Periods: {PERIODS}")
    print(f"Windows: {WINDOWS}")
    print(f"Channels: {len(CHANNELS)}")
    print()

    progress = load_progress()
    completed = set(progress["completed"])

    pre = pd.read_parquet(PREPROC_PATH)
    print(f"Loaded: {len(pre)} rows")

    for src_col, dst_col in CHANNELS:
        if dst_col in completed:
            print(f"  SKIP {dst_col} (already done)")
            continue

        print(f"  Processing {src_col} -> {dst_col} ...", flush=True)
        t0 = time.time()

        series = pre[src_col].values.copy()
        nan_mask = np.isnan(series)
        fill_val = np.nanmedian(series) if nan_mask.any() else 0.0
        series[nan_mask] = fill_val

        res = MSTL(series, periods=PERIODS, windows=WINDOWS).fit()
        resid = res.resid.copy()
        if nan_mask.any():
            resid[nan_mask] = np.nan

        pre[dst_col] = resid
        elapsed = time.time() - t0
        print(f"  DONE {dst_col}: std={np.nanstd(resid):.4f}  ({elapsed:.0f}s)", flush=True)

        # Save after each channel
        pre.to_parquet(PREPROC_PATH, index=False)
        progress["completed"].append(dst_col)
        save_progress(progress)
        print(f"  Saved checkpoint ({len(progress['completed'])}/{len(CHANNELS)})", flush=True)
        print()

    # Cleanup
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
    print(f"All {len(CHANNELS)} channels done.")


if __name__ == "__main__":
    main()
