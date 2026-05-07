"""
mstl_annual_ray.py
==================
Esegue MSTL(24h+168h+8760h) su tutti i canali in parallelo via Ray cluster.
Ogni canale = 1 task remoto. Serializzazione via bytes per evitare problemi
di compatibilita numpy tra client Windows e worker Linux.

Usage: python scripts/mstl_annual_ray.py
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PREPROC_PATH = Path("results/preprocessed.parquet")
PERIODS = [24, 168, 8760]
WINDOWS = [25, 169, 8761]

CHANNELS = [
    ("arcsinh_lmp", "mstl_resid_arcsinh", False),
    ("log_return",  "mstl_resid_lr",      False),
    ("ilr_1", "mstl_resid_ilr_1", True),
    ("ilr_2", "mstl_resid_ilr_2", True),
    ("ilr_3", "mstl_resid_ilr_3", True),
    ("ilr_4", "mstl_resid_ilr_4", True),
    ("ilr_5", "mstl_resid_ilr_5", True),
    ("ilr_6", "mstl_resid_ilr_6", True),
    ("ilr_7", "mstl_resid_ilr_7", True),
]


def fit_mstl_remote(series_bytes, shape, dtype_str, has_nan, periods, windows):
    """
    Remote MSTL fit. Receives and returns raw bytes to avoid numpy pickle issues.
    """
    import numpy as np
    from statsmodels.tsa.seasonal import MSTL
    import time

    series = np.frombuffer(series_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    nan_mask = np.isnan(series)
    fill_val = np.nanmedian(series) if nan_mask.any() else 0.0
    series[nan_mask] = fill_val

    t0 = time.time()
    res = MSTL(series, periods=periods, windows=windows).fit()
    resid = res.resid.copy()
    if nan_mask.any():
        resid[nan_mask] = np.nan

    elapsed = time.time() - t0
    std = float(np.nanstd(resid))

    return resid.tobytes(), resid.shape, resid.dtype.str, std, elapsed


def main():
    import ray

    print("=" * 65)
    print("  MSTL Annual — Ray Cluster")
    print("=" * 65)
    print(f"  Periods: {PERIODS}")
    print(f"  Windows: {WINDOWS}")
    print(f"  Channels: {len(CHANNELS)}")

    # Load data locally
    pre = pd.read_parquet(PREPROC_PATH)
    print(f"  Loaded: {len(pre)} rows")

    # Connect to Ray
    ray.init(
        "ray://datalab-rayclnt.unitus.it:10001",
        ignore_reinit_error=True,
        runtime_env={"pip": ["statsmodels"]},
    )

    cluster_cpus = int(ray.cluster_resources().get("CPU", 0))
    print(f"  Ray cluster: {cluster_cpus} CPUs")

    # Define remote function
    fit_remote = ray.remote(num_cpus=2)(fit_mstl_remote)

    # Submit all channels
    t0 = time.time()
    futures = {}

    for src_col, dst_col, is_ilr in CHANNELS:
        series = pre[src_col].values.astype(np.float64)
        payload = (series.tobytes(), series.shape, series.dtype.str,
                   bool(np.isnan(series).any()), PERIODS, WINDOWS)
        ref = ray.put(payload)
        future = fit_remote.remote(*payload)
        futures[future] = (src_col, dst_col)
        print(f"  Submitted: {src_col} -> {dst_col}", flush=True)

    print(f"\n  {len(futures)} tasks submitted. Waiting...\n", flush=True)

    # Collect results as they complete
    done = 0
    while futures:
        ready, _ = ray.wait(list(futures.keys()), num_returns=1, timeout=60)

        if not ready:
            elapsed = time.time() - t0
            print(f"  ... waiting ({elapsed/60:.0f} min elapsed, {done}/{len(CHANNELS)} done)",
                  flush=True)
            continue

        for ref in ready:
            src_col, dst_col = futures.pop(ref)
            try:
                raw_bytes, shape, dtype_str, std, ch_elapsed = ray.get(ref)
                resid = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
                pre[dst_col] = resid
                done += 1
                print(f"  [{done}/{len(CHANNELS)}] {dst_col}: std={std:.4f}  "
                      f"({ch_elapsed/60:.1f} min)", flush=True)
            except Exception as e:
                print(f"  FAILED {dst_col}: {e}", flush=True)

    # Save
    pre.to_parquet(PREPROC_PATH, index=False)
    ray.shutdown()

    elapsed = time.time() - t0
    print(f"\n  All done in {elapsed/60:.1f} min")
    print(f"  Saved: {PREPROC_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
