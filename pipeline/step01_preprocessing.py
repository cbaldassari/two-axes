"""
step01_preprocessing.py
=======================
NEPOOL TDA Pipeline -- Step 01: Preprocessing

Input  : isone_dataset.parquet
Output : results/preprocessed.parquet
Plots  : results/step01/

Features:
  lmp, arcsinh_lmp, log_return,
  mstl_resid_arcsinh (MSTL 24h+168h),
  mstl_resid_lr (MSTL 24h+168h),
  ilr_1..7 (SBP ILR),
  mstl_resid_ilr_1..7 (MSTL 24h+168h+8760h for ILR, 24h+168h for price)
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

PLOT_DIR = Path(C.RESULTS_DIR) / "step01"

FUEL_DISPLAY = ["Gas", "Nuclear", "Hydro", "Wind", "Solar", "Coal", "Oil", "Other"]
FUEL_COLORS  = ["#e07b39", "#7b52ab", "#4da6e8", "#6abf69", "#f7c948",
                "#5a5a5a", "#b04040", "#aaaaaa"]


# =====================================================================
#  Feature builders
# =====================================================================

def build_price_features(df):
    lmp = df["lmp"].values.astype(np.float64)
    arcsinh_lmp = np.arcsinh(lmp)
    log_ret = np.empty(len(lmp))
    log_ret[0] = np.nan
    log_ret[1:] = np.diff(arcsinh_lmp)
    return pd.DataFrame({
        "lmp": lmp,
        "arcsinh_lmp": arcsinh_lmp,
        "log_return": log_ret,
    }, index=df.index)


def build_ilr_features(df):
    """SBP ILR from 8 fuel shares."""
    delta = C.ILR["zero_replacement_delta"]
    fuel_cols = [c for c in C.ILR["fuel_cols"] if c in df.columns]
    shares = df[fuel_cols].fillna(0.0).values.astype(np.float64)
    all_zero = shares.sum(axis=1) == 0
    shares[shares <= 0] = delta
    shares = shares / shares.sum(axis=1, keepdims=True)
    log_s = np.log(shares)

    GAS, NUC, HYD, WIN, SOL, COA, OIL, OTH = range(8)

    def balance(pos, neg):
        r_p, r_n = len(pos), len(neg)
        coeff = np.sqrt(r_p * r_n / (r_p + r_n))
        return coeff * (log_s[:, pos].mean(axis=1) - log_s[:, neg].mean(axis=1))

    ilr = pd.DataFrame({
        "ilr_1": balance([GAS, NUC, COA, OIL, OTH], [HYD, WIN, SOL]),
        "ilr_2": balance([GAS, COA, OIL], [NUC, OTH]),
        "ilr_3": balance([GAS], [COA, OIL]),
        "ilr_4": balance([COA], [OIL]),
        "ilr_5": balance([NUC], [OTH]),
        "ilr_6": balance([HYD], [WIN, SOL]),
        "ilr_7": balance([SOL], [WIN]),
    }, index=df.index)
    ilr.loc[all_zero] = np.nan
    return ilr


# =====================================================================
#  Plots
# =====================================================================

def make_plots(out, df_raw):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    dt = pd.to_datetime(out["datetime"])
    lmp = out["lmp"].values

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dt, lmp, lw=0.4, color="#2c6fad", alpha=0.8)
    ax.axhline(np.percentile(lmp, 99), color="red", lw=0.8, ls="--")
    ax.set_title("ISONE LMP 2021-2025"); ax.set_ylabel("$/MWh")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "01_lmp_timeseries.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lmp, bins=120, color="#2c6fad", alpha=0.75, edgecolor="none")
    ax.set_title("LMP distribution"); fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_lmp_distribution.png"); plt.close(fig)

    # Fuel mix monthly
    df2 = df_raw.copy()
    df2["month"] = pd.to_datetime(df2["datetime"]).dt.to_period("M")
    share_cols = [c for c in C.ILR["fuel_cols"] if c in df2.columns]
    if share_cols:
        monthly = df2.groupby("month")[share_cols].mean()
        monthly.index = monthly.index.astype(str)
        fig, ax = plt.subplots(figsize=(16, 5))
        bottom = np.zeros(len(monthly))
        for i, col in enumerate(share_cols):
            vals = monthly[col].values
            ax.bar(range(len(monthly)), vals, bottom=bottom,
                   color=FUEL_COLORS[i], label=FUEL_DISPLAY[i], width=0.85)
            bottom += vals
        ax.set_xticks(range(0, len(monthly), 3))
        ax.set_xticklabels(list(monthly.index)[::3], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Share"); ax.legend(ncol=4, fontsize=8)
        ax.set_title("Monthly fuel mix"); fig.tight_layout()
        fig.savefig(PLOT_DIR / "04_fuel_mix_monthly.png"); plt.close(fig)

    print(f"  Plots saved -> {PLOT_DIR}/")


# =====================================================================
#  Main
# =====================================================================

def main():
    t0 = time.time()

    print("\n" + "=" * 65)
    print("  NEPOOL TDA -- Step 01: Preprocessing")
    print("=" * 65)
    print(f"  Input  : {C.DATA_PATH}")
    print("-" * 65)

    # 1. Load & clean
    print(f"\n  [1/5] Load & clean ...", flush=True)
    df = (pd.read_parquet(C.DATA_PATH)
            .sort_values("datetime").reset_index(drop=True))
    df["datetime"] = pd.to_datetime(df["datetime"])
    n_raw = len(df)

    df = df[(df["datetime"] >= pd.Timestamp(C.DATASET["start_date"])) &
            (df["datetime"] < pd.Timestamp(C.DATASET["end_date"]) + pd.Timedelta(days=1))
            ].reset_index(drop=True)
    df = df[~df.duplicated(subset=["datetime"], keep="first")].reset_index(drop=True)

    mw_cols = [c for c in ["hydro","natural_gas","nuclear","oil","other","solar","wind","coal"]
               if c in df.columns]
    if mw_cols:
        df = df[~(df[mw_cols] < 0).any(axis=1)].reset_index(drop=True)

    df_raw = df.copy()
    print(f"  [1/5] Done ({n_raw:,} -> {len(df):,} rows)", flush=True)

    # 2. Price features
    print(f"\n  [2/5] Price features ...", flush=True)
    price = build_price_features(df)
    print(f"  [2/5] Done", flush=True)

    # 3. ILR
    print(f"\n  [3/5] ILR SBP (8 fuel -> 7 coords) ...", flush=True)
    ilr = build_ilr_features(df)
    print(f"  [3/5] Done ({ilr.isnull().any(axis=1).sum()} NaN rows)", flush=True)

    # 4. Assembly
    out = pd.concat([df[["datetime"]], price, ilr], axis=1)
    valid = out["arcsinh_lmp"].notna()
    out = out[valid].reset_index(drop=True)

    # 5. MSTL
    print(f"\n  [4/5] MSTL deseasonalization ...", flush=True)
    print(f"    Price: periods=[24, 168]", flush=True)
    print(f"    ILR:   periods=[24, 168, 8760] (+ annual cycle)", flush=True)
    from statsmodels.tsa.seasonal import MSTL
    from joblib import Parallel, delayed

    # All channels: MSTL 24h + 168h + 8760h (daily + weekly + annual)
    # Annual: strong for ILR (ACF=0.40), weak for LMP (ACF=0.11) but included for consistency
    price_jobs = [
        ("arcsinh_lmp", "mstl_resid_arcsinh", [24, 168, 8760], [25, 169, 8761]),
        ("log_return",  "mstl_resid_lr",      [24, 168, 8760], [25, 169, 8761]),
    ]
    ilr_jobs = [
        (f"ilr_{i}", f"mstl_resid_ilr_{i}", [24, 168, 8760], [25, 169, 8761])
        for i in range(1, 8)
    ]

    def _fit_mstl(series_vals, nan_mask_vals, dst_col, periods, windows):
        from statsmodels.tsa.seasonal import MSTL as _MSTL
        res = _MSTL(series_vals, periods=periods, windows=windows,
                     stl_kwargs={"inner_iter": 1, "outer_iter": 0}).fit()
        resid = res.resid.copy()
        if nan_mask_vals is not None:
            resid[nan_mask_vals] = np.nan
        return dst_col, resid

    all_jobs = price_jobs + ilr_jobs
    # Pre-extract arrays for multiprocessing (avoids pickling the whole DataFrame)
    job_args = []
    for src_col, dst_col, periods, windows in all_jobs:
        is_ilr = src_col.startswith("ilr_")
        nan_mask_vals = out[src_col].isna().values if is_ilr else None
        fill_val = float(out[src_col].median()) if is_ilr else 0.0
        series_vals = out[src_col].fillna(fill_val).values.astype(np.float64)
        job_args.append((series_vals, nan_mask_vals, dst_col, periods, windows))

    gen = Parallel(n_jobs=-1, prefer="processes", return_as="generator")(
        delayed(_fit_mstl)(*args) for args in job_args
    )
    results = list(tqdm(gen, total=len(all_jobs), desc="  MSTL", unit="ch", ncols=70))
    for dst_col, resid in results:
        out[dst_col] = resid
        print(f"    {dst_col}  std={np.nanstd(resid):.4f}", flush=True)
    print(f"  [4/5] Done", flush=True)

    # 6. Save
    print(f"\n  [5/5] Save ...", flush=True)
    out_path = Path(C.RESULTS_DIR) / "preprocessed.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"  [5/5] Done ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    make_plots(out, df_raw)

    elapsed = time.time() - t0
    print(f"\n  {len(out):,} rows, {out.shape[1]} cols, {elapsed:.1f}s")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
