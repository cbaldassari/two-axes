"""
step02_representations.py
=========================
NEPOOL TDA Pipeline -- Step 02: Build 3 Representations

Creates sliding windows (512h context, 6h stride) and produces:
  exp_C/embeddings.parquet     — MOMENT embedding (1024D)
  exp_FE/embeddings.parquet    — 19 hand-crafted features
  exp_COMBO/embeddings.parquet — FE(19D) + MOMENT PCA-MP(~55D)

Input  : results/preprocessed.parquet
Output : results/exp_{C,FE,COMBO}/embeddings.parquet

Usage:
  python pipeline/step02_representations.py                # all 3
  python pipeline/step02_representations.py --only MOMENT  # just MOMENT
  python pipeline/step02_representations.py --only FE      # just FE
  python pipeline/step02_representations.py --skip-moment  # FE + COMBO (reuse MOMENT)
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
CONTEXT_LEN = C.EMBEDDING["context_len"]
STRIDE_H    = C.EMBEDDING["stride_h"]
SEED        = C.RANDOM_STATE


# =====================================================================
#  Sliding windows
# =====================================================================

def create_sliding_windows(df: pd.DataFrame):
    """
    Create sliding windows of CONTEXT_LEN hours with STRIDE_H stride.
    Returns (windows_resid, windows_lmp, timestamps).
      windows_resid : (N, CONTEXT_LEN) — mstl_resid_arcsinh
      windows_lmp   : (N, CONTEXT_LEN) — raw LMP
      timestamps    : (N,) — last hour of each window
    """
    resid = df["mstl_resid_arcsinh"].values.astype(np.float64)
    lmp   = df["lmp"].values.astype(np.float64)
    dt    = pd.to_datetime(df["datetime"]).values

    n = len(resid)
    starts = list(range(0, n - CONTEXT_LEN + 1, STRIDE_H))

    windows_r = np.empty((len(starts), CONTEXT_LEN), dtype=np.float32)
    windows_l = np.empty((len(starts), CONTEXT_LEN), dtype=np.float32)
    ts = np.empty(len(starts), dtype="datetime64[ns]")

    for i, s in enumerate(starts):
        windows_r[i] = resid[s:s + CONTEXT_LEN]
        windows_l[i] = lmp[s:s + CONTEXT_LEN]
        ts[i] = dt[s + CONTEXT_LEN - 1]

    return windows_r, windows_l, ts


# =====================================================================
#  MOMENT embedding
# =====================================================================

def run_moment(windows_resid: np.ndarray) -> np.ndarray:
    """
    Embed windows with MOMENT-1-large.
    Input:  (N, 512) float32
    Output: (N, 1024) float32
    """
    import torch
    from momentfm import MOMENTPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading MOMENT on {device} ...", flush=True)

    model = MOMENTPipeline.from_pretrained(
        C.EMBEDDING["model"],
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model = model.to(device)
    model.eval()

    N = len(windows_resid)
    batch_size = C.EMBEDDING["batch_size"]

    # Infer d_model
    with torch.no_grad():
        dummy = torch.randn(1, 1, CONTEXT_LEN, device=device)
        out = model(x_enc=dummy)
        d_model = out.embeddings.shape[-1]
    print(f"  d_model={d_model}", flush=True)

    embeddings = np.empty((N, d_model), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = windows_resid[start:end]
        x = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            out = model(x_enc=x)
        embeddings[start:end] = out.embeddings.float().cpu().numpy()

        if (start // batch_size) % 20 == 0:
            print(f"    {end}/{N} windows", flush=True)

    return embeddings


# =====================================================================
#  Feature Engineering (19 features)
# =====================================================================

def _acf(x, lag):
    """Autocorrelation at given lag."""
    n = len(x)
    if lag >= n:
        return 0.0
    m = x.mean()
    var = ((x - m) ** 2).sum()
    if var < 1e-15:
        return 0.0
    cov = ((x[:n - lag] - m) * (x[lag:] - m)).sum()
    return float(cov / var)


def compute_features(windows_resid: np.ndarray, windows_lmp: np.ndarray) -> np.ndarray:
    """
    Compute 19 features per window.
    First 16: statistics on mstl_resid_arcsinh.
    Last 3: lmp_mean, lmp_p95, lmp_std on raw LMP.
    """
    N = len(windows_resid)
    feats = np.empty((N, 19), dtype=np.float32)

    for i in range(N):
        r = windows_resid[i].astype(np.float64)
        l = windows_lmp[i].astype(np.float64)

        feats[i, 0]  = r.mean()
        feats[i, 1]  = r.std()
        feats[i, 2]  = float(skew(r))
        feats[i, 3]  = float(kurtosis(r))
        feats[i, 4]  = r.min()
        feats[i, 5]  = r.max()
        feats[i, 6]  = r.max() - r.min()
        feats[i, 7]  = float(np.median(r))
        feats[i, 8]  = float(np.percentile(r, 5))
        feats[i, 9]  = float(np.percentile(r, 95))
        feats[i, 10] = float(np.percentile(r, 75) - np.percentile(r, 25))
        feats[i, 11] = _acf(r, 1)
        feats[i, 12] = _acf(r, 6)
        feats[i, 13] = _acf(r, 24)
        feats[i, 14] = _acf(r, 168)
        # vol_24h: mean absolute intra-day change
        n_full = (len(r) // 24) * 24
        if n_full >= 24:
            daily_diffs = np.abs(np.diff(r[:n_full].reshape(-1, 24), axis=1))
            feats[i, 15] = float(daily_diffs.mean())
        else:
            feats[i, 15] = float(np.abs(np.diff(r)).mean())
        # Raw LMP statistics
        feats[i, 16] = l.mean()
        feats[i, 17] = float(np.percentile(l, 95))
        feats[i, 18] = l.std()

        if i % 2000 == 0 and i > 0:
            print(f"    {i}/{N} windows", flush=True)

    return feats


# =====================================================================
#  COMBO: FE + MOMENT PCA-MP
# =====================================================================

def create_combo(moment_emb: np.ndarray, fe_feats: np.ndarray) -> np.ndarray:
    """
    Combine FE (19D) + MOMENT compressed via PCA Marchenko-Pastur.
    Output: StandardScaled concatenation.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from utils.pca_dim_selector import marchenko_pastur_n_components

    # PCA-MP on MOMENT
    X_m = StandardScaler().fit_transform(moment_emb)
    n, p = X_m.shape
    pca_full = PCA(random_state=SEED).fit(X_m)
    n_comp, lam_max = marchenko_pastur_n_components(
        pca_full.explained_variance_, n, p
    )
    pca = PCA(n_components=n_comp, random_state=SEED)
    X_m_pca = pca.fit_transform(X_m)
    var_expl = pca.explained_variance_ratio_.sum()
    print(f"  MOMENT PCA-MP: {p}D -> {n_comp}D ({var_expl:.1%} variance)", flush=True)

    # Concatenate
    combo = np.hstack([fe_feats, X_m_pca])
    combo = StandardScaler().fit_transform(combo).astype(np.float32)
    print(f"  COMBO: {fe_feats.shape[1]} + {n_comp} = {combo.shape[1]}D", flush=True)

    return combo


# =====================================================================
#  Save helpers
# =====================================================================

def save_embedding(emb: np.ndarray, timestamps: np.ndarray,
                   exp_name: str, col_prefix: str):
    """Save embedding parquet with datetime + feature columns."""
    out_dir = RESULTS_DIR / f"exp_{exp_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [f"{col_prefix}_{i}" for i in range(emb.shape[1])]
    df = pd.DataFrame(emb, columns=cols)
    df.insert(0, "datetime", timestamps)

    out_path = out_dir / "embeddings.parquet"
    df.to_parquet(out_path, index=False)
    mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path} ({mb:.1f} MB, {emb.shape})", flush=True)


def save_fe_with_names(feats: np.ndarray, timestamps: np.ndarray):
    """Save FE with named columns."""
    out_dir = RESULTS_DIR / "exp_FE"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(feats, columns=C.FE_FEATURES)
    df.insert(0, "datetime", timestamps)

    out_path = out_dir / "embeddings.parquet"
    df.to_parquet(out_path, index=False)
    mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path} ({mb:.1f} MB, {feats.shape})", flush=True)


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Step 02 — Build representations")
    parser.add_argument("--only", choices=["MOMENT", "FE", "COMBO"],
                        help="Build only one representation")
    parser.add_argument("--skip-moment", action="store_true",
                        help="Skip MOMENT (reuse existing exp_C/embeddings.parquet)")
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "=" * 65)
    print("  NEPOOL TDA -- Step 02: Build Representations")
    print("=" * 65)

    # 1. Load preprocessed
    print(f"\n  [1/5] Load preprocessed ...", flush=True)
    df = pd.read_parquet(RESULTS_DIR / "preprocessed.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"  {len(df):,} rows", flush=True)

    # 2. Create sliding windows
    print(f"\n  [2/5] Sliding windows (ctx={CONTEXT_LEN}h, stride={STRIDE_H}h) ...",
          flush=True)
    win_r, win_l, timestamps = create_sliding_windows(df)
    N = len(timestamps)
    print(f"  {N:,} windows", flush=True)

    do_moment = args.only in (None, "MOMENT") and not args.skip_moment
    do_fe     = args.only in (None, "FE")
    do_combo  = args.only in (None, "COMBO")

    # 3. MOMENT
    moment_emb = None
    if do_moment:
        print(f"\n  [3/5] MOMENT embedding ...", flush=True)
        t_m = time.time()
        moment_emb = run_moment(win_r)
        print(f"  MOMENT done ({time.time() - t_m:.1f}s)", flush=True)
        save_embedding(moment_emb, timestamps, "C", "emb")
    elif do_combo or args.skip_moment:
        # Load existing MOMENT embedding for COMBO
        emb_path = RESULTS_DIR / "exp_C" / "embeddings.parquet"
        if emb_path.exists():
            print(f"\n  [3/5] Loading existing MOMENT from {emb_path} ...", flush=True)
            df_emb = pd.read_parquet(emb_path)
            ts_col = [c for c in df_emb.columns if "time" in c.lower() or "date" in c.lower()][0]
            moment_emb = df_emb.drop(columns=[ts_col]).values.astype(np.float32)
            print(f"  Loaded: {moment_emb.shape}", flush=True)
        else:
            print(f"  WARNING: {emb_path} not found, skipping COMBO", flush=True)
            do_combo = False
    else:
        print(f"\n  [3/5] MOMENT skipped", flush=True)

    # 4. Feature Engineering
    fe_feats = None
    if do_fe or do_combo:
        print(f"\n  [4/5] Feature engineering (19 features) ...", flush=True)
        t_f = time.time()
        fe_feats = compute_features(win_r, win_l)
        print(f"  FE done ({time.time() - t_f:.1f}s)", flush=True)
        if do_fe:
            save_fe_with_names(fe_feats, timestamps)
    else:
        print(f"\n  [4/5] FE skipped", flush=True)

    # 5. COMBO
    if do_combo and moment_emb is not None and fe_feats is not None:
        print(f"\n  [5/5] COMBO (FE + MOMENT PCA-MP) ...", flush=True)
        if len(moment_emb) != len(fe_feats):
            print(f"  ERROR: MOMENT ({len(moment_emb)}) != FE ({len(fe_feats)}) windows",
                  flush=True)
        else:
            combo = create_combo(moment_emb, fe_feats)
            save_embedding(combo, timestamps, "COMBO", "combo")
    else:
        print(f"\n  [5/5] COMBO skipped", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
