"""
step02_embeddings.py
====================
NEPOOL TDA Pipeline -- Step 02: MOMENT Multi-Channel Embedding

Input  : results/preprocessed_windows.parquet  (1 row/day, 8 channels x 512 padded)
         results/preprocessing_meta.json        (channel list, context_len)
Output : results/embeddings.parquet             (1 row/day, 8 x 1024D = 8192 cols)
Plots  : results/step02/

Each of the 8 channels (1 arcsinh-LMP + 7 ILR) is embedded independently
by MOMENT-1-large in embedding mode. MOMENT does NOT learn cross-channel
relationships -- that is handled downstream by cross-channel cosine similarity
in step03.

Input format per channel: (batch, 1, 512) with input_mask indicating the
last 24 positions as real data (the rest is zero-padding).

Install: pip install --no-deps momentfm
"""

import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

RESULTS_DIR = Path(C.RESULTS_DIR)
PLOT_DIR    = RESULTS_DIR / "step02"
BATCH_SIZE  = C.EMBEDDING["batch_size"]


# =====================================================================
#  MOMENT encoder
# =====================================================================

def load_moment(device: str):
    """Load MOMENT in embedding mode. Returns (model, d_model)."""
    import torch
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        C.EMBEDDING["model"],
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model = model.to(device)
    model.eval()

    # Infer d_model
    with torch.no_grad():
        dummy = torch.randn(1, 1, C.EMBEDDING["context_len"], device=device)
        mask = torch.zeros(1, C.EMBEDDING["context_len"], device=device)
        mask[:, -C.EMBEDDING["window_h"]:] = 1.0
        out = model(x_enc=dummy, input_mask=mask)
        d_model = out.embeddings.shape[-1]

    return model, d_model


def embed_channel(model, windows: np.ndarray, device: str,
                  window_h: int = 24) -> np.ndarray:
    """
    Embed all daily windows for a single channel.

    windows : (N, context_len) float32 -- zero-padded, last window_h are real
    returns : (N, d_model) float32
    """
    import torch

    N, T = windows.shape
    context_len = T

    # Input mask: 1 for real data (last window_h positions), 0 for padding
    mask_template = np.zeros(context_len, dtype=np.float32)
    mask_template[-window_h:] = 1.0

    results = []
    for start in range(0, N, BATCH_SIZE):
        batch = windows[start:start + BATCH_SIZE]
        B = len(batch)

        x = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)  # (B, 1, T)
        mask = torch.tensor(
            np.tile(mask_template, (B, 1)), dtype=torch.float32, device=device
        )  # (B, T)

        with torch.no_grad():
            out = model(x_enc=x, input_mask=mask)
        results.append(out.embeddings.float().cpu().numpy())

    return np.concatenate(results, axis=0)


# =====================================================================
#  Diagnostic plots
# =====================================================================

def make_plots(dates: np.ndarray, all_embeddings: dict[str, np.ndarray]) -> None:
    """Post-extraction diagnostics."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "savefig.facecolor": "white"})

    channels = list(all_embeddings.keys())

    # 01: Per-channel embedding norms over time
    fig, ax = plt.subplots(figsize=(14, 5))
    for ch in channels:
        norms = np.linalg.norm(all_embeddings[ch], axis=1)
        label = ch.replace("mstl_resid_", "")
        ax.plot(dates, norms, lw=0.5, alpha=0.7, label=label)
    ax.set_title("Per-channel embedding L2-norm over time", fontweight="bold")
    ax.set_ylabel("||emb||_2")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_channel_norms.png")
    plt.close(fig)

    # 02: Embedding variance per channel
    fig, ax = plt.subplots(figsize=(10, 5))
    ch_vars = []
    ch_labels = []
    for ch in channels:
        v = all_embeddings[ch].var(axis=0).mean()
        ch_vars.append(v)
        ch_labels.append(ch.replace("mstl_resid_", ""))
    ax.bar(range(len(ch_vars)), ch_vars, color=plt.cm.tab10(np.linspace(0, 1, len(ch_vars))))
    ax.set_xticks(range(len(ch_labels)))
    ax.set_xticklabels(ch_labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean variance per dimension")
    ax.set_title("Embedding variance per channel", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_channel_variance.png")
    plt.close(fig)

    print(f"  Plots saved -> {PLOT_DIR}/", flush=True)


# =====================================================================
#  Main
# =====================================================================

def main():
    t0 = time.time()

    print("\n" + "=" * 65)
    print("  NEPOOL TDA -- Step 02: MOMENT Multi-Channel Embedding")
    print("=" * 65)

    # Load metadata
    meta_path = RESULTS_DIR / "preprocessing_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    channels = meta["channels"]
    context_len = meta["context_len"]
    window_h = meta["window_h"]

    print(f"  Model    : {C.EMBEDDING['model']}")
    print(f"  Channels : {len(channels)}")
    print(f"  Context  : {context_len} (window={window_h}h, padded)")
    print("-" * 65)

    # ── 1. Load windows ────────────────────────────────────────────
    print(f"\n  [1/4] Load daily windows ...", flush=True)
    windows_df = pd.read_parquet(RESULTS_DIR / "preprocessed_windows.parquet")
    dates = pd.to_datetime(windows_df["date"]).values
    N = len(windows_df)
    print(f"  [1/4] Done ({N} days, {len(channels)} channels)", flush=True)

    # ── 2. Load MOMENT ─────────────────────────────────────────────
    print(f"\n  [2/4] Load MOMENT ...", flush=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, d_model = load_moment(device)
    print(f"  [2/4] Done (d_model={d_model}, device={device})", flush=True)

    # ── 3. Embed all channels ──────────────────────────────────────
    print(f"\n  [3/4] Embed {len(channels)} channels x {N} days ...", flush=True)
    all_embeddings = {}

    for ch in channels:
        col_name = f"{ch}_window"
        if col_name not in windows_df.columns:
            print(f"    WARNING: {col_name} not found, skipping", flush=True)
            continue

        # Extract windows array
        windows = np.stack(windows_df[col_name].values).astype(np.float32)

        t_ch = time.time()
        emb = embed_channel(model, windows, device, window_h=window_h)
        elapsed_ch = time.time() - t_ch

        all_embeddings[ch] = emb
        short_name = ch.replace("mstl_resid_", "")
        print(f"    {short_name}: ({N}, {d_model})  "
              f"norm={np.linalg.norm(emb, axis=1).mean():.2f}  "
              f"{elapsed_ch:.1f}s", flush=True)

    # Sanity check
    for ch, emb in all_embeddings.items():
        nan_count = np.isnan(emb).sum()
        inf_count = np.isinf(emb).sum()
        if nan_count > 0 or inf_count > 0:
            raise ValueError(f"Channel {ch}: {nan_count} NaN, {inf_count} Inf")

    total_dim = sum(e.shape[1] for e in all_embeddings.values())
    print(f"\n  Total embedding dim: {len(channels)} x {d_model} = {total_dim}D", flush=True)
    print(f"  [3/4] Done", flush=True)

    # ── 4. Save ────────────────────────────────────────────────────
    print(f"\n  [4/4] Save ...", flush=True)

    # Build wide DataFrame: date + emb_{channel}_{dim}
    emb_df = pd.DataFrame({"date": dates})
    for ch in channels:
        emb = all_embeddings[ch]
        short = ch.replace("mstl_resid_", "")
        cols = [f"emb_{short}_{i}" for i in range(emb.shape[1])]
        emb_df = pd.concat([emb_df, pd.DataFrame(emb, columns=cols)], axis=1)

    out_path = RESULTS_DIR / "embeddings.parquet"
    emb_df.to_parquet(out_path, index=False)
    print(f"  [4/4] Done ({out_path.stat().st_size / 1e6:.1f} MB -> {out_path})", flush=True)

    # Plots
    print(f"\n  Plots ...", flush=True)
    make_plots(dates, all_embeddings)

    # Report
    elapsed = time.time() - t0
    print("\n" + "-" * 65)
    print("  EMBEDDING REPORT")
    print("-" * 65)
    print(f"  Model           : {C.EMBEDDING['model']}")
    print(f"  Days            : {N:>8,}")
    print(f"  Channels        : {len(channels):>8}")
    print(f"  d_model         : {d_model:>8}")
    print(f"  Total dim       : {total_dim:>8,}")
    print(f"  File size       : {out_path.stat().st_size / 1e6:>7.1f} MB")
    print(f"  Elapsed         : {elapsed:>7.1f}s")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
