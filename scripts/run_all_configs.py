"""
run_all_configs.py
==================
Esegue tutte le 6 configurazioni (3 rappresentazioni x 2 clustering)
e produce la tabella di confronto finale.

Prerequisiti:
  - results/preprocessed.parquet aggiornato (con MSTL annuale)
  - results/exp_C/embeddings.parquet (MOMENT)
  - results/exp_FE/embeddings.parquet (Feature Engineering)
  - results/exp_COMBO/embeddings.parquet (Combinato)

Usage: python scripts/run_all_configs.py
"""

import subprocess
import sys
import time
import json
from pathlib import Path

PYTHON = sys.executable
PROJECT = Path(__file__).parent.parent

EXPERIMENTS = ["C", "FE", "COMBO"]
EXP_NAMES = {"C": "A:MOMENT", "FE": "B:FE", "COMBO": "C:COMBO"}


def run(script, args, label):
    cmd = [PYTHON, str(PROJECT / script)] + args
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}", flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(PROJECT))
    elapsed = time.time() - t0
    status = "OK" if ret.returncode == 0 else "FAIL"
    print(f"  [{status}] {label} ({elapsed:.1f}s)", flush=True)
    return ret.returncode == 0


def main():
    t_start = time.time()

    for exp in EXPERIMENTS:
        name = EXP_NAMES[exp]

        # Step 03: Diffusion Maps (shared input for both clustering methods)
        run("pipeline/step03_pca.py", ["--exp", exp], f"Step03 DiffMaps {name}")

        # Step 04a: ToMATo + Tukey
        run("pipeline/step04_tomato.py", ["--exp", exp], f"Step04 ToMATo {name}")

        # Step 04b: Ward + Tukey
        run("pipeline/step04_ward.py", ["--exp", exp], f"Step04 Ward {name}")

        # Step 05a: Quality for ToMATo
        run("pipeline/step05_cluster_quality.py", ["--exp", exp],
            f"Step05 Quality ToMATo {name}")

        # Step 05b: Quality for Ward (uses --labels-dir and --out-suffix)
        run("pipeline/step05_cluster_quality.py",
            ["--exp", exp, "--labels-dir", "step04_ward", "--out-suffix", "_ward"],
            f"Step05 Quality Ward {name}")

    # ── Collect results ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON TABLE")
    print("=" * 80)

    header = (f"{'Config':<20s} {'K':>3s} {'eta2':>7s} {'Sil':>7s} "
              f"{'Trans%':>7s} {'Sojourn':>8s} {'Tukey':>10s}")
    print(header)
    print("-" * 70)

    for exp in EXPERIMENTS:
        name = EXP_NAMES[exp]
        for method, quality_dir in [("ToMATo", "step05"), ("Ward", "step05_ward")]:
            quality_path = PROJECT / f"results/exp_{exp}/{quality_dir}/quality_report.json"

            if quality_path.exists():
                with open(quality_path) as f:
                    q = json.load(f)

                config = f"{name}+{method}"
                geo = q.get("geometric", {})
                eco = q.get("economic", {})

                K = len(eco.get("lmp_per_regime", {}))
                eta2 = eco.get("eta_squared", "?")
                sil = geo.get("silhouette_avg", "?")
                trans = eco.get("transition_rate", "?")
                sojourn = eco.get("sojourn_mean_hours", "?")
                tukey_sig = eco.get("pairwise_significant", "?")
                tukey_tot = eco.get("pairwise_total", "?")

                eta2_s = f"{eta2:.4f}" if isinstance(eta2, float) else str(eta2)
                sil_s  = f"{sil:.4f}" if isinstance(sil, float) else str(sil)
                trans_s = f"{trans:.4f}" if isinstance(trans, float) else str(trans)
                soj_s  = f"{sojourn:.0f}" if isinstance(sojourn, (int, float)) else str(sojourn)

                print(f"{config:<20s} {K:>3d} {eta2_s:>7s} {sil_s:>7s} "
                      f"{trans_s:>7s} {soj_s:>8s} {tukey_sig:>4s}/{tukey_tot:>4s}")
            else:
                print(f"{name}+{method:<8s}  — quality report not found")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
