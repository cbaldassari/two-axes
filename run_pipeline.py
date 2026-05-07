"""
run_pipeline.py
===============
NEPOOL TDA Pipeline — esecuzione locale, tutto da zero.

Pipeline:
  1. step01_preprocessing.py       — MSTL(24h+168h+8760h) su tutti i canali
  2. step02_representations.py     — MOMENT(1024D) + FE(19D) + COMBO(74D)
  3. step03_pca.py --exp {C,FE,COMBO}          — Diffusion Maps
  4. step04_tomato.py --exp {C,FE,COMBO}       — ToMATo + Tukey
     step04_ward.py --exp {C,FE,COMBO}         — Ward + Tukey
  5. step05_cluster_quality.py (ToMATo + Ward)  — metriche + plot

Uso
---
  python run_pipeline.py                   # tutto da zero
  python run_pipeline.py --from-step 2     # salta preprocessing
  python run_pipeline.py --from-step 3     # salta step01+02
  python run_pipeline.py --skip-moment     # riusa MOMENT embedding esistente
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON      = sys.executable
PROJECT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = PROJECT_DIR / "results"

EXPERIMENTS = ["C", "FE", "COMBO"]
EXP_NAMES   = {"C": "A:MOMENT", "FE": "B:FE", "COMBO": "C:COMBO"}


def run(script: str, args: list[str], label: str) -> bool:
    cmd = [PYTHON, str(PROJECT_DIR / script), *args]
    bar = "-" * 65
    print(f"\n{bar}\n  >> {label}\n{bar}", flush=True)
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    elapsed = time.time() - t0
    status = "OK" if ret.returncode == 0 else "FAIL"
    print(f"  [{status}] {label}  ({elapsed:.1f}s)", flush=True)
    return ret.returncode == 0


def print_table():
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON TABLE")
    print("=" * 80)

    header = (f"{'Config':<22s} {'K':>3s} {'eta2':>7s} {'Sil':>7s} "
              f"{'Trans%':>7s} {'Sojourn':>8s} {'Tukey':>10s}")
    print(header)
    print("-" * 72)

    for exp in EXPERIMENTS:
        name = EXP_NAMES[exp]
        for method, qdir in [("ToMATo", "step05"), ("Ward", "step05_ward")]:
            qpath = RESULTS_DIR / f"exp_{exp}" / qdir / "quality_report.json"
            if not qpath.exists():
                print(f"{name}+{method:<8s}  — not found")
                continue
            with open(qpath) as f:
                q = json.load(f)
            geo = q.get("geometric", {})
            eco = q.get("economic", {})
            K = len(eco.get("lmp_per_regime", {}))
            eta2 = eco.get("eta_squared", "?")
            sil = geo.get("silhouette_avg", "?")
            trans = eco.get("transition_rate", "?")
            soj = eco.get("sojourn_mean_hours", "?")
            tsig = eco.get("pairwise_significant", "?")
            ttot = eco.get("pairwise_total", "?")
            print(f"{name}+{method:<22s} {K:>3} "
                  f"{eta2 if isinstance(eta2, str) else f'{eta2:.3f}':>7s} "
                  f"{sil if isinstance(sil, str) else f'{sil:.3f}':>7s} "
                  f"{trans if isinstance(trans, str) else f'{trans:.2f}':>7s} "
                  f"{soj if isinstance(soj, str) else f'{soj:.0f}h':>8s} "
                  f"{tsig}/{ttot}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="NEPOOL TDA — full pipeline")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Start from step N (default: 1)")
    parser.add_argument("--skip-moment", action="store_true",
                        help="Reuse existing MOMENT embedding")
    args = parser.parse_args()

    t_start = time.time()
    failed = []

    print("=" * 65)
    print("  NEPOOL TDA — Full Pipeline (local)")
    print("=" * 65, flush=True)

    # Step 1: Preprocessing
    if args.from_step <= 1:
        if not run("pipeline/step01_preprocessing.py", [],
                    "Step 01 — Preprocessing"):
            failed.append("step01")
            print("ABORT: step01 failed"); return

    # Step 2: Representations (MOMENT + FE + COMBO)
    if args.from_step <= 2:
        s2_args = ["--skip-moment"] if args.skip_moment else []
        if not run("pipeline/step02_representations.py", s2_args,
                    "Step 02 — Representations"):
            failed.append("step02")
            print("ABORT: step02 failed"); return

    # Steps 3-5: per experiment
    if args.from_step <= 5:
        for exp in EXPERIMENTS:
            name = EXP_NAMES[exp]

            # Step 3: Diffusion Maps
            if args.from_step <= 3:
                if not run("pipeline/step03_pca.py", ["--exp", exp],
                           f"Step 03 — DiffMaps {name}"):
                    failed.append(f"step03 {name}")
                    continue

            # Step 4: ToMATo + Ward
            if args.from_step <= 4:
                run("pipeline/step04_tomato.py", ["--exp", exp],
                    f"Step 04a — ToMATo {name}")
                run("pipeline/step04_ward.py", ["--exp", exp],
                    f"Step 04b — Ward {name}")

            # Step 5: Quality (ToMATo + Ward)
            if args.from_step <= 5:
                run("pipeline/step05_cluster_quality.py",
                    ["--exp", exp],
                    f"Step 05a — Quality ToMATo {name}")
                run("pipeline/step05_cluster_quality.py",
                    ["--exp", exp, "--labels-dir", "step04_ward",
                     "--out-suffix", "_ward"],
                    f"Step 05b — Quality Ward {name}")

    # Final table
    print_table()

    elapsed = time.time() - t_start
    print(f"\nTempo totale: {elapsed/60:.1f} min")
    if failed:
        print(f"Falliti: {failed}")
    else:
        print("Tutti gli step completati.")
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
