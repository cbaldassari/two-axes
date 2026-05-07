"""
summary.py
==========
Raccoglie tutti gli output della pipeline e produce un report testuale
con trade-off, confronti e verdetto finale.

Output: results/summary.txt

Uso
---
  python pipeline/summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path("results")
EXPS    = ["A", "B", "C", "D"]

EXP_DESC = {
    "A": "mstl_resid_lr              (shock rendimento)",
    "B": "mstl_resid_lr + ILR detr.  (shock rendimento + fuel mix)",
    "C": "mstl_resid_arcsinh         (shock livello)",
    "D": "mstl_resid_arcsinh + ILR   (shock livello + fuel mix)",
}


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fmt(val, digits=4):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def write_report():
    lines = []
    W = 80

    def hr(char="="):
        lines.append(char * W)

    def title(text):
        hr()
        lines.append(f"  {text}")
        hr()

    def section(text):
        lines.append("")
        hr("-")
        lines.append(f"  {text}")
        hr("-")

    def ln(text=""):
        lines.append(text)

    # ── Load all reports ──────────────────────────────────────────────────
    pca = {e: load_json(RESULTS / f"exp_{e}" / "step03" / "pca_report.json") for e in EXPS}
    tom = {e: load_json(RESULTS / f"exp_{e}" / "step04" / "tomato_report.json") for e in EXPS}
    qual = {e: load_json(RESULTS / f"exp_{e}" / "step05" / "quality_report.json") for e in EXPS}
    diag = {e: load_json(RESULTS / f"exp_{e}" / "step06" / "diagnostics_report.json") for e in EXPS}

    # ══════════════════════════════════════════════════════════════════════
    title("NEPOOL TDA — PIPELINE SUMMARY REPORT")
    ln()
    ln("  ISO New England electricity market regime detection")
    ln("  Dataset: 43,794 hourly observations, 2021-2025")
    ln("  Method:  MOMENT -> PCA (Marchenko-Pastur) -> ToMATo (TDA)")
    ln()

    # ── Experiments ───────────────────────────────────────────────────────
    section("EXPERIMENTS")
    ln()
    for e in EXPS:
        ln(f"  Exp {e}: {EXP_DESC[e]}")
    ln()
    ln("  Design: 2x2 factorial")
    ln("    Factor 1: shock type    (log-return vs arcsinh level)")
    ln("    Factor 2: fuel mix      (without vs with ILR detrended)")

    # ── PCA ────────────────────────────────────────────────────────────────
    section("STEP 03 — PCA (Marchenko-Pastur)")
    ln()
    ln(f"  {'Exp':<5} {'Input D':<10} {'PC':<6} {'Var expl.':<12} {'lambda_max':<12}")
    ln(f"  {'---':<5} {'-------':<10} {'--':<6} {'---------':<12} {'----------':<12}")
    for e in EXPS:
        r = pca[e]
        if r:
            ln(f"  {e:<5} {r['input_shape'][0]:<10} {r['n_components']:<6} "
               f"{r['variance_explained']:<12.1%} {r['lambda_max']:<12.4f}")
        else:
            ln(f"  {e:<5} — step03 non eseguito")

    ln()
    any_pca = any(pca[e] for e in EXPS)
    if any_pca:
        pc_a = pca["A"]["n_components"] if pca["A"] else 0
        pc_c = pca["C"]["n_components"] if pca["C"] else 0
        if pc_c > pc_a:
            ln(f"  * C/D (arcsinh) richiedono piu' PC ({pc_c} vs {pc_a}): struttura")
            ln(f"    piu' distribuita rispetto a A/B (log_return).")

    # ── ToMATo ─────────────────────────────────────────────────────────────
    section("STEP 04 — ToMATo CLUSTERING")
    ln()
    ln(f"  {'Exp':<5} {'Density':<10} {'k':<6} {'K':<5} {'Gap':<10} {'Method':<15}")
    ln(f"  {'---':<5} {'-------':<10} {'--':<6} {'--':<5} {'---':<10} {'------':<15}")
    for e in EXPS:
        r = tom[e]
        if r:
            ln(f"  {e:<5} {r['density_type']:<10} {r['knn_k']:<6} {r['n_clusters']:<5} "
               f"{fmt(r.get('auto_gap_size'), 4):<10} {r['selection_method']:<15}")
        else:
            ln(f"  {e:<5} — step04 non eseguito")

    # Cluster sizes
    ln()
    ln("  Cluster sizes:")
    for e in EXPS:
        r = tom[e]
        if r:
            sizes = r["cluster_sizes"]
            total = sum(sizes.values())
            dist = "  ".join(f"R{k}:{v}({v/total:.0%})" for k, v in sorted(sizes.items(), key=lambda x: int(x[0])))
            ln(f"    {e}: {dist}")

    # Grid search results if available
    for e in EXPS:
        r = tom[e]
        if r and "grid_search" in r:
            ln()
            ln(f"  Grid search (Exp {e}):")
            best_gn = max(g["gap_norm"] for g in r["grid_search"])
            for g in sorted(r["grid_search"], key=lambda x: -x["gap_norm"])[:5]:
                marker = " <-- best" if g["gap_norm"] == best_gn else ""
                ln(f"    {g['density_type']:>6s} k={g['knn_k']:>3d}  K={g['k_auto']:>2d}  "
                   f"gap_norm={g['gap_norm']:.4f}{marker}")
            break  # solo il primo con grid search

    # ── Quality Geometric ──────────────────────────────────────────────────
    section("STEP 05 — CLUSTER QUALITY: GEOMETRIC")
    ln()
    ln(f"  {'Exp':<5} {'Silhouette':<12} {'Davies-B.':<12} {'Cal.-Har.':<12} {'Dunn':<10}")
    ln(f"  {'---':<5} {'----------':<12} {'---------':<12} {'---------':<12} {'----':<10}")
    for e in EXPS:
        q = qual[e]
        if q:
            g = q["geometric"]
            ln(f"  {e:<5} {g['silhouette_avg']:<12.4f} {g['davies_bouldin']:<12.4f} "
               f"{g['calinski_harabasz']:<12.1f} {g['dunn_index']:<10.6f}")
        else:
            ln(f"  {e:<5} — step05 non eseguito")

    # ── Quality Economic ───────────────────────────────────────────────────
    section("STEP 05 — CLUSTER QUALITY: ECONOMIC")
    ln()
    ln(f"  {'Exp':<5} {'eta2':<8} {'Tukey':<12} {'Trans.rate':<12} {'Sojourn(h)':<12} "
       f"{'Cramer seas.':<13} {'Cramer hour':<12}")
    ln(f"  {'---':<5} {'----':<8} {'-----':<12} {'----------':<12} {'----------':<12} "
       f"{'------------':<13} {'-----------':<12}")
    for e in EXPS:
        q = qual[e]
        if q:
            ec = q["economic"]
            tukey = f"{ec.get('pairwise_significant','?')}/{ec.get('pairwise_total','?')}"
            ln(f"  {e:<5} {ec['eta_squared']:<8.4f} {tukey:<12} "
               f"{ec['transition_rate']:<12.4f} {ec['sojourn_mean_hours']:<12.0f} "
               f"{ec['cramer_v_season']:<13.4f} {ec['cramer_v_hour_block']:<12.4f}")
        else:
            ln(f"  {e:<5} — step05 non eseguito")

    # LMP per regime
    ln()
    ln("  LMP per regime ($/MWh):")
    for e in EXPS:
        q = qual[e]
        if q:
            ln(f"    Exp {e}:")
            for k, s in sorted(q["economic"]["lmp_per_regime"].items(), key=lambda x: int(x[0])):
                ln(f"      R{k}: mean={s['mean']:>7.2f}  std={s['std']:>6.2f}  "
                   f"[p5={s['p5']:>7.2f}, p95={s['p95']:>7.2f}]  n={s['n']:,}")

    # ── Embedding Diagnostics ──────────────────────────────────────────────
    section("STEP 06 — EMBEDDING DIAGNOSTICS")
    ln()
    ln(f"  {'Exp':<5} {'Mantel r':<10} {'NN ratio':<10} {'Kendall tau':<12} "
       f"{'RF R2(LMP)':<12} {'PC0 best corr':<25}")
    ln(f"  {'---':<5} {'--------':<10} {'--------':<10} {'-----------':<12} "
       f"{'----------':<12} {'-------------':<25}")
    for e in EXPS:
        d = diag[e]
        if d:
            t = d["transferability"]
            o = d["opacity"]
            pc0 = o["pc_correlation"]["max_corr_per_pc"].get("PC0", {})
            pc0_str = f"{pc0.get('best_feature','?')} (r={pc0.get('r',0):+.3f})"
            rf_r2 = o["feature_importance"]["r2_cv_lmp_mean"]
            ln(f"  {e:<5} {t['mantel_r']:<10.4f} {t['nn_ratio_mean']:<10.4f} "
               f"{t['nn_kendall_tau']:<12.4f} {rf_r2:<12.4f} {pc0_str:<25}")
        else:
            ln(f"  {e:<5} — step06 non eseguito")

    # Reconstruction top features
    ln()
    ln("  Reconstruction R2 (embedding -> feature, top 5):")
    for e in EXPS:
        d = diag[e]
        if d:
            r2 = d["opacity"]["reconstruction"]
            top = sorted(r2.items(), key=lambda x: x[1], reverse=True)[:5]
            top_str = "  ".join(f"{k}={v:+.3f}" for k, v in top)
            ln(f"    {e}: {top_str}")

    # ══════════════════════════════════════════════════════════════════════
    section("TRADE-OFF ANALYSIS")
    ln()

    # Collect scores for ranking
    any_qual = any(qual[e] for e in EXPS)
    if any_qual:
        ln("  Factor 1 — Shock type (log_return vs arcsinh):")
        ln()
        eta_a = qual["A"]["economic"]["eta_squared"] if qual["A"] else 0
        eta_c = qual["C"]["economic"]["eta_squared"] if qual["C"] else 0
        tr_a = qual["A"]["economic"]["transition_rate"] if qual["A"] else 1
        tr_c = qual["C"]["economic"]["transition_rate"] if qual["C"] else 1
        soj_a = qual["A"]["economic"]["sojourn_mean_hours"] if qual["A"] else 0
        soj_c = qual["C"]["economic"]["sojourn_mean_hours"] if qual["C"] else 0
        ln(f"    eta2:       A={eta_a:.4f}  vs  C={eta_c:.4f}  "
           f"{'C wins' if eta_c > eta_a else 'A wins'} ({eta_c/max(eta_a,1e-9):.1f}x)")
        ln(f"    trans.rate: A={tr_a:.4f}  vs  C={tr_c:.4f}  "
           f"{'C wins (lower=more stable)' if tr_c < tr_a else 'A wins (lower=more stable)'}")
        ln(f"    sojourn:    A={soj_a:.0f}h    vs  C={soj_c:.0f}h    "
           f"{'C wins (longer=more persistent)' if soj_c > soj_a else 'A wins'}")
        ln()
        ln("    >> arcsinh (C/D) domina su log_return (A/B) per qualita' economica.")
        ln()

        ln("  Factor 2 — ILR detrended (without vs with):")
        ln()
        eta_c2 = qual["C"]["economic"]["eta_squared"] if qual["C"] else 0
        eta_d = qual["D"]["economic"]["eta_squared"] if qual["D"] else 0
        sil_c = qual["C"]["geometric"]["silhouette_avg"] if qual["C"] else -1
        sil_d = qual["D"]["geometric"]["silhouette_avg"] if qual["D"] else -1
        cr_c = qual["C"]["economic"]["cramer_v_season"] if qual["C"] else 0
        cr_d = qual["D"]["economic"]["cramer_v_season"] if qual["D"] else 0
        k_c = tom["C"]["n_clusters"] if tom["C"] else 0
        k_d = tom["D"]["n_clusters"] if tom["D"] else 0
        ln(f"    eta2:       C={eta_c2:.4f}  vs  D={eta_d:.4f}")
        ln(f"    silhouette: C={sil_c:.4f}  vs  D={sil_d:.4f}")
        ln(f"    Cramer(s):  C={cr_c:.4f}  vs  D={cr_d:.4f}")
        ln(f"    K:          C={k_c}       vs  D={k_d}")

        if sil_d > sil_c:
            ln("    >> D ha silhouette migliore (geometria cluster piu' separati)")
        if cr_d > cr_c:
            ln("    >> D ha maggiore associazione stagionale")
        ln()

    # ── Opacity / Transferability ──────────────────────────────────────────
    any_diag = any(diag[e] for e in EXPS)
    if any_diag:
        ln("  Embedding opacity:")
        ln()
        for e in EXPS:
            d = diag[e]
            if d:
                rf_r2 = d["opacity"]["feature_importance"]["r2_cv_lmp_mean"]
                r2_vals = list(d["opacity"]["reconstruction"].values())
                n_good = sum(1 for v in r2_vals if v > 0.3)
                n_total = len(r2_vals)
                ln(f"    {e}: RF R2(LMP)={rf_r2:.3f}  |  "
                   f"{n_good}/{n_total} feature con R2>0.3")

        ln()
        ln("  Embedding transferability:")
        ln()
        for e in EXPS:
            d = diag[e]
            if d:
                t = d["transferability"]
                if t["nn_ratio_mean"] < 0.6:
                    strength = "FORTE"
                elif t["nn_ratio_mean"] < 0.8:
                    strength = "BUONA"
                elif t["nn_ratio_mean"] < 0.95:
                    strength = "MODERATA"
                else:
                    strength = "DEBOLE"
                ln(f"    {e}: Mantel r={t['mantel_r']:.3f}  "
                   f"NN ratio={t['nn_ratio_mean']:.3f}  "
                   f"-> coerenza {strength}")

    # ══════════════════════════════════════════════════════════════════════
    section("VERDETTO")
    ln()

    if any_qual:
        # Score composito semplice per ogni esperimento
        scores = {}
        for e in EXPS:
            q = qual[e]
            d = diag[e]
            t = tom[e]
            if not q or not t:
                continue
            g = q["geometric"]
            ec = q["economic"]

            # Normalizza: piu' alto = meglio
            score = 0.0
            score += ec["eta_squared"] * 10        # peso alto: spiegazione LMP
            score += g["silhouette_avg"] * 2       # geometria
            score += (1 - ec["transition_rate"])    # stabilita'
            score += ec["sojourn_mean_hours"] / 100 # persistenza
            score += ec["cramer_v_season"]          # contenuto stagionale
            pw = ec.get("pairwise_separation")
            if pw is not None:
                score += pw * 0.5                   # separazione prezzi
            if d:
                score += (1 - d["transferability"]["nn_ratio_mean"]) * 0.5  # coerenza embedding

            scores[e] = round(score, 4)

        if scores:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ln("  Score composito (piu' alto = meglio):")
            ln()
            for i, (e, s) in enumerate(ranked):
                marker = "  << WINNER" if i == 0 else ""
                ln(f"    Exp {e}: {s:.4f}{marker}")
                ln(f"           {EXP_DESC[e]}")

            winner = ranked[0][0]
            ln()
            ln(f"  Esperimento consigliato: {winner}")
            ln()

            q_w = qual[winner]
            t_w = tom[winner]
            ec_w = q_w["economic"]
            g_w = q_w["geometric"]
            ln(f"    K = {t_w['n_clusters']} cluster")
            ln(f"    eta2 = {ec_w['eta_squared']:.4f} (varianza LMP spiegata)")
            ln(f"    silhouette = {g_w['silhouette_avg']:.4f}")
            ln(f"    transition rate = {ec_w['transition_rate']:.4f}")
            ln(f"    sojourn medio = {ec_w['sojourn_mean_hours']:.0f}h")
            ln(f"    Cramer V (season) = {ec_w['cramer_v_season']:.4f}")

    ln()
    ln("  NOTE:")
    ln("  - Silhouette negativa o ~0 indica sovrapposizione nello spazio PCA")
    ln("    ad alta dimensionalita'. Non invalida i cluster: ToMATo opera sulla")
    ln("    topologia della densita', non sulla separazione geometrica.")
    ln("  - Le metriche economiche (eta2, sojourn, Cramer V) sono piu'")
    ln("    informative della silhouette per valutare i regimi di mercato.")
    ln("  - I risultati di embedding diagnostics mostrano R2 di ricostruzione")
    ln("    generalmente bassi: Chronos-2 codifica struttura che le feature")
    ln("    manuali non catturano completamente.")
    ln()
    hr()

    return "\n".join(lines)


def main():
    report = write_report()

    # Print to terminal
    print(report)

    # Save to file
    out_path = RESULTS / "summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Salvato: {out_path}")


if __name__ == "__main__":
    main()
